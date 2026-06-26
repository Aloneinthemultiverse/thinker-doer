# thinker-doer | self-contained Kaggle T4x2 kernel
# Serves Phi-4 (GPU0:8081) + Qwythos-9B (GPU1:8080) via llama.cpp, runs couple vs doer-solo.
# All harness code inlined (repo has no remote). GGUFs auto-resolved (picks Q4_K_M).
import json, os, re, subprocess, tempfile, time, urllib.request

# ----------------------------- config (edit repo IDs if needed) ---------------
PHI4_REPO = os.environ.get("PHI4_REPO", "bartowski/phi-4-GGUF")  # full-weights repo has no GGUF
QWY_REPO  = os.environ.get("QWY_REPO",  "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")
QUANT     = os.environ.get("TD_QUANT", "Q4_K_M")
# Kaggle assigns a SINGLE 16GB GPU (P100), not T4x2. To fit both models on one card,
# run Phi-4 at a smaller quant and keep contexts tiny (tasks are small bug-fixes).
PHI4_QUANT = os.environ.get("PHI4_QUANT", "Q3_K_M")  # ~7GB vs ~9GB for Q4_K_M

def sh(c): print("$",c,flush=True); return subprocess.run(c,shell=True)

# ----------------------------- 1. deps (prebuilt CUDA wheel — no compile) ------
# Building llama.cpp from source on Kaggle fails: FindCUDAToolkit can't create the
# CUDA::cuda_driver target (libcuda is stub-only). Use the prebuilt llama-cpp-python
# CUDA 12.x wheel instead — it ships a precompiled server, zero compilation.
sh("pip -q install huggingface_hub")
sh("pip -q install 'llama-cpp-python[server]' "
   "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
from huggingface_hub import hf_hub_download, list_repo_files

def resolve_gguf(repo, quant=QUANT):
    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    if not files: raise RuntimeError(f"no gguf in {repo}")
    # prefer the requested quant; else any q4/q3; else smallest-named (avoid giant F16/Q8).
    pick = next((f for f in files if quant.lower() in f.lower()), None)
    if not pick:
        small = [f for f in files if "q4" in f.lower() or "q3" in f.lower()]
        pick = sorted(small)[0] if small else sorted(files, key=len)[0]
    print(f"  {repo} -> {pick}", flush=True)
    return hf_hub_download(repo, pick)

print("resolving GGUFs...", flush=True)
PHI4 = resolve_gguf(PHI4_REPO, PHI4_QUANT); QWY = resolve_gguf(QWY_REPO)

# ----------------------------- 2. serve both ----------------------------------
# llama_cpp.server is OpenAI-compatible (/v1/chat/completions). Kaggle gives ONE 16GB
# GPU, so both share it; keep contexts small so the two KV caches fit alongside weights.
try: NG=int(subprocess.run("nvidia-smi -L",shell=True,capture_output=True,text=True).stdout.count("GPU "))
except Exception: NG=1
g_qwy = "1" if NG>=2 else "0"; print(f"GPUs={NG} -> qwythos on cuda{g_qwy}, phi4 on cuda0",flush=True)
def serve(gpu, gguf, port, ctx):
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    # small batch -> small compute-graph buffer; prior run OOM'd in graph_compute at batch 512
    # on the single 16GB GPU even though weights+KV fit.
    return subprocess.Popen(["python","-m","llama_cpp.server","--model",gguf,
        "--host","0.0.0.0","--port",str(port),"--n_gpu_layers","-1",
        "--n_ctx",str(ctx),"--n_batch","64","--n_ubatch","64"], env=env)
QCTX = int(os.environ.get("TD_QCTX", "4096" if NG<2 else "32768"))
PCTX = int(os.environ.get("TD_PCTX", "4096" if NG<2 else "16384"))
serve(g_qwy, QWY, 8080, QCTX)
serve("0", PHI4, 8081, PCTX)
for port in (8080, 8081):
    up=False
    for _ in range(180):
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",timeout=5).read(); up=True; break
        except Exception: time.sleep(5)
    print(f"  :{port} {'UP' if up else 'DOWN'}",flush=True)
    if not up: raise SystemExit(f"server :{port} down")

# ----------------------------- 3. inlined harness -----------------------------
DOER_EP="http://127.0.0.1:8080/v1/chat/completions"; THINK_EP="http://127.0.0.1:8081/v1/chat/completions"
_CODE=re.compile(r"```(?:python)?\s*(.*?)```",re.DOTALL); MAXR=4

def _chat(ep,msgs,temp,mt):
    body=json.dumps({"messages":msgs,"temperature":temp,"top_p":0.95,"max_tokens":mt,"stream":False}).encode()
    req=urllib.request.Request(ep,data=body,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=600).read())["choices"][0]["message"]["content"]
def think(m): return _chat(THINK_EP,m,0.4,3072)
def doer(m):  return _chat(DOER_EP,m,0.2,1536)
def block(r):
    bs=[b.strip() for b in _CODE.findall(r) if "def " in b]
    return (max(bs,key=lambda b:(b.count("def "),len(b)))+"\n") if bs else None

SB=""  # sandbox set per-task
def rt():
    p=subprocess.run("python -m pytest -q",shell=True,cwd=SB,capture_output=True,text=True,timeout=60)
    return (p.stdout+p.stderr).strip()
def passed(o=None):
    o=o if o is not None else rt(); l=o.lower()
    return "fail" not in l and "error" not in l and ("passed" in l or "ok" in l)
def hint(o):
    if not o or "FAIL" not in o.upper(): return ""
    ls=[x.strip() for x in o.splitlines() if x.strip()]
    return next((x for x in reversed(ls) if "Error" in x or x.startswith("assert") or "!=" in x),"")[:200]

def solve(task,rel="buggy.py"):
    src=open(os.path.join(SB,rel)).read(); fail=rt(); last=None
    if passed(fail): return True,0,"already"
    for rnd in range(1,MAXR+1):
        usr=f"{task}\n\nCurrent file:\n```python\n{src}```\n"
        if fail: usr+=f"\nTests FAIL:\n{fail[:700]}\n"
        if hint(fail): usr+=f"\nFocus: `{hint(fail)}`. Reason about exact expected value."
        usr+="\nDecompose, reason about EVERY bug, state the exact fix."
        cot=think([{"role":"system","content":"Expert Python architect+debugger. Precise about return values."},{"role":"user","content":usr}])
        rep=doer([{"role":"system","content":"Output ONLY the complete corrected file in one ```python block."},{"role":"user","content":f"Task: {task}\n\nExpert plan:\n{cot[-2500:]}\n\nOriginal:\n```python\n{src}```\nWrite the FULL corrected file."}])
        body=block(rep)
        if body:
            open(os.path.join(SB,rel),"w").write(body); out=rt()
            if passed(out): return True,rnd,"coupling"
            if body.strip()==(last or "").strip(): break
            last=body; src=body; fail=out
        else: fail="[no code block]"
    return passed(),MAXR,"exhausted"

def solo(task,rel="buggy.py"):
    src=open(os.path.join(SB,rel)).read()
    rep=doer([{"role":"system","content":"Output ONLY the complete corrected file in one ```python block."},{"role":"user","content":f"Task: {task}\n\nFile:\n```python\n{src}```\nWrite the FULL corrected file."}])
    b=block(rep)
    if b: open(os.path.join(SB,rel),"w").write(b)
    return passed(),1,"doer-solo"

# ----------------------------- 4. tasks (inline seed; expand later) -----------
TASKS={
 "running_max":(
  "Fix running_max in buggy.py so all tests pass.",
  "def running_max(nums):\n    out=[]; m=0\n    for n in nums:\n        if n>m: m=n\n        out.append(m)\n    return out\n",
  "from buggy import running_max\ndef test_b(): assert running_max([1,2,1,3])==[1,2,2,3]\ndef test_n(): assert running_max([-3,-5,-2])==[-3,-3,-2]\ndef test_s(): assert running_max([7])==[7]\n"),
 "dedup_keep_order":(
  "Fix dedup so it removes duplicates but KEEPS first-seen order.",
  "def dedup(xs):\n    return list(set(xs))\n",
  "from buggy import dedup\ndef test_o(): assert dedup([3,1,3,2,1])==[3,1,2]\ndef test_e(): assert dedup([])==[]\n"),
 "fizzbuzz":(
  "Fix fizzbuzz(n) -> list 1..n with Fizz/Buzz/FizzBuzz rules.",
  "def fizzbuzz(n):\n    r=[]\n    for i in range(1,n+1):\n        if i%3==0: r.append('Fizz')\n        elif i%5==0: r.append('Buzz')\n        else: r.append(i)\n    return r\n",
  "from buggy import fizzbuzz\ndef test_fb(): assert fizzbuzz(15)[-1]=='FizzBuzz'\ndef test_n(): assert fizzbuzz(2)==[1,2]\n"),
}

def run(mode):
    global SB; fn=solo if mode=="solo" else solve; ok=0
    for name,(prompt,bug,test) in TASKS.items():
        SB=tempfile.mkdtemp(prefix="td_")
        open(os.path.join(SB,"buggy.py"),"w").write(bug)
        open(os.path.join(SB,"test_buggy.py"),"w").write(test)
        try: p,r,via=fn(prompt)
        except Exception as e: p,r,via=False,0,f"err:{str(e)[:40]}"
        ok+=p; print(f"  {'PASS' if p else 'fail':4} {name:20} {via:10} r{r}",flush=True)
    pct=100*ok/len(TASKS); print(f"\n{mode.upper()}: {ok}/{len(TASKS)} = {pct:.1f}%\n",flush=True); return pct

print("\n========== COUPLE =========="); c=run("couple")
print("========== DOER-SOLO =========="); s=run("solo")
print(f"\n*** COUPLING LIFT = {c-s:+.1f} points  (couple {c:.1f} - solo {s:.1f}) ***")
