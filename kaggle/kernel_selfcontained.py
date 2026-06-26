# thinker-doer | self-contained Kaggle T4x2 kernel
# Serves Phi-4 (GPU0:8081) + Qwythos-9B (GPU1:8080) via llama.cpp, runs couple vs doer-solo.
# All harness code inlined (repo has no remote). GGUFs auto-resolved (picks Q4_K_M).
import json, os, re, subprocess, tempfile, time, urllib.request

# ----------------------------- config (edit repo IDs if needed) ---------------
PHI4_REPO = os.environ.get("PHI4_REPO", "bartowski/phi-4-GGUF")  # full-weights repo has no GGUF
QWY_REPO  = os.environ.get("QWY_REPO",  "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")
QUANT     = os.environ.get("TD_QUANT", "Q4_K_M")

def sh(c): print("$",c,flush=True); return subprocess.run(c,shell=True)

# ----------------------------- 1. deps + llama.cpp ----------------------------
sh("pip -q install huggingface_hub")
from huggingface_hub import hf_hub_download, list_repo_files

def resolve_gguf(repo):
    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    if not files: raise RuntimeError(f"no gguf in {repo}")
    # prefer the requested quant; else the alphabetically-first Q4_* ; else just refuse
    # the giant F16/Q8 (would OOM on a 16GB T4) — fall back to smallest by name heuristic.
    pick = next((f for f in files if QUANT.lower() in f.lower()), None)
    if not pick:
        q4 = [f for f in files if "q4" in f.lower()]
        pick = sorted(q4)[0] if q4 else sorted(files, key=len)[0]
    print(f"  {repo} -> {pick}", flush=True)
    return hf_hub_download(repo, pick)

LS = "/kaggle/working/llama.cpp/build/bin/llama-server"
if not os.path.exists(LS):
    # FindCUDAToolkit can't locate CUDA::cuda_driver on Kaggle (libcuda lives in stubs/ only).
    sh("ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so")
    sh("cd /kaggle/working && git clone --depth 1 https://github.com/ggerganov/llama.cpp")
    sh("cd /kaggle/working/llama.cpp && cmake -B build -DGGML_CUDA=ON "
       "-DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build --config Release -j --target llama-server")
    assert os.path.exists(LS), "llama-server build FAILED (see cmake/link errors above)"

print("resolving GGUFs...", flush=True)
PHI4 = resolve_gguf(PHI4_REPO); QWY = resolve_gguf(QWY_REPO)

# ----------------------------- 2. serve both, one per GPU ---------------------
try: NG=int(subprocess.run("nvidia-smi -L",shell=True,capture_output=True,text=True).stdout.count("GPU "))
except Exception: NG=1
g_qwy = "1" if NG>=2 else "0"; print(f"GPUs={NG} -> qwythos on cuda{g_qwy}, phi4 on cuda0",flush=True)
subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g_qwy} {LS} -m {QWY} --host 0.0.0.0 --port 8080 -ngl 99 -c 32768 --parallel 1", shell=True)
subprocess.Popen(f"CUDA_VISIBLE_DEVICES=0 {LS} -m {PHI4} --host 0.0.0.0 --port 8081 -ngl 99 -c 16384 --parallel 1", shell=True)
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
