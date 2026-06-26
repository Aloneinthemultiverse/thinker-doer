# thinker-doer | HumanEval benchmark on Kaggle T4x2
# Phi-4 (thinker) plans -> Qwythos-9B (doer) writes the function -> run hidden test -> retry.
# Reports pass@1 for COUPLE vs DOER-SOLO over the first N problems + the coupling lift.
# Must be launched from the Kaggle UI via "Save & Run All" to get T4x2 (API push forces 1 GPU).
import gzip, json, os, re, subprocess, sys, tempfile, time, urllib.request

# ----------------------------- config -----------------------------------------
PHI4_REPO = os.environ.get("PHI4_REPO", "bartowski/phi-4-GGUF")
QWY_REPO  = os.environ.get("QWY_REPO",  "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")
QUANT     = os.environ.get("TD_QUANT", "Q4_K_M")
PHI4_QUANT = os.environ.get("PHI4_QUANT", "Q4_K_M")  # T4x2 has room; use full Q4 for quality
N_PROBLEMS = int(os.environ.get("TD_N", "40"))       # first N HumanEval problems
RETRY = os.environ.get("TD_RETRY", "1") == "1"       # verify->fix one retry

def sh(c): print("$",c,flush=True); return subprocess.run(c,shell=True)

# ----------------------------- 1. deps (prebuilt CUDA wheel) ------------------
sh("pip -q install huggingface_hub")
sh("pip -q install 'llama-cpp-python[server]' "
   "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
from huggingface_hub import hf_hub_download, list_repo_files

def resolve_gguf(repo, quant=QUANT):
    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    if not files: raise RuntimeError(f"no gguf in {repo}")
    pick = next((f for f in files if quant.lower() in f.lower()), None)
    if not pick:
        small = [f for f in files if "q4" in f.lower() or "q3" in f.lower()]
        pick = sorted(small)[0] if small else sorted(files, key=len)[0]
    print(f"  {repo} -> {pick}", flush=True)
    return hf_hub_download(repo, pick)

print("resolving GGUFs...", flush=True)
PHI4 = resolve_gguf(PHI4_REPO, PHI4_QUANT); QWY = resolve_gguf(QWY_REPO)

# ----------------------------- 2. serve both ----------------------------------
try: NG=int(subprocess.run("nvidia-smi -L",shell=True,capture_output=True,text=True).stdout.count("GPU "))
except Exception: NG=1
g_qwy = "1" if NG>=2 else "0"; print(f"GPUs={NG} -> qwythos on cuda{g_qwy}, phi4 on cuda0",flush=True)
def serve(gpu, gguf, port, ctx):
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    return subprocess.Popen(["python","-m","llama_cpp.server","--model",gguf,
        "--host","0.0.0.0","--port",str(port),"--n_gpu_layers","-1",
        "--n_ctx",str(ctx),"--n_batch","64","--n_ubatch","64"], env=env)
QCTX = int(os.environ.get("TD_QCTX", "8192" if NG>=2 else "4096"))
PCTX = int(os.environ.get("TD_PCTX", "8192" if NG>=2 else "4096"))
serve(g_qwy, QWY, 8080, QCTX)
serve("0", PHI4, 8081, PCTX)
for port in (8080, 8081):
    up=False
    for _ in range(240):
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",timeout=5).read(); up=True; break
        except Exception: time.sleep(5)
    print(f"  :{port} {'UP' if up else 'DOWN'}",flush=True)
    if not up: raise SystemExit(f"server :{port} down")

# ----------------------------- 3. clients -------------------------------------
DOER_EP="http://127.0.0.1:8080/v1/chat/completions"; THINK_EP="http://127.0.0.1:8081/v1/chat/completions"
def _chat(ep,msgs,temp,mt):
    body=json.dumps({"messages":msgs,"temperature":temp,"top_p":0.95,"max_tokens":mt,"stream":False}).encode()
    req=urllib.request.Request(ep,data=body,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=600).read())["choices"][0]["message"]["content"]
def think(m,mt=1536): return _chat(THINK_EP,m,0.3,mt)
def doer(m,mt=1024):  return _chat(DOER_EP,m,0.2,mt)

# ----------------------------- 4. HumanEval data ------------------------------
DATA="/kaggle/working/HumanEval.jsonl.gz"
if not os.path.exists(DATA):
    urllib.request.urlretrieve(
        "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz", DATA)
PROBS=[json.loads(l) for l in gzip.open(DATA,"rt",encoding="utf-8")][:N_PROBLEMS]
print(f"loaded {len(PROBS)} HumanEval problems",flush=True)

# ----------------------------- 5. couple/solo codegen -------------------------
def extract_code(text):
    text=re.sub(r"<think>.*?</think>","",text,flags=re.S); text=re.sub(r"<think>.*","",text,flags=re.S)
    m=re.search(r"```(?:python)?\n(.*?)```",text,re.S)
    return (m.group(1) if m else text).strip()

def _think_plan(prompt, prior_fail=""):
    usr=("Plan the implementation of this Python function. State the algorithm, edge cases, and "
         "the exact return behavior. Do NOT write the final code block.\n\n"+prompt)
    if prior_fail: usr+=("\n\nThe previous attempt FAILED its tests:\n"+prior_fail+
                         "\nDiagnose the bug and state the corrected approach.")
    return think([{"role":"system","content":"You are an expert Python architect. Plan precisely; "
                   "be exact about return values, types, ordering, and edge cases."},
                  {"role":"user","content":usr}])

def _doer_from_plan(prompt, plan):
    usr=("Implement the function from this plan. Return ONLY the complete function definition "
         "(signature + body) in one ```python block. No explanation, no tests.\n\n"
         f"Plan:\n{plan[-3000:]}\n\nFunction to complete:\n{prompt}")
    return extract_code(doer([{"role":"system","content":"You output only clean Python code."},
                              {"role":"user","content":usr}]))

def _doer_solo(prompt, prior_fail=""):
    usr=("Complete this Python function. Return ONLY the full function definition (signature + "
         "body) in one ```python block. No explanation, no tests.\n\n"+prompt)
    if prior_fail: usr+=("\n\nYour previous attempt FAILED its tests:\n"+prior_fail+
                         "\nReturn a corrected full function.")
    return extract_code(doer([{"role":"system","content":"You are an expert Python programmer."},
                              {"role":"user","content":usr}]))

def generate(prompt, couple, prior_fail=""):
    if couple:
        plan=_think_plan(prompt, prior_fail)
        code=_doer_from_plan(prompt, plan)
        return code or _doer_solo(prompt, prior_fail)
    return _doer_solo(prompt, prior_fail)

# ----------------------------- 6. verify --------------------------------------
def _header(prompt, entry):
    m=re.search(rf"^def\s+{re.escape(entry)}\s*\(", prompt, re.M)
    return prompt[:m.start()] if m else ""
def _ensure_sig(code, prompt, entry):
    return code if re.search(rf"def\s+{re.escape(entry)}\s*\(", code) else prompt+"\n"+code
def run_test(code, test, entry, header="", timeout=15):
    prog=header+"\n"+code+"\n\n"+test+f"\n\ncheck({entry})\nprint('__PASSED__')\n"
    with tempfile.NamedTemporaryFile("w",suffix=".py",delete=False,encoding="utf-8") as f:
        f.write(prog); path=f.name
    try:
        p=subprocess.run([sys.executable,path],capture_output=True,text=True,timeout=timeout)
        return "__PASSED__" in p.stdout, (p.stderr or p.stdout)[-300:]
    except subprocess.TimeoutExpired: return False,"timeout"
    except Exception as e: return False,str(e)[:300]
    finally:
        try: os.unlink(path)
        except OSError: pass

def solve_one(p, couple):
    entry=p["entry_point"]; header=_header(p["prompt"],entry)
    code=_ensure_sig(generate(p["prompt"],couple), p["prompt"], entry)
    ok,err=run_test(code,p["test"],entry,header=header)
    used=False
    if not ok and RETRY:
        used=True
        code=_ensure_sig(generate(p["prompt"],couple,prior_fail=err), p["prompt"], entry)
        ok,err=run_test(code,p["test"],entry,header=header)
    return ok,used

# ----------------------------- 7. run both ------------------------------------
def run(couple):
    label="COUPLE (Phi-4 plans -> Qwythos)" if couple else "DOER-SOLO (Qwythos)"
    print(f"\n========== {label} ==========",flush=True)
    passed=saves=0; t0=time.time()
    for i,p in enumerate(PROBS,1):
        try: ok,used=solve_one(p,couple)
        except Exception as e: ok,used=False,False; print("  err",str(e)[:60])
        passed+=ok; saves+=ok and used
        print(f"[{i}/{len(PROBS)}] {p['task_id']:<14} {'PASS' if ok else 'FAIL'}"
              f"{' (saved by retry)' if ok and used else ''}",flush=True)
    pct=100*passed/len(PROBS)
    print(f"\n{label} pass@1: {passed}/{len(PROBS)} = {pct:.1f}%  "
          f"(retry rescued {saves}, {time.time()-t0:.0f}s)\n",flush=True)
    return pct

c=run(couple=True); s=run(couple=False)
print(f"\n*** HUMANEVAL[{N_PROBLEMS}] COUPLING LIFT = {c-s:+.1f} points  "
      f"(couple {c:.1f} - solo {s:.1f}) ***")
