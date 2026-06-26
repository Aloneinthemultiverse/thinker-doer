# thinker-doer | SWE-bench Lite inference on Kaggle T4x2
# Real SWE-bench Lite (25-instance slice). For each instance: clone repo@base_commit,
# localize buggy files (git grep on issue identifiers), couple/solo rewrites them, git diff
# -> model_patch. Writes preds_couple.json + preds_solo.json for the sb-cli CLOUD grader.
#
# Honest scaffold limits (so the score reads as a conservative FLOOR, both modes equal):
#   - one-shot, no agentic iteration / no test feedback during solve
#   - edits the top localized .py file(s) only; misses cross-file & very-large-file fixes
#   - 9B doer => expect a LOW absolute score; the couple-vs-solo DELTA is the real signal
#
# Launch from Kaggle UI via "Save & Run All" (T4x2). Then submit the two files:
#   sb-cli submit swe-bench_lite test --predictions_path preds_couple.json --run_id td_couple
#   sb-cli submit swe-bench_lite test --predictions_path preds_solo.json   --run_id td_solo
import json, os, re, subprocess, time, urllib.request

PHI4_REPO = os.environ.get("PHI4_REPO", "bartowski/phi-4-GGUF")
QWY_REPO  = os.environ.get("QWY_REPO",  "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")
QUANT     = os.environ.get("TD_QUANT", "Q4_K_M")
PHI4_QUANT= os.environ.get("PHI4_QUANT", "Q4_K_M")
START     = int(os.environ.get("TD_START", "0"))  # chunk offset: 0, 25, 50, ... (edit per run)
N_INST    = int(os.environ.get("TD_N", "25"))     # SWE-bench Lite chunk size
MAX_FILES = int(os.environ.get("TD_MAXFILES", "2"))   # localized files to edit per instance
MAX_LINES = int(os.environ.get("TD_MAXLINES", "800")) # skip rewriting files larger than this

def sh(c,**k): print("$",c,flush=True); return subprocess.run(c,shell=True,**k)

# ----------------------------- 1. deps + serve --------------------------------
sh("pip -q install huggingface_hub datasets")
sh("pip -q install 'llama-cpp-python[server]' "
   "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
from huggingface_hub import hf_hub_download, list_repo_files

def resolve_gguf(repo, quant=QUANT):
    files=[f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    if not files: raise RuntimeError(f"no gguf in {repo}")
    pick=next((f for f in files if quant.lower() in f.lower()), None)
    if not pick:
        small=[f for f in files if "q4" in f.lower() or "q3" in f.lower()]
        pick=sorted(small)[0] if small else sorted(files,key=len)[0]
    print(f"  {repo} -> {pick}",flush=True); return hf_hub_download(repo,pick)

print("resolving GGUFs...",flush=True)
PHI4=resolve_gguf(PHI4_REPO,PHI4_QUANT); QWY=resolve_gguf(QWY_REPO)

try: NG=int(subprocess.run("nvidia-smi -L",shell=True,capture_output=True,text=True).stdout.count("GPU "))
except Exception: NG=1
g_qwy="1" if NG>=2 else "0"; print(f"GPUs={NG} -> qwythos cuda{g_qwy}, phi4 cuda0",flush=True)

def _health(port):
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",timeout=5).read(); return True
    except Exception: return False
def serve_fallback(gpu, gguf, port, ctx_list):
    """Doer gets the biggest context that actually fits — try high, step down on OOM."""
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    for ctx in ctx_list:
        print(f"  trying :{port} ctx={ctx} ...",flush=True)
        p=subprocess.Popen(["python","-m","llama_cpp.server","--model",gguf,"--host","0.0.0.0",
            "--port",str(port),"--n_gpu_layers","-1","--n_ctx",str(ctx),
            "--n_batch","64","--n_ubatch","64"], env=env)
        for _ in range(120):
            if _health(port): print(f"  :{port} UP at ctx={ctx}",flush=True); return ctx
            if p.poll() is not None: break          # server died (OOM) -> next ctx
            time.sleep(5)
        try: p.kill()
        except Exception: pass
        time.sleep(3)
    raise SystemExit(f":{port} never came up at any ctx")

# Qwythos (doer, reads the repo): push to 200K, step down to what the 16GB card holds.
QCTX=serve_fallback(g_qwy, QWY, 8080, [200000,131072,98304,65536,32768])
# Phi-4 (critic/planner): small ctx is plenty.
serve_fallback("0", PHI4, 8081, [16384,8192])
print(f"doer ctx = {QCTX}",flush=True)

# ----------------------------- 2. clients -------------------------------------
DOER_EP="http://127.0.0.1:8080/v1/chat/completions"; THINK_EP="http://127.0.0.1:8081/v1/chat/completions"
def _chat(ep,msgs,temp,mt):
    body=json.dumps({"messages":msgs,"temperature":temp,"top_p":0.95,"max_tokens":mt,"stream":False}).encode()
    req=urllib.request.Request(ep,data=body,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=900).read())["choices"][0]["message"]["content"]
def think(m): return _chat(THINK_EP,m,0.3,2048)
def doer(m,mt=4096):  return _chat(DOER_EP,m,0.2,mt)

# ----------------------------- 3. SWE-bench Lite data -------------------------
from datasets import load_dataset
DS=load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
END=min(START+N_INST,len(DS))
INSTANCES=[DS[i] for i in range(START,END)]
print(f"SWE-bench Lite chunk [{START}:{END}] of {len(DS)} -> {len(INSTANCES)} instances",flush=True)

# ----------------------------- 4. repo checkout + localize --------------------
WORK="/kaggle/working/repos"; os.makedirs(WORK,exist_ok=True)
_STOP=set("the and for with this that from have been will your you not are was but all can has "
          "when then they them what which while should would could into over under code file "
          "test tests error issue bug fix python self def return None True False import class".split())

def checkout(repo, commit):
    dest=os.path.join(WORK, repo.replace("/","__"))
    if os.path.isdir(os.path.join(dest,".git")):
        sh(f"cd {dest} && git checkout -q -f {commit} 2>/dev/null || git fetch -q --depth 1 origin {commit} && git checkout -q -f {commit}")
        return dest
    url=f"https://github.com/{repo}.git"
    rc=sh(f"git init -q {dest} && cd {dest} && git remote add origin {url} && "
          f"git fetch -q --depth 1 origin {commit} && git checkout -q -f FETCH_HEAD").returncode
    if rc!=0:   # shallow-by-sha unsupported -> full clone fallback
        sh(f"rm -rf {dest} && git clone -q {url} {dest} && cd {dest} && git checkout -q -f {commit}")
    return dest

def identifiers(text):
    toks=re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text)
    seen=[];
    for t in toks:
        if t.lower() not in _STOP and t not in seen: seen.append(t)
    return seen[:25]

def localize(repo_dir, problem):
    """Rank .py files by how many issue-identifiers they contain (git grep)."""
    score={}
    for sym in identifiers(problem):
        out=subprocess.run(f"cd {repo_dir} && git grep -l -F -- {json.dumps(sym)} -- '*.py'",
                           shell=True,capture_output=True,text=True).stdout
        for f in out.split():
            if "/test" in f or f.startswith("test"): continue   # don't edit tests
            score[f]=score.get(f,0)+1
    ranked=sorted(score, key=lambda f:-score[f])
    keep=[]
    for f in ranked:
        try: n=sum(1 for _ in open(os.path.join(repo_dir,f),encoding="utf-8",errors="ignore"))
        except OSError: continue
        if n<=MAX_LINES: keep.append(f)
        if len(keep)>=MAX_FILES: break
    return keep

# ----------------------------- 5. generate fix --------------------------------
_CODE=re.compile(r"```(?:python)?\s*(.*?)```",re.DOTALL)
def _block(reply, original):
    bs=[b for b in _CODE.findall(reply) if "def " in b or "class " in b or "import " in b]
    return max(bs,key=len).strip()+"\n" if bs else None

def fix_file(problem, path, src, couple):
    if couple:
        plan=think([{"role":"system","content":"You are an expert software engineer. Diagnose the "
                     "root cause from the issue and the file, and state the precise minimal fix. "
                     "Do not write the whole file."},
                    {"role":"user","content":f"GitHub issue:\n{problem[:4000]}\n\nFile {path}:\n"
                     f"```python\n{src}\n```\nWhat is the bug and the exact minimal change?"}])
        usr=(f"Issue:\n{problem[:2500]}\n\nExpert diagnosis:\n{plan[-2500:]}\n\nFile {path}:\n"
             f"```python\n{src}\n```\nReturn the COMPLETE corrected file in one ```python block.")
    else:
        usr=(f"GitHub issue:\n{problem[:4000]}\n\nFile {path}:\n```python\n{src}\n```\n"
             f"Fix the bug described in the issue. Return the COMPLETE corrected file in one "
             f"```python block.")
    reply=doer([{"role":"system","content":"You output only the complete corrected file in one "
                 "```python code block. Preserve everything except the necessary fix."},
                {"role":"user","content":usr}])
    return _block(reply, src)

def make_patch(repo_dir, problem, files, couple):
    sh(f"cd {repo_dir} && git checkout -q -f .")          # clean slate
    for f in files:
        ap=os.path.join(repo_dir,f)
        try: src=open(ap,encoding="utf-8",errors="ignore").read()
        except OSError: continue
        new=fix_file(problem, f, src, couple)
        if new and new.strip()!=src.strip():
            open(ap,"w",encoding="utf-8").write(new)
    patch=subprocess.run(f"cd {repo_dir} && git diff",shell=True,capture_output=True,text=True).stdout
    sh(f"cd {repo_dir} && git checkout -q -f .")          # reset for next mode
    return patch

# ----------------------------- 6. run -----------------------------------------
def run():
    couple_preds=[]; solo_preds=[]
    for i,inst in enumerate(INSTANCES,1):
        iid=inst["instance_id"]; repo=inst["repo"]; commit=inst["base_commit"]; prob=inst["problem_statement"]
        t0=time.time()
        try:
            rd=checkout(repo,commit); files=localize(rd,prob)
        except Exception as e:
            print(f"[{i}/{len(INSTANCES)}] {iid:<28} SKIP (setup: {str(e)[:50]})",flush=True)
            couple_preds.append({"instance_id":iid,"model_name_or_path":"td-couple","model_patch":""})
            solo_preds.append({"instance_id":iid,"model_name_or_path":"td-solo","model_patch":""})
            continue
        if not files:
            print(f"[{i}/{len(INSTANCES)}] {iid:<28} no-localize",flush=True); files=[]
        try: cp=make_patch(rd,prob,files,couple=True) if files else ""
        except Exception as e: cp=""; print("  couple err",str(e)[:50])
        try: sp=make_patch(rd,prob,files,couple=False) if files else ""
        except Exception as e: sp=""; print("  solo err",str(e)[:50])
        couple_preds.append({"instance_id":iid,"model_name_or_path":"td-couple","model_patch":cp})
        solo_preds.append({"instance_id":iid,"model_name_or_path":"td-solo","model_patch":sp})
        print(f"[{i}/{len(INSTANCES)}] {iid:<28} files={len(files)} "
              f"couple={'Y' if cp.strip() else '-'} solo={'Y' if sp.strip() else '-'} "
              f"{time.time()-t0:.0f}s",flush=True)
    cfile=f"/kaggle/working/preds_couple_{START}_{END}.json"
    sfile=f"/kaggle/working/preds_solo_{START}_{END}.json"
    json.dump(couple_preds, open(cfile,"w")); json.dump(solo_preds, open(sfile,"w"))
    nc=sum(1 for p in couple_preds if p["model_patch"].strip())
    ns=sum(1 for p in solo_preds   if p["model_patch"].strip())
    print(f"\nwrote {os.path.basename(cfile)} ({nc} non-empty) + {os.path.basename(sfile)} ({ns} non-empty)")
    print(f"NEXT CHUNK: set TD_START={END} and re-run.  After all chunks: merge the JSON lists and submit:")
    print("  sb-cli submit swe-bench_lite test --predictions_path preds_couple_ALL.json --run_id td_couple")
    print("  sb-cli submit swe-bench_lite test --predictions_path preds_solo_ALL.json   --run_id td_solo")

run()
