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

# ----------------------------- 5. generate fix (SEARCH/REPLACE, never destructive) ----
# The OLD approach asked for a full-file rewrite; the 9B truncated it, so the git diff
# DELETED the file -> every test errored ("failed run"). Fix: minimal SEARCH/REPLACE edits.
# The model copies the exact lines to change and their replacement; we splice them in and
# REJECT the edit unless the result still parses. Worst case = empty patch (unresolved),
# never a destroyed file (failed run).
import ast
_SR=re.compile(r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE", re.DOTALL)
_FMT=("Respond ONLY with one or more edit blocks in this EXACT format:\n"
      "<<<<<<< SEARCH\n<lines copied verbatim from the file, including indentation>\n"
      "=======\n<the replacement lines>\n>>>>>>> REPLACE\n"
      "Keep SEARCH blocks small (just the lines that change). Copy them EXACTLY.")

def _apply_blocks(src, reply):
    """Apply each SEARCH/REPLACE block. Exact match first, then trailing-ws-tolerant,
    then stripped-line span match (handles indentation drift). Returns (new_src, n_applied)."""
    out=src; applied=0
    for sm, rep in _SR.findall(reply):
        s=sm.strip("\n"); r=rep.strip("\n")
        if not s.strip():
            continue
        if s in out:
            out=out.replace(s, r, 1); applied+=1; continue
        rstrip=lambda t: "\n".join(l.rstrip() for l in t.split("\n"))
        if rstrip(s) in rstrip(out):
            out=rstrip(out).replace(rstrip(s), rstrip(r), 1); applied+=1; continue
        # span match on stripped lines; re-indent replacement to the matched block's indent
        ol=out.split("\n"); sl=s.split("\n"); k=len(sl)
        tgt=[l.strip() for l in sl]; hit=-1
        for i in range(len(ol)-k+1):
            if [l.strip() for l in ol[i:i+k]]==tgt: hit=i; break
        if hit>=0:
            indent=re.match(r"\s*", ol[hit]).group()
            rl=[(indent+l if l.strip() else l) for l in r.split("\n")]
            ol[hit:hit+k]=rl; out="\n".join(ol); applied+=1
    return out, applied

def _valid_py(code):
    try: ast.parse(code); return True
    except Exception: return False

def fix_file(problem, path, src, couple):
    """Return (new_src, applied) — only an edit that applies AND still parses; else (src,0)."""
    if couple:
        plan=think([{"role":"system","content":"You are an expert software engineer. From the "
                     "issue and file, diagnose the root cause and state the precise minimal fix "
                     "(which lines, what they become). Do not write code blocks."},
                    {"role":"user","content":f"GitHub issue:\n{problem[:4000]}\n\nFile {path}:\n"
                     f"```python\n{src[:60000]}\n```\nWhat is the bug and the exact minimal change?"}])
        usr=(f"Issue:\n{problem[:2500]}\n\nExpert diagnosis:\n{plan[-2500:]}\n\nFile {path}:\n"
             f"```python\n{src[:60000]}\n```\n\n{_FMT}")
    else:
        usr=(f"GitHub issue:\n{problem[:4000]}\n\nFile {path}:\n```python\n{src[:60000]}\n```\n\n{_FMT}")
    reply=doer([{"role":"system","content":"You are an expert Python engineer making a minimal "
                 "bug fix via SEARCH/REPLACE edit blocks. Output only the blocks."},
                {"role":"user","content":usr}], mt=2048)
    new, applied=_apply_blocks(src, reply)
    if applied and new!=src and _valid_py(new):
        return new, applied
    return src, 0

def make_patch(repo_dir, problem, files, couple):
    sh(f"cd {repo_dir} && git checkout -q -f .")          # clean slate
    for f in files:
        ap=os.path.join(repo_dir,f)
        try: src=open(ap,encoding="utf-8",errors="ignore").read()
        except OSError: continue
        new, applied=fix_file(problem, f, src, couple)
        if applied:
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
