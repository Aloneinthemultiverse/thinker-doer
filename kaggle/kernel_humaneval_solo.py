# thinker-doer | Qwythos-9B SOLO HumanEval (single model -> fits one GPU -> API-runnable)
# Loads ONLY Qwythos, runs full HumanEval pass@1 with verify-retry, saves per-problem JSON.
import gzip, json, os, re, subprocess, sys, tempfile, time, urllib.request

QWY_REPO=os.environ.get("QWY_REPO","empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")
QUANT=os.environ.get("TD_QUANT","Q4_K_M")
N=int(os.environ.get("TD_N","164"))          # full HumanEval
RETRY=os.environ.get("TD_RETRY","1")=="1"
def sh(c): print("$",c,flush=True); return subprocess.run(c,shell=True)

sh("pip -q install huggingface_hub")
sh("pip -q install 'llama-cpp-python[server]' --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
from huggingface_hub import hf_hub_download, list_repo_files
def resolve(repo,q=QUANT):
    fs=[f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    pick=next((f for f in fs if q.lower() in f.lower()),None) or sorted([f for f in fs if "q4" in f.lower()] or fs,key=len)[0]
    print("  ",repo,"->",pick,flush=True); return hf_hub_download(repo,pick)
QWY=resolve(QWY_REPO)

env=dict(os.environ,CUDA_VISIBLE_DEVICES="0")
subprocess.Popen(["python","-m","llama_cpp.server","--model",QWY,"--host","0.0.0.0","--port","8080",
    "--n_gpu_layers","-1","--n_ctx","8192","--n_batch","64","--n_ubatch","64"],env=env)
up=False
for _ in range(180):
    try: urllib.request.urlopen("http://127.0.0.1:8080/v1/models",timeout=5).read(); up=True; break
    except Exception: time.sleep(5)
print(" :8080",("UP" if up else "DOWN"),flush=True)
if not up: raise SystemExit("server down")

EP="http://127.0.0.1:8080/v1/chat/completions"
def doer(msgs,mt=1024):
    body=json.dumps({"messages":msgs,"temperature":0.2,"top_p":0.95,"max_tokens":mt,"stream":False}).encode()
    req=urllib.request.Request(EP,data=body,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=600).read())["choices"][0]["message"]["content"]

DATA="/kaggle/working/HumanEval.jsonl.gz"
if not os.path.exists(DATA):
    urllib.request.urlretrieve("https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",DATA)
PROBS=[json.loads(l) for l in gzip.open(DATA,"rt",encoding="utf-8")][:N]
print(f"loaded {len(PROBS)} HumanEval problems",flush=True)

def extract(t):
    t=re.sub(r"<think>.*?</think>","",t,flags=re.S); t=re.sub(r"<think>.*","",t,flags=re.S)
    m=re.search(r"```(?:python)?\n(.*?)```",t,re.S); return (m.group(1) if m else t).strip()
def gen(prompt,prior=""):
    usr=("Complete this Python function. Return ONLY the full function definition in one ```python block.\n\n"+prompt)
    if prior: usr+=("\n\nPrevious attempt FAILED:\n"+prior+"\nReturn a corrected full function.")
    return extract(doer([{"role":"system","content":"You are an expert Python programmer."},{"role":"user","content":usr}]))
def header(p,e):
    m=re.search(rf"^def\s+{re.escape(e)}\s*\(",p,re.M); return p[:m.start()] if m else ""
def ensure(c,p,e): return c if re.search(rf"def\s+{re.escape(e)}\s*\(",c) else p+"\n"+c
def run_test(code,test,entry,hdr=""):
    prog=hdr+"\n"+code+"\n\n"+test+f"\n\ncheck({entry})\nprint('__PASSED__')\n"
    with tempfile.NamedTemporaryFile("w",suffix=".py",delete=False,encoding="utf-8") as f: f.write(prog); path=f.name
    try:
        p=subprocess.run([sys.executable,path],capture_output=True,text=True,timeout=15)
        return "__PASSED__" in p.stdout,(p.stderr or p.stdout)[-300:]
    except Exception as e: return False,str(e)[:200]
    finally:
        try: os.unlink(path)
        except OSError: pass

OUT="/kaggle/working/qwythos_humaneval.json"; results=[]
if os.path.exists(OUT):
    try: results=json.load(open(OUT,encoding="utf-8"))
    except Exception: results=[]
done={r["task_id"] for r in results}
for i,p in enumerate(PROBS,1):
    if p["task_id"] in done:
        print(f"[{i}/{len(PROBS)}] {p['task_id']} cached",flush=True); continue
    hdr=header(p["prompt"],p["entry_point"])
    code=ensure(gen(p["prompt"]),p["prompt"],p["entry_point"])
    ok,err=run_test(code,p["test"],p["entry_point"],hdr); used=False
    if not ok and RETRY:
        used=True; code=ensure(gen(p["prompt"],err),p["prompt"],p["entry_point"]); ok,err=run_test(code,p["test"],p["entry_point"],hdr)
    results.append({"task_id":p["task_id"],"ok":bool(ok),"used_retry":bool(used),"code":code})
    json.dump(results,open(OUT,"w"))   # checkpoint every problem
    print(f"[{i}/{len(PROBS)}] {p['task_id']} {'PASS' if ok else 'FAIL'}{' (retry)' if ok and used else ''}",flush=True)

passed=sum(r["ok"] for r in results); saves=sum(r["ok"] and r["used_retry"] for r in results)
print(f"\n*** Qwythos-solo HumanEval pass@1: {passed}/{len(results)} = {100*passed/len(results):.1f}% "
      f"(single-shot {passed-saves}/{len(results)} = {100*(passed-saves)/len(results):.1f}%, retry rescued {saves}) ***")
print(f"saved {OUT}")
