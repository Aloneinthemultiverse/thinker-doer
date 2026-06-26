"""Kaggle T4x2 kernel — paste cell-by-cell, or run as a script in a GPU-T4x2 notebook.

Serves Phi-4 (GPU0, :8081) + Qwythos-9B (GPU1, :8080) with llama.cpp, then runs the
couple-vs-doer-solo benchmark to measure the COUPLING LIFT. fp16 only (Turing); ctx capped.

Set the two GGUF paths below (Kaggle dataset, HF download, or /kaggle/input/...).
"""
import os
import subprocess
import time
import urllib.request

# ---- 0. config -------------------------------------------------------------
PHI4_GGUF = os.environ.get("PHI4_GGUF", "/kaggle/input/phi-4-gguf/phi-4-Q4_K_M.gguf")
QWY_GGUF = os.environ.get("QWY_GGUF", "/kaggle/input/qwythos-9b-gguf/qwythos-9b-Q4_K_M.gguf")
REPO = os.environ.get("TD_REPO", "/kaggle/working/thinker-doer")
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "/kaggle/working/llama.cpp/build/bin/llama-server")


def sh(cmd, **kw):
    print("$", cmd)
    return subprocess.run(cmd, shell=True, **kw)


# ---- 1. build llama.cpp (CUDA) once ---------------------------------------
def build_llama():
    if os.path.exists(LLAMA_SERVER):
        return
    sh("cd /kaggle/working && git clone --depth 1 https://github.com/ggerganov/llama.cpp")
    sh("cd /kaggle/working/llama.cpp && cmake -B build -DGGML_CUDA=ON "
       "-DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build --config Release -j --target llama-server")


# ---- 2. launch both servers, one per GPU ----------------------------------
def serve():
    procs = []
    procs.append(subprocess.Popen(
        f"CUDA_VISIBLE_DEVICES=1 {LLAMA_SERVER} -m {QWY_GGUF} "
        f"--host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --parallel 1",
        shell=True))
    procs.append(subprocess.Popen(
        f"CUDA_VISIBLE_DEVICES=0 {LLAMA_SERVER} -m {PHI4_GGUF} "
        f"--host 0.0.0.0 --port 8081 -ngl 99 -c 16384 --parallel 1",
        shell=True))
    return procs


def wait_ready(timeout=600):
    for port in (8080, 8081):
        ok = False
        for _ in range(timeout // 5):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5).read()
                ok = True
                break
            except Exception:
                time.sleep(5)
        print(f"  :{port} {'UP' if ok else 'DOWN'}")
        if not ok:
            raise RuntimeError(f"server :{port} never came up")


# ---- 3. run the benchmark both ways ---------------------------------------
def benchmark():
    os.environ.update(
        DOER_ENDPOINT="http://127.0.0.1:8080/v1/chat/completions",
        THINKER_ENDPOINT="http://127.0.0.1:8081/v1/chat/completions",
        DOER_MODEL="qwythos-9b", THINKER_MODEL="phi-4")
    import sys
    sys.path.insert(0, REPO)
    from eval.run_couple import run_slice
    data = os.path.join(REPO, "eval", "data")
    print("\n========== COUPLE ==========")
    run_slice(data)
    print("\n========== DOER-SOLO (baseline) ==========")
    os.environ["TD_FORCE_SOLO"] = "1"
    run_slice(data)
    print("\nLift = couple% - doer-solo%  <-- the number this whole repo exists to find.")


if __name__ == "__main__":
    build_llama()
    p = serve()
    wait_ready()
    benchmark()
