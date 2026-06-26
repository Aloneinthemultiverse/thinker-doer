# Kaggle T4×2 — dual-GPU thinker/doer setup

32 GB total (16+16). Both models resident, **no swap**: thinker on GPU0, doer on GPU1.
T4 is Turing → **fp16 only** (no bf16). Cap context ~64K (1M is theoretical on T4).

## 1. Enable
Notebook → Settings → Accelerator → **GPU T4 ×2**. Internet **On** (to pull GGUF/weights).

## 2. Serve both models (llama.cpp, one server per GPU)

```bash
# GPU 1 — DOER: Qwythos-9B on :8080
CUDA_VISIBLE_DEVICES=1 ./llama-server \
  -m qwythos-9b-Q4_K_M.gguf --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 65536 --parallel 1 &

# GPU 0 — THINKER: Phi-4 14B on :8081
CUDA_VISIBLE_DEVICES=0 ./llama-server \
  -m phi-4-Q4_K_M.gguf --host 0.0.0.0 --port 8081 \
  -ngl 99 -c 16384 --parallel 1 &
```

(vLLM alternative: launch two `vllm serve` processes pinned with `CUDA_VISIBLE_DEVICES`,
`--port 8080` / `--port 8081`, `--dtype float16`.)

## 3. Point the harness at them
Defaults already match (`:8080` doer, `:8081` thinker). Override only if needed:

```bash
export DOER_ENDPOINT=http://127.0.0.1:8080/v1/chat/completions
export THINKER_ENDPOINT=http://127.0.0.1:8081/v1/chat/completions
export DOER_MODEL=qwythos-9b
export THINKER_MODEL=phi-4
```

## 4. Run the benchmark (couple vs doer-solo)

```bash
python -m eval.run_couple eval/data                 # couple
TD_FORCE_SOLO=1 python -m eval.run_couple eval/data  # doer-solo baseline
```

The delta between the two = the **coupling lift**. That's the number this repo exists to find.

## Memory budget (Q4, fp16 KV)
| | weights | KV @ ctx | GPU |
|--|--|--|--|
| Phi-4 14B | ~9 GB | ~3 GB @16K | GPU0 (~12/16) |
| Qwythos-9B | ~7 GB | ~6 GB @64K | GPU1 (~13/16) |

Leave headroom — drop ctx if you OOM. Add VibeThinker-3B (~3 GB) to GPU0 only if testing
it vs Phi-4 for the math seat.
