# Next: Agentless-style scaffold for the couple

## Results so far (real, verified)

| Benchmark | Couple (Phi-4→Qwythos) | Qwythos solo | Lift |
|---|---|---|---|
| HumanEval[40] | 92.5% | 82.5% | **+10** |
| SWE-bench Lite[25] (official, Modal) | 12% (3/25) | 4% (1/25) | **+8** |

The SWE-bench number is **scaffold-limited, not model-limited**: 14/25 (couple) and 16/25 (solo)
were **empty patches** because flat `git grep` localization found the wrong file — Qwythos never
got a shot. Of instances it *did* attempt: 3/11 = 27% resolved.

## The fix — adopt Agentless (github.com/openautocoder/agentless)

SOTA *simple* approach (27.3% on Lite, **no agent loop**). It proved precise **localization**
matters more than a fancy agent loop — exactly our hole.

### 3-step hierarchical localization (replaces flat git-grep)
1. **File** — build a repo **skeleton** (file tree + each file's class/`def` *signatures* via `ast`,
   no bodies). Give the compact map to the LLM → rank candidate files.
2. **Function** — show classes/funcs of the chosen files → LLM picks which functions.
3. **Edit** — show those funcs' code → fine-grained edit locations.

### Repair
Sample **N** patch candidates → filter (applies + AST-parses, already in v16) → rank → pick best.

### Split across the couple (plays to both strengths)
- **Phi-4** = hierarchical localization reasoning (planning strength)
- **Qwythos** = candidate patch generation (coding/agentic strength)

## 3 quick wins regardless
- (a) Parse the issue for explicit `*.py` paths + traceback `File "..."` frames — issues usually
  **name the file** → strongest localization signal.
- (b) Retry-on-empty — try the next candidate file instead of bailing.
- (c) Drop the `MAX_LINES=800` cap — the real buggy file is often large; Qwythos has the context.

**Goal:** cut the ~14–16/25 empty patches → more attempts → higher resolved count; the +8 couple
lift should hold. **File to edit:** `kaggle/kernel_swebench.py`.

## Ops notes (learned the hard way)
- Kaggle: only **UI "Save & Run All" (committed)** gets T4×2 and persists output (API push = single
  P100, interactive runs lose output). Committed output is retrievable via REST `kernels/output`.
- **sb-cli cloud grader is broken/flaky** (marked correct patches "failed"). Use the **official
  harness with `--modal true`** instead — but it imports the unix-only `resource` module, so run it
  from **Linux/WSL** (Windows fails), Modal free tier does the Docker eval in the cloud.
- Ignore AI-marketing benchmark claims (e.g. "Phi-4 70–76% SWE" = Verdent the *product*, not the
  model). Trust only measured numbers.
