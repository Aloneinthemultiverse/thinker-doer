# thinker-doer

A fully-local **thinker/doer couple**: two specialist models coupled through a shared
blackboard graph, solving agentic coding/reasoning tasks with a verify→retry loop.
Descends directly from the validated `vibe-thinker` couple (Reasoner+Actor) — same
proven shape, stronger seats.

## Roster (one specialist per axis)

| Seat | Model | Strength | ~VRAM (Q4) |
|------|-------|----------|------------|
| **Thinker / planner / knowledge** | Phi-4 14B | MMLU 84.8, MATH 80.4, decomposition | ~9 GB |
| **Doer / coder / agentic / tools** | Qwythos-9B | SWE 68, Terminal-Bench 71.5, native tools, 1M ctx | ~7 GB |
| _(optional math co-proc)_ | VibeThinker-3B | AIME-class math | ~3 GB |
| **Communication** | shared graph (blackboard) | — | — |
| **Knowledge gap** | retrieval via graph | _retrieve, don't memorize_ | — |

## The coupling (proven pattern, ported from vibe-thinker `duo.py`)

```
task ─▶ THINKER (Phi-4)  ── posts plan/insight ─▶  shared graph
                                                       │
                          reads plan ◀────────────────┘
        DOER (Qwythos) ── writes ONE clean file ─▶ verifier (run_tests)
                                                       │
              fail ─▶ retry-with-hint (failing assertion fed back) ─▶ THINKER
              pass ─▶ done
```

Guards (inherited): `MAX_ROUNDS`, no-progress detector (identical attempt → stop),
graceful degrade to single-model. Everything local, $0.

## Why this can win (estimated, to be measured)

Fusion = best-member + synergy where the task splits cleanly. Phi-4 fills exactly the
rows Qwythos is weak on (knowledge, math) and vice-versa:

| Benchmark | Combo (est) | Phi-4 | Qwythos | Opus 4.8 | GPT-5.5 |
|-----------|-------------|-------|---------|----------|---------|
| Agentic / Terminal-Bench | **79–83** ⭐ | low | 71.5 | 76.0 | 78.2 |
| Math / gsm8k | **89–91** ⭐ | ~90 | 61.3 | 92.0 | 91.8 |
| General Knowledge / MMLU | 84–86 | 84.8 | 57.5 | 91.1 | 90.3 |
| Complex Code / SWE-bench | 73–76 | ~50s | 68.2 | 93.2 | 82.6 |

⭐ = plausibly **beats frontier, fully local**. These are projections — the point of this
repo is to replace them with **measured** numbers on Kaggle T4×2.

## Hardware

Benchmarking rig: **Kaggle T4×2 (32 GB total)** — both models resident, no swap.
Thinker on GPU0, doer on GPU1. Context capped ~64K on T4 (1M is theoretical here).
See `kaggle/`.

## Layout

```
agent/   llm.py (dual endpoint) · graph.py (blackboard) · tools.py · couple.py (the loop)
eval/    run_couple.py — combo vs thinker-solo vs doer-solo on a task slice
kaggle/  dual-GPU setup notebook
```
