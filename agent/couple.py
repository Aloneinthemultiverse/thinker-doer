"""Thinker/Doer couple — Phi-4 plans, Qwythos executes, verifier anchors, retry-with-hint.

Ported from vibe-thinker `duo.py` (validated 2026-06-23). Same proven shape, stronger seats:

  THINKER (Phi-4 14B, :8081)  — THINKS: decomposes the task, diagnoses, states the exact fix.
                                Allowed long CoT — the doer distills it.
  DOER    (Qwythos-9B, :8080) — HANDS: transcribes the plan into ONE clean file, writes it.
  VERIFIER (run_tests)        — anchors. On fail, the exact failing assertion is fed back.

They communicate ONLY through the shared graph (blackboard). Guards: MAX_ROUNDS, no-progress
detector (identical attempt -> stop), graceful degrade to doer-solo. Everything local, $0.
"""
import os
import re

from . import llm, tools
from .graph import Graph

MAX_ROUNDS = int(os.environ.get("TD_MAX_ROUNDS", "4"))
THINK_TOKENS = int(os.environ.get("TD_THINK_TOKENS", "3072"))

_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _hint(test_output):
    """Actionable nudge from verifier output — the single failing assertion."""
    if not test_output or "FAIL" not in test_output.upper():
        return ""
    lines = [l.strip() for l in test_output.splitlines() if l.strip()]
    return next((l for l in reversed(lines)
                 if "Error" in l or l.startswith("assert") or "!=" in l), "")[:200]


def _think(task, current_src, failing, history=""):
    """Phi-4 THINKS: CoT diagnosing the bug(s) + exact fix. Returns (cot, short_insight)."""
    hint = _hint(failing)
    usr = f"{task}\n\nCurrent file:\n```python\n{current_src}```\n"
    if history:
        usr += f"\nPrior attempts (brief, don't repeat them):\n{history}\n"
    if failing:
        usr += f"\nThe tests still FAIL. Latest output:\n{failing[:700]}\n"
    if hint:
        usr += (f"\nFocus: the failing check is `{hint}`. Reason about the EXACT expected "
                "value (types, case, order) and what the code must return to match it.")
    usr += "\nDecompose the problem, reason step by step about EVERY bug, state the exact fix."
    try:
        cot = llm.chat_thinker(
            [{"role": "system", "content": "You are an expert Python architect and debugger. "
              "Plan precisely; be exact about return values — types, case, ordering."},
             {"role": "user", "content": usr}],
            temperature=0.4, max_tokens=THINK_TOKENS)
    except Exception as e:
        return f"(thinker error: {e})", "thinker-unavailable"
    tail = [l.strip() for l in cot.splitlines() if l.strip()][-3:]
    return cot, " ".join(tail)[:300]


def _best_block(reply):
    blocks = [b.strip() for b in _CODE.findall(reply) if b.strip() and "def " in b]
    return (max(blocks, key=lambda b: (b.count("def "), len(b))) + "\n") if blocks else None


def _do(task, cot, current_src, rel):
    """Qwythos HANDS: transcribe the plan into ONE clean corrected file, write, verify."""
    try:
        reply = llm.chat_doer(
            [{"role": "system", "content": "Output ONLY the complete corrected file inside "
              "one ```python code block. No prose."},
             {"role": "user", "content":
              f"Task: {task}\n\nAn expert planned the fix:\n{cot[-2500:]}\n\n"
              f"Original file:\n```python\n{current_src}```\n\n"
              "Write the FULL corrected file based on the expert's plan."}],
            temperature=0.2, max_tokens=1536)
    except Exception as e:
        return False, None, f"[doer error: {e}]"
    body = _best_block(reply)
    if not body:
        return False, None, "[doer produced no code block]"
    res = tools.write_file(rel, body)
    if res.startswith("ERROR"):
        return False, body, res
    out = tools.run_tests()
    return tools.tests_pass(out), body, out


def solve(task, rel="buggy.py", verbose=True):
    """One-agent facade. Returns (ok, rounds, via): 'coupling' | 'doer-solo' | 'already'."""
    def say(*a):
        if verbose:
            try:
                print(" ".join(str(x) for x in a))
            except UnicodeEncodeError:
                print(" ".join(str(x) for x in a).encode("ascii", "replace").decode())

    if tools.tests_pass():
        return True, 0, "already"
    if not llm.thinker_healthy():
        say("[couple] thinker down -> doer-solo")
        return _doer_solo(task, rel, say)

    g = Graph()
    g.add("task", "system", task, ns="shared")
    src = tools.read_file(rel)
    if src.startswith("ERROR"):
        return False, 0, "no-file"
    failing = tools.run_tests()
    last_body = None

    for rnd in range(1, MAX_ROUNDS + 1):
        say(f"\n=== round {rnd}/{MAX_ROUNDS} ===")
        # a failing verifier from the prior round is an objective DOUBT region -> escalate
        # to the thinker (Phi-4). On round 1 there's no doubt yet; the plan is the entry.
        if rnd > 1 and failing:
            g.open_doubt(_hint(failing) or "tests still failing")
        history = g.history(kind="result", k=3) if rnd > 1 else ""
        cot, insight = _think(task, src, failing, history)
        # thinker writes to its private KG, then promotes the plan to the shared blackboard
        pid = g.add("plan", "thinker", insight, ns="phi4")
        g.promote(pid, author="thinker")
        say(f"  [thinker] {len(cot)} chars | {insight[:80]}")

        passed, body, out = _do(task, cot, src, rel)
        # doer writes its result to its private KG, then commits it to shared
        rid = g.add("result", "doer", (out.splitlines() or [""])[0][:140],
                    ns="qwythos", payload={"passed": passed})
        g.promote(rid, author="doer")
        say(f"  [doer] -> {'PASS' if passed else 'fail'}")

        if passed:
            say("  verifier PASSED -> success (coupling)")
            return True, rnd, "coupling"
        if body:
            if body.strip() == (last_body or "").strip():
                say("  [guard] no-progress (identical attempt) -> doer-solo")
                return _doer_solo(task, rel, say)
            last_body, src = body, body
        failing = out

    return _doer_solo(task, rel, say)


def _doer_solo(task, rel, say):
    """Graceful degrade: Qwythos alone, one shot + verify."""
    src = tools.read_file(rel)
    try:
        reply = llm.chat_doer(
            [{"role": "system", "content": "Output ONLY the complete corrected file in one "
              "```python block."},
             {"role": "user", "content": f"Task: {task}\n\nFile:\n```python\n{src}```\n"
              "Write the FULL corrected file."}],
            temperature=0.2, max_tokens=1536)
    except Exception as e:
        return False, 0, f"doer-error:{e}"
    body = _best_block(reply)
    if body:
        tools.write_file(rel, body)
    ok = tools.tests_pass()
    say(f"  [doer-solo] -> {'PASS' if ok else 'fail'}")
    return ok, 1, "doer-solo"
