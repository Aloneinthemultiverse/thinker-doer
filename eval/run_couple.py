"""Minimal benchmark: couple vs doer-solo on a slice of buggy-file fix tasks.

Each task = a dir with buggy.py + test_buggy.py. Drops the couple into each, measures
pass-rate and rounds. Run the same slice with TD_FORCE_SOLO=1 to get the doer-solo
baseline, so you can read the *coupling lift* directly.

Usage (on Kaggle, after both servers are up):
    python -m eval.run_couple /path/to/tasks
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import couple, tools  # noqa: E402


def run_slice(tasks_dir):
    tasks = sorted(d for d in os.listdir(tasks_dir)
                   if os.path.isdir(os.path.join(tasks_dir, d)))
    results, solo = [], os.environ.get("TD_FORCE_SOLO") == "1"
    for name in tasks:
        sb = tempfile.mkdtemp(prefix="td_")
        shutil.copytree(os.path.join(tasks_dir, name), sb, dirs_exist_ok=True)
        os.environ["TD_SANDBOX"] = tools.SANDBOX = sb
        prompt = _read_prompt(sb, name)
        if solo:
            ok, rnds, via = couple._doer_solo(prompt, "buggy.py", lambda *a: None)
        else:
            ok, rnds, via = couple.solve(prompt, verbose=False)
        results.append((name, ok, rnds, via))
        print(f"  {'PASS' if ok else 'fail':4}  {name:28} {via:10} r{rnds}")
    passed = sum(1 for _, ok, _, _ in results if ok)
    mode = "DOER-SOLO" if solo else "COUPLE"
    print(f"\n{mode}: {passed}/{len(results)} = {100*passed/max(1,len(results)):.1f}%")
    return results


def _read_prompt(sb, name):
    p = os.path.join(sb, "prompt.txt")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else \
        f"Fix the bug(s) in buggy.py so all tests in test_buggy.py pass. ({name})"


if __name__ == "__main__":
    run_slice(sys.argv[1] if len(sys.argv) > 1 else "eval/data")
