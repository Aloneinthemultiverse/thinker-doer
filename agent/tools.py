"""Sandbox file + test tools the doer uses. Verifier (run_tests) anchors the loop."""
import os
import subprocess

SANDBOX = os.environ.get("TD_SANDBOX", os.getcwd())


def _path(rel):
    return os.path.join(SANDBOX, rel)


def read_file(rel):
    try:
        return open(_path(rel), encoding="utf-8").read()
    except OSError as e:
        return f"ERROR: {e}"


def write_file(rel, body):
    try:
        with open(_path(rel), "w", encoding="utf-8") as f:
            f.write(body)
        return f"OK wrote {rel}"
    except OSError as e:
        return f"ERROR: {e}"


def run_tests(cmd=None, timeout=60):
    """Run the sandbox test command; return combined output. Default: pytest -q."""
    cmd = cmd or os.environ.get("TD_TEST_CMD", "python -m pytest -q")
    try:
        p = subprocess.run(cmd, shell=True, cwd=SANDBOX, capture_output=True,
                           text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return "FAIL: test timeout"


def tests_pass(out=None):
    out = out if out is not None else run_tests()
    low = out.lower()
    return ("fail" not in low and "error" not in low
            and ("passed" in low or "ok" in low))
