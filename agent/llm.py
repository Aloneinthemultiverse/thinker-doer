"""Thin client for two local llama/vLLM servers (OpenAI-compatible /v1/chat/completions).

Two named seats, each its own endpoint (its own GPU on T4x2):

  THINKER (Phi-4 14B)   — plans, decomposes, reasons, knows things.   default :8081
  DOER    (Qwythos-9B)  — writes code, runs tools, drives the loop.   default :8080

Override via env so the same code runs on Kaggle (two ports) or a single shared server.
"""
import json
import os
import urllib.request

DOER_ENDPOINT = os.environ.get("DOER_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")
THINKER_ENDPOINT = os.environ.get("THINKER_ENDPOINT", "http://127.0.0.1:8081/v1/chat/completions")

DOER_MODEL = os.environ.get("DOER_MODEL", "qwythos-9b")
THINKER_MODEL = os.environ.get("THINKER_MODEL", "phi-4")


def _post(endpoint, body, timeout):
    req = urllib.request.Request(
        endpoint, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _chat(endpoint, model, messages, temperature, max_tokens, top_p=0.95, timeout=600):
    body = {"model": model, "messages": messages, "temperature": temperature,
            "top_p": top_p, "max_tokens": max_tokens, "stream": False}
    return _post(endpoint, body, timeout)


def chat_thinker(messages, temperature=0.4, max_tokens=3072):
    """Phi-4: long reasoning / planning. Allowed to ramble — the doer distills it."""
    return _chat(THINKER_ENDPOINT, THINKER_MODEL, messages, temperature, max_tokens)


def chat_doer(messages, temperature=0.2, max_tokens=1536):
    """Qwythos: precise single-block code emission / tool use."""
    return _chat(DOER_ENDPOINT, DOER_MODEL, messages, temperature, max_tokens)


def _healthy(endpoint):
    base = endpoint.rsplit("/v1/", 1)[0] + "/v1/models"
    try:
        urllib.request.urlopen(base, timeout=5).read()
        return True
    except Exception:
        return False


def thinker_healthy():
    return _healthy(THINKER_ENDPOINT)


def doer_healthy():
    return _healthy(DOER_ENDPOINT)
