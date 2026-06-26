"""Lightweight shared blackboard — the couple's only communication channel.

The thinker posts `plan`/`insight` nodes; the doer posts `result` nodes; the verifier's
output anchors the loop. Deliberately minimal (in-memory list) for the Kaggle benchmark —
the heavy GitNexus/vector graph from vibe-thinker is the phase-2 substrate, not needed to
prove the coupling. Same node-identity idea: typed nodes with author + text + payload.
"""
import time


class Graph:
    def __init__(self, name="shared"):
        self.name = name
        self.nodes = []

    def add_node(self, kind, author, text, payload=None):
        node = {"kind": kind, "author": author, "text": text,
                "payload": payload or {}, "t": time.time()}
        self.nodes.append(node)
        return node

    def latest(self, kind=None, author=None):
        for n in reversed(self.nodes):
            if (kind is None or n["kind"] == kind) and (author is None or n["author"] == author):
                return n
        return None

    def history(self, kind=None, k=3):
        items = [n for n in self.nodes if kind is None or n["kind"] == kind]
        return "\n".join(f"- {n['author']}: {n['text']}" for n in items[-k:])
