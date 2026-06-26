"""Knowledge-graph blackboard — GitNexus-inspired (typed nodes + relationship edges +
traversal), with three namespaces so the couple keeps private memory but shares commitments.

Inspiration (concept only, original code): GitNexus models code as a graph of symbol nodes
joined by relationship edges, and answers impact by traversing upstream/downstream. We borrow
that shape for the couple's working memory:

  ns="qwythos"  — DOER's private KG: code structure it discovers (symbols, edits, results).
  ns="phi4"     — THINKER's private KG: plans, reasoning insights, knowledge it recalls.
  ns="shared"   — the blackboard: only COMMITTED plans/results/conflicts/doubts cross here.

Node kinds: task | plan | insight | edit | result | conflict | doubt | symbol
Edge relations: addresses | refines | produces | contradicts | depends_on | escalates

Deliberately dependency-free (stdlib only) so it runs clean on Kaggle. The heavy GitNexus
vector+graph engine is the phase-2 substrate; this proves the coupling first.
"""
import itertools
import time

_NODE_KINDS = {"task", "plan", "insight", "edit", "result", "conflict", "doubt", "symbol"}
_RELATIONS = {"addresses", "refines", "produces", "contradicts", "depends_on", "escalates"}


class KG:
    def __init__(self):
        self.nodes = {}          # id -> node dict
        self.edges = []          # list of {src, dst, rel}
        self._ids = itertools.count(1)

    # --------------------------------------------------------------- writes
    def add(self, kind, author, text, ns="shared", payload=None):
        assert kind in _NODE_KINDS, f"bad kind {kind}"
        nid = f"{ns}:{kind}:{next(self._ids)}"
        self.nodes[nid] = {"id": nid, "kind": kind, "author": author, "ns": ns,
                           "text": text, "payload": payload or {}, "t": time.time()}
        return nid

    def link(self, src, dst, rel):
        assert rel in _RELATIONS, f"bad relation {rel}"
        self.edges.append({"src": src, "dst": dst, "rel": rel})

    def promote(self, nid, author=None):
        """Copy a private node into the shared namespace (a commitment), linked back."""
        n = self.nodes[nid]
        sid = self.add(n["kind"], author or n["author"], n["text"], ns="shared",
                       payload=n["payload"])
        self.link(sid, nid, "refines")
        return sid

    # --------------------------------------------------------------- reads
    def latest(self, kind=None, ns=None, author=None):
        for nid in reversed(list(self.nodes)):
            n = self.nodes[nid]
            if (kind is None or n["kind"] == kind) and (ns is None or n["ns"] == ns) \
                    and (author is None or n["author"] == author):
                return n
        return None

    def query(self, text, ns=None, k=5):
        """Cheap lexical retrieval over node text (stand-in for GitNexus vector search)."""
        terms = {w.lower() for w in text.split() if len(w) > 2}
        scored = []
        for n in self.nodes.values():
            if ns and n["ns"] != ns:
                continue
            words = {w.lower() for w in n["text"].split()}
            s = len(terms & words)
            if s:
                scored.append((s, n))
        scored.sort(key=lambda x: (-x[0], -x[1]["t"]))
        return [n for _, n in scored[:k]]

    def neighbors(self, nid, rel=None):
        out = []
        for e in self.edges:
            if e["src"] == nid and (rel is None or e["rel"] == rel):
                out.append(self.nodes.get(e["dst"]))
        return [n for n in out if n]

    def history(self, kind=None, ns="shared", k=3):
        items = [n for n in self.nodes.values()
                 if (kind is None or n["kind"] == kind) and n["ns"] == ns]
        items.sort(key=lambda n: n["t"])
        return "\n".join(f"- {n['author']}: {n['text']}" for n in items[-k:])

    # --------------------------------------------------------------- doubt/conflict
    def open_doubt(self, text, src_nid=None):
        """A doubt region = an objective uncertainty (e.g. a failing verifier). Posting one
        is the escalation trigger: the doer asks the thinker to resolve it."""
        did = self.add("doubt", "doer", text, ns="shared")
        if src_nid:
            self.link(did, src_nid, "escalates")
        return did

    def open_conflict(self, text, a_nid, b_nid):
        cid = self.add("conflict", "system", text, ns="shared")
        self.link(cid, a_nid, "contradicts")
        self.link(cid, b_nid, "contradicts")
        return cid


# Back-compat alias for the scaffold's earlier name.
Graph = KG
