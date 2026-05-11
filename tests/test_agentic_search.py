"""
Unit tests for `Index.query(agentic_search=True)`.

The SDK side is verified against a mock control-plane API: we assert that
every per-query retrieval kwarg (``mode``, ``alpha``, ``dynamic_hybrid``,
``filter``, ``exact``, ``phrase``, ``fuzzy``) is forwarded verbatim on
the multihop body, that validation kicks in on incompatible combos, and
that the response shape returned to the user is a regular ``QueryResponse``
with the multihop-only fields (``final_answer``, ``chain``, ``n_hops``,
``hops``) populated as expected.
"""
from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import pytest

from dasein import Client
from dasein.index import Index


# ── shared mock server (mirrors test_client.py pattern) ────────────────────

class _Handler(BaseHTTPRequestHandler):
    _responses: dict = {}
    _request_log: list = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        _Handler._request_log.append({
            "path": self.path,
            "headers": dict(self.headers),
            "body": json.loads(body) if body else None,
        })
        resp = _Handler._responses.get(self.path)
        if resp is None:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detail":"not found"}')
            return
        self.send_response(resp.get("status", 200))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp.get("body", {})).encode())

    def log_message(self, *args, **kwargs):  # silence noisy default logger
        return


@pytest.fixture(scope="module")
def server():
    s = HTTPServer(("127.0.0.1", 0), _Handler)
    port = s.server_address[1]
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    s.shutdown()


@pytest.fixture(autouse=True)
def reset():
    _Handler._responses = {}
    _Handler._request_log = []


def _stub_multihop_response(index_id: str, *, hops_payload=None,
                             final_answer="Inception", n_hops=3,
                             include_top_level_answer: bool = True):
    """Install a canned successful multihop response on the mock server."""
    if hops_payload is None:
        # Default: 3 hops, last hop returns 2 fused ids with text+metadata.
        hops_payload = [
            {"hop": 0, "sub_query_text": "subq1",
             "fused_ids": [], "fused_scores": [],
             "fused_texts": {}, "fused_metadata": {},
             "answer": "intermediate-1", "timings_ms": {"total_ms": 200.0}},
            {"hop": 1, "sub_query_text": "subq2",
             "fused_ids": [], "fused_scores": [],
             "fused_texts": {}, "fused_metadata": {},
             "answer": "intermediate-2", "timings_ms": {"total_ms": 220.0}},
            {"hop": 2, "sub_query_text": "subq3",
             "fused_ids": ["docA", "docB"],
             "fused_scores": [0.91, 0.78],
             "fused_texts": {"docA": "text A", "docB": "text B"},
             "fused_metadata": {"docA": {"year": 2010},
                                "docB": {"year": 2014}},
             "answer": final_answer,
             "timings_ms": {"total_ms": 250.0}},
        ]
    body = {
        "question": "the question",
        "chain": ["#1", "#2 of #1", "#3 of #2"],
        "n_hops": n_hops,
        "max_hops_cap": 5,
        "hops": hops_payload,
    }
    if include_top_level_answer:
        body["final_answer"] = final_answer
    _Handler._responses[f"/indexes/{index_id}/multihop/query"] = {
        "status": 200, "body": body,
    }


def _make_idx(server: str) -> Index:
    client = Client(api_key="dsk_test", base_url=server, max_retries=0)
    return Index(client=client, index_id="ix1")


def _last_body() -> dict:
    return _Handler._request_log[-1]["body"]


# ── happy-path: each retrieval mode forwards correctly ─────────────────────

def test_agentic_dense_mode_forwards_kwargs(server):
    _stub_multihop_response("ix1")
    idx = _make_idx(server)
    resp = idx.query("who founded apple?",
                     top_k=10, mode="dense", agentic_search=True)
    body = _last_body()
    assert _Handler._request_log[-1]["path"] == "/indexes/ix1/multihop/query"
    assert body["question"] == "who founded apple?"
    assert body["top_k"] == 10
    assert body["mode"] == "dense"
    assert body["dynamic_hybrid"] is False
    assert body["alpha"] == 0.5
    assert body["exact"] is False
    assert body["phrase"] is False
    assert body["fuzzy"] is False
    assert body["include_answer"] is False
    assert "filter" not in body or body["filter"] is None
    # response shape — final_answer is opt-in, off by default
    assert resp.final_answer is None
    assert resp.n_hops == 3
    assert len(resp.results) == 2
    assert resp.results[0].id == "docA"
    assert resp.results[0].score == pytest.approx(0.91)
    assert resp.results[0].text == "text A"
    assert resp.results[0].metadata == {"year": 2010}


def test_agentic_hybrid_alpha_forwarded(server):
    _stub_multihop_response("ix1")
    idx = _make_idx(server)
    idx.query("foo", agentic_search=True, mode="hybrid", alpha=0.7)
    body = _last_body()
    assert body["mode"] == "hybrid"
    # Public alpha is the dense weight (Pinecone / Weaviate); the SDK
    # flips to the server's internal BM25-weight convention on the wire.
    # alpha=0.7 (dense-leaning) → body["alpha"]=0.3 (BM25-weight=0.3).
    assert body["alpha"] == pytest.approx(0.3)
    assert body["dynamic_hybrid"] is False


def test_agentic_dynamic_hybrid_forwarded(server):
    _stub_multihop_response("ix1")
    idx = _make_idx(server)
    idx.query("foo", agentic_search=True, mode="hybrid",
              dynamic_hybrid=True, top_k=20)
    body = _last_body()
    assert body["mode"] == "hybrid"
    assert body["dynamic_hybrid"] is True
    assert body["top_k"] == 20


def test_agentic_filter_forwarded(server):
    _stub_multihop_response("ix1")
    idx = _make_idx(server)
    idx.query("foo", agentic_search=True,
              filter={"year": {"$gte": 2010}, "category": {"$in": ["sci-fi", "drama"]}})
    body = _last_body()
    assert body["filter"] == {"year": {"$gte": 2010},
                              "category": {"$in": ["sci-fi", "drama"]}}


def test_agentic_bm25_modifiers_forwarded(server):
    _stub_multihop_response("ix1")
    idx = _make_idx(server)
    idx.query("foo", agentic_search=True, mode="hybrid",
              exact=True, phrase=True, fuzzy=True)
    body = _last_body()
    assert body["mode"] == "hybrid"
    assert body["exact"] is True
    assert body["phrase"] is True
    assert body["fuzzy"] is True


def test_agentic_return_hops_exposes_full_trace(server):
    _stub_multihop_response("ix1")
    idx = _make_idx(server)
    resp = idx.query("foo", agentic_search=True, return_hops=True)
    assert resp.hops is not None
    assert len(resp.hops) == 3
    assert resp.hops[2]["sub_query_text"] == "subq3"
    assert resp.chain == ["#1", "#2 of #1", "#3 of #2"]
    # SDK always sends verbose=True (we need the final hop's fused_ids
    # to materialize response.results); return_hops only controls
    # whether we expose the per-hop trace to the caller.
    assert _last_body()["verbose"] is True


def test_agentic_return_hops_default_false_hides_trace(server):
    _stub_multihop_response("ix1")
    idx = _make_idx(server)
    resp = idx.query("foo", agentic_search=True)
    # User-facing hops are hidden, but the SDK still asked for them
    # over the wire so it could shape `results`.
    assert resp.hops is None
    assert _last_body()["verbose"] is True
    # results were still populated from the last hop's fused_ids
    assert len(resp.results) == 2
    assert resp.results[0].id == "docA"


def test_agentic_include_answer_default_off(server):
    _stub_multihop_response("ix1")
    idx = _make_idx(server)
    resp = idx.query("foo", agentic_search=True)
    assert resp.final_answer is None


def test_agentic_include_answer_opt_in_populates_field(server):
    _stub_multihop_response("ix1", final_answer="Inception")
    idx = _make_idx(server)
    resp = idx.query("foo", agentic_search=True, include_answer=True)
    assert resp.final_answer == "Inception"
    # the toggle must also be forwarded over the wire so the server can
    # actually run the reader on the final hop
    assert _last_body()["include_answer"] is True


def test_agentic_include_answer_handles_missing_server_field(server):
    _stub_multihop_response("ix1", include_top_level_answer=False)
    idx = _make_idx(server)
    resp = idx.query("foo", agentic_search=True, include_answer=True)
    assert resp.final_answer is None


# ── validation: incompatible combos must fail loud ─────────────────────────

def test_agentic_requires_text(server):
    idx = _make_idx(server)
    with pytest.raises(ValueError, match="requires `text`"):
        idx.query(vector=[0.1, 0.2, 0.3], agentic_search=True)


def test_agentic_rejects_vector(server):
    idx = _make_idx(server)
    with pytest.raises(ValueError, match="text-only"):
        idx.query(text="foo", vector=[0.1, 0.2], agentic_search=True)


def test_agentic_rejects_include_vectors(server):
    idx = _make_idx(server)
    with pytest.raises(ValueError, match="does not return vectors"):
        idx.query(text="foo", agentic_search=True, include_vectors=True)


def test_agentic_rejects_bad_mode(server):
    idx = _make_idx(server)
    with pytest.raises(ValueError, match="mode must be"):
        idx.query(text="foo", agentic_search=True, mode="weird")


def test_agentic_dynamic_hybrid_top_k_cap(server):
    idx = _make_idx(server)
    with pytest.raises(ValueError, match="top_k <= 100"):
        idx.query(text="foo", agentic_search=True,
                  dynamic_hybrid=True, top_k=101)


# ── response: empty hops still produces a usable QueryResponse ─────────────

def test_agentic_empty_hops_yields_empty_results(server):
    _Handler._responses["/indexes/ix1/multihop/query"] = {
        "status": 200,
        "body": {
            "question": "x", "chain": [],
            "final_answer": "", "n_hops": 0,
            "max_hops_cap": 5, "hops": [],
        },
    }
    idx = _make_idx(server)
    resp = idx.query("x", agentic_search=True)
    assert len(resp.results) == 0
    # final_answer is opt-in; without include_answer=True it stays None
    # regardless of what the server returned.
    assert resp.final_answer is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
