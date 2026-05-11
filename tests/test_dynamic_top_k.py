"""
Unit tests for Dynamic Top-K — both surfaces.

Surface 1 (managed toggle): ``Index.query(dynamic_hybrid=True,
dynamic_top_k=True)`` and the agentic equivalent
``Index.query(agentic_search=True, dynamic_hybrid=True,
dynamic_top_k=True)``. We assert the body sent to the API carries
``dynamic_top_k=true`` only when the caller asked for it, and that the
SDK enforces the ``dynamic_hybrid=True`` pairing client-side.

Surface 2 (BYO retriever): ``Client.predict_dynamic_top_k(text,
query_vector=...)``. We assert it hits ``/v1/predict_dynamic_top_k``
with the right body and shapes the response into a
:class:`DynamicTopKResult`.
"""
from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import pytest

from dasein import Client, DynamicTopKResult
from dasein.index import Index


# ── shared mock server (mirrors test_client.py / test_agentic_search.py) ───

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

    def log_message(self, *args, **kwargs):
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


def _stub_query_response(index_id: str):
    _Handler._responses[f"/v1/indexes/{index_id}/query"] = {
        "status": 200,
        "body": {
            "results": [
                {"id": "docA", "score": 0.91},
                {"id": "docB", "score": 0.78},
            ],
            "mode": "hybrid",
        },
    }


def _stub_multihop_response(index_id: str):
    _Handler._responses[f"/indexes/{index_id}/multihop/query"] = {
        "status": 200,
        "body": {
            "question": "the question",
            "chain": ["#1", "#2"],
            "n_hops": 2,
            "max_hops_cap": 5,
            "hops": [
                {"hop": 0, "sub_query_text": "subq1",
                 "fused_ids": [], "fused_scores": [],
                 "fused_texts": {}, "fused_metadata": {},
                 "answer": "intermediate", "timings_ms": {"total_ms": 200.0}},
                {"hop": 1, "sub_query_text": "subq2",
                 "fused_ids": ["docA"], "fused_scores": [0.91],
                 "fused_texts": {"docA": "text A"},
                 "fused_metadata": {"docA": {"year": 2010}},
                 "answer": "Inception", "timings_ms": {"total_ms": 250.0}},
            ],
        },
    }


def _make_idx(server: str) -> Index:
    client = Client(api_key="dsk_test", base_url=server, max_retries=0)
    return Index(client=client, index_id="ix1")


def _last_body() -> dict:
    return _Handler._request_log[-1]["body"]


# ── single-hop toggle: dynamic_hybrid + dynamic_top_k ──────────────────────

def test_single_hop_dynamic_top_k_forwarded(server):
    _stub_query_response("ix1")
    idx = _make_idx(server)
    idx.query(vector=[0.1, 0.2], top_k=20, mode="hybrid",
              dynamic_hybrid=True, dynamic_top_k=True)
    body = _last_body()
    assert body["dynamic_hybrid"] is True
    assert body["dynamic_top_k"] is True
    assert body["top_k"] == 20


def test_single_hop_dynamic_top_k_off_by_default(server):
    _stub_query_response("ix1")
    idx = _make_idx(server)
    idx.query(vector=[0.1, 0.2], dynamic_hybrid=True)
    body = _last_body()
    # Off by default — only present when caller opted in
    assert "dynamic_top_k" not in body or body["dynamic_top_k"] is False


def test_single_hop_dynamic_top_k_requires_dynamic_hybrid(server):
    idx = _make_idx(server)
    with pytest.raises(ValueError, match="dynamic_top_k requires dynamic_hybrid"):
        idx.query(vector=[0.1, 0.2], dynamic_top_k=True)


def test_single_hop_dynamic_top_k_top_k_ceiling_still_applies(server):
    """`top_k` remains a hard ceiling. The server clips to
    min(top_k, K_pred); we just verify the SDK still enforces the
    dynamic_hybrid 100-cap."""
    idx = _make_idx(server)
    with pytest.raises(ValueError, match="top_k <= 100"):
        idx.query(vector=[0.1, 0.2], top_k=101,
                  dynamic_hybrid=True, dynamic_top_k=True)


# ── agentic toggle: dynamic_top_k forwards on multihop body ────────────────

def test_agentic_dynamic_top_k_forwarded(server):
    _stub_multihop_response("ix1")
    idx = _make_idx(server)
    idx.query("foo", agentic_search=True, mode="hybrid",
              dynamic_hybrid=True, dynamic_top_k=True, top_k=20)
    body = _last_body()
    assert body["dynamic_hybrid"] is True
    assert body["dynamic_top_k"] is True
    assert body["top_k"] == 20


def test_agentic_dynamic_top_k_off_by_default(server):
    _stub_multihop_response("ix1")
    idx = _make_idx(server)
    idx.query("foo", agentic_search=True, mode="hybrid",
              dynamic_hybrid=True)
    body = _last_body()
    assert body["dynamic_top_k"] is False


def test_agentic_dynamic_top_k_requires_dynamic_hybrid(server):
    idx = _make_idx(server)
    with pytest.raises(ValueError, match="dynamic_hybrid"):
        idx.query("foo", agentic_search=True, mode="hybrid",
                  dynamic_top_k=True)


# ── BYO surface: client.predict_dynamic_top_k ──────────────────────────────

def test_predict_dynamic_top_k_returns_struct(server):
    _Handler._responses["/v1/predict_dynamic_top_k"] = {
        "status": 200,
        "body": {
            "alpha": 0.42,
            "top_k_dense": 3,
            "top_k_hybrid": 5,
            "usage_current_month": 7,
        },
    }
    client = Client(api_key="dsk_test", base_url=server, max_retries=0)
    r = client.predict_dynamic_top_k("who founded apple?",
                                      query_vector=[0.1, 0.2, 0.3])
    assert isinstance(r, DynamicTopKResult)
    assert r.alpha == pytest.approx(0.42)
    assert r.top_k_dense == 3
    assert r.top_k_hybrid == 5

    # Body forwarded as expected
    body = _last_body()
    assert body["text"] == "who founded apple?"
    assert body["query_vector"] == [0.1, 0.2, 0.3]


def test_predict_dynamic_top_k_omits_query_vector_when_none(server):
    _Handler._responses["/v1/predict_dynamic_top_k"] = {
        "status": 200,
        "body": {"alpha": 0.5, "top_k_dense": 4, "top_k_hybrid": 8,
                 "usage_current_month": 1},
    }
    client = Client(api_key="dsk_test", base_url=server, max_retries=0)
    r = client.predict_dynamic_top_k("text only", model_id="bge-large-en-v1.5")
    assert r.top_k_hybrid == 8
    body = _last_body()
    assert "query_vector" not in body
    assert body["model_id"] == "bge-large-en-v1.5"


def test_predict_dynamic_top_k_rejects_empty_text(server):
    client = Client(api_key="dsk_test", base_url=server, max_retries=0)
    with pytest.raises(ValueError, match="non-empty"):
        client.predict_dynamic_top_k("")
    with pytest.raises(ValueError, match="non-empty"):
        client.predict_dynamic_top_k("   ")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
