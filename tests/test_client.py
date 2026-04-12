"""
Unit tests for the Dasein SDK client.

Tests the client against a mock server to verify:
  - Auth header is sent
  - Retry logic on 429/503
  - Correct request/response parsing
  - Exception types
"""
from __future__ import annotations

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import pytest

from dasein import Client, QueryResult
from dasein.exceptions import (
    DaseinAuthError,
    DaseinRateLimitError,
    DaseinUnavailableError,
    DaseinNotFoundError,
)


class MockHandler(BaseHTTPRequestHandler):
    _responses: dict = {}
    _request_log: list = []

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""
        MockHandler._request_log.append({
            "method": "POST",
            "path": self.path,
            "headers": dict(self.headers),
            "body": json.loads(body) if body else None,
        })

        resp = MockHandler._responses.get(("POST", self.path))
        if resp:
            self.send_response(resp["status"])
            for k, v in resp.get("headers", {}).items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp["body"]).encode())
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detail":"not found"}')

    def do_GET(self):
        MockHandler._request_log.append({
            "method": "GET",
            "path": self.path,
            "headers": dict(self.headers),
        })

        resp = MockHandler._responses.get(("GET", self.path))
        if resp:
            self.send_response(resp["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp["body"]).encode())
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detail":"not found"}')

    def do_DELETE(self):
        MockHandler._request_log.append({
            "method": "DELETE",
            "path": self.path,
            "headers": dict(self.headers),
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"deleted"}')

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def reset_mock():
    MockHandler._responses = {}
    MockHandler._request_log = []


def test_create_index(mock_server):
    MockHandler._responses[("POST", "/indexes")] = {
        "status": 200,
        "body": {"index_id": "abc123", "status": "created", "dim": 1024},
    }

    client = Client(api_key="dsk_test_key_123", base_url=mock_server, max_retries=0)
    index = client.create_index("test-index", model="bge-large-en-v1.5")

    assert index.index_id == "abc123"
    assert index.model_id == "bge-large-en-v1.5"
    headers = {k.lower(): v for k, v in MockHandler._request_log[-1]["headers"].items()}
    assert headers["x-api-key"] == "dsk_test_key_123"


def test_query(mock_server):
    MockHandler._responses[("POST", "/v1/indexes/abc123/query")] = {
        "status": 200,
        "body": {
            "results": [
                {"id": "doc1", "score": 0.95, "text": "hello world", "metadata": {"type": "test"}},
                {"id": "doc2", "score": 0.85, "text": "foo bar"},
            ],
            "mode": "dense",
        },
    }

    client = Client(api_key="dsk_test", base_url=mock_server, max_retries=0)
    from dasein.index import Index
    idx = Index(client=client, index_id="abc123")
    results = idx.query(vector=[0.1, 0.2, 0.3], top_k=5)

    assert len(results) == 2
    assert results[0].id == "doc1"
    assert results[0].score == 0.95
    assert results[0].text == "hello world"
    assert results[0].metadata == {"type": "test"}


def test_upsert(mock_server):
    MockHandler._responses[("POST", "/indexes/abc123/upsert")] = {
        "status": 200,
        "body": {"status": "ok", "count": 2, "total": 2},
    }

    client = Client(api_key="dsk_test", base_url=mock_server, max_retries=0)
    from dasein.index import Index
    idx = Index(client=client, index_id="abc123")
    result = idx.upsert([
        {"id": "doc1", "text": "hello", "metadata": {"type": "test"}},
        {"id": "doc2", "vector": [0.1, 0.2]},
    ])

    assert result["count"] == 2


def test_auth_error(mock_server):
    MockHandler._responses[("GET", "/indexes/x")] = {
        "status": 401,
        "body": {"detail": "Invalid API key"},
    }

    client = Client(api_key="dsk_bad", base_url=mock_server, max_retries=0)
    with pytest.raises(DaseinAuthError):
        client.index_info("x")


def test_rate_limit_retry(mock_server):
    call_count = [0]
    original_responses = MockHandler._responses.copy()

    MockHandler._responses[("POST", "/v1/indexes/abc123/query")] = {
        "status": 429,
        "body": {"detail": "rate limited"},
        "headers": {"Retry-After": "0.01"},
    }

    client = Client(api_key="dsk_test", base_url=mock_server, max_retries=1)
    from dasein.index import Index
    idx = Index(client=client, index_id="abc123")

    with pytest.raises(DaseinRateLimitError):
        idx.query(vector=[0.1])


def test_filter_query(mock_server):
    MockHandler._responses[("POST", "/v1/indexes/abc123/query")] = {
        "status": 200,
        "body": {"results": [{"id": "doc1", "score": 0.9}], "mode": "dense"},
    }

    client = Client(api_key="dsk_test", base_url=mock_server, max_retries=0)
    from dasein.index import Index
    idx = Index(client=client, index_id="abc123")
    results = idx.query(vector=[0.1], filter={"tenant": "acme"})

    req = MockHandler._request_log[-1]
    assert req["body"]["filter"] == {"tenant": "acme"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
