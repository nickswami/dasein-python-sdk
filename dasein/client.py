"""
Dasein Python SDK client.

Usage:
    from dasein import Client

    client = Client(api_key="dsk_...")
    index = client.create_index("my-docs", model="bge-large-en-v1.5")
    index.upsert([{"id": "doc1", "text": "Hello world", "metadata": {"type": "greeting"}}])
    results = index.query("hello", top_k=5)
"""
from __future__ import annotations

import os
import time
from typing import Any
import httpx

from dasein.index import Index, _decode_vector, _resp_json
from dasein.types import IndexInfo, QueryResult, QueryResponse
from dasein.exceptions import (
    DaseinAuthError,
    DaseinQuotaError,
    DaseinNotFoundError,
    DaseinUnavailableError,
    DaseinRateLimitError,
    DaseinError,
)

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

DEFAULT_BASE_URL = "https://api.daseinai.ai"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
__version__ = "0.4.8"


class Client:
    """
    Dasein API client.

    Args:
        api_key: Your Dasein API key (starts with dsk_)
        base_url: API base URL (defaults to https://api.daseinai.ai)
        timeout: Request timeout in seconds
        max_retries: Max retries for 429/503 responses
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        if not api_key:
            raise DaseinAuthError("API key is required")

        self.api_key = api_key
        self.base_url = (base_url or os.environ.get("DASEIN_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "X-API-Key": api_key,
                "Authorization": f"Bearer {api_key}",
            },
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=1000,
                max_keepalive_connections=200,
            ),
        )

    @staticmethod
    def _extract_detail(resp: httpx.Response) -> str:
        try:
            return resp.json().get("detail", resp.text)
        except Exception:
            return resp.text or f"HTTP {resp.status_code}"

    _IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "DELETE"})
    _IDEMPOTENT_POST_PATHS = frozenset({"/query", "/upsert", "/upsert-binary", "/batch_query"})

    def _is_safe_retry(self, method: str, path: str, status: int = 503) -> bool:
        """503/504/connection retries are only safe for idempotent methods or read-only POSTs.

        Upserts are idempotent by document ID (replace semantics), so 503/504
        are safe to retry — the worst case is re-embedding the same texts.
        """
        if method.upper() in self._IDEMPOTENT_METHODS:
            return True
        if any(path.endswith(s) for s in self._IDEMPOTENT_POST_PATHS):
            return True
        return False

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make an HTTP request with retry logic for 429/503.

        429 is always retried (request was rejected, not processed).
        503/504 and connection errors are only retried for idempotent
        methods or known-safe read paths (e.g., /query) to avoid
        duplicating side effects on upsert/build/delete.
        """
        safe_retry_base = self._is_safe_retry(method, path)
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.request(method, path, **kwargs)

                if resp.status_code == 401:
                    raise DaseinAuthError(self._extract_detail(resp))
                if resp.status_code == 403:
                    detail = self._extract_detail(resp)
                    detail_lower = detail.lower()
                    _QUOTA_KEYWORDS = (
                        "limit", "quota", "plan", "trial", "upgrade", "exceed",
                        "subscription", "past due", "expired", "subscribe",
                        "payment", "reactivate",
                    )
                    _AUTH_KEYWORDS = (
                        "api key", "api_key", "credential", "revoked",
                        "invalid key", "missing key", "authentication",
                    )
                    if any(kw in detail_lower for kw in _QUOTA_KEYWORDS):
                        raise DaseinQuotaError(detail)
                    if any(kw in detail_lower for kw in _AUTH_KEYWORDS):
                        raise DaseinAuthError(detail)
                    raise DaseinError(detail)
                if resp.status_code == 404:
                    raise DaseinNotFoundError(self._extract_detail(resp))

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    detail = self._extract_detail(resp) or "Rate limit exceeded"
                    detail_lower = detail.lower()
                    is_quota_cap = any(kw in detail_lower for kw in ("embed", "quota", "monthly", "allowance"))
                    if is_quota_cap:
                        raise DaseinQuotaError(detail)
                    if attempt < self.max_retries:
                        time.sleep(max(retry_after, 2 ** attempt))
                        continue
                    raise DaseinRateLimitError(detail, retry_after=retry_after)

                if resp.status_code in (503, 504):
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    safe_retry = self._is_safe_retry(method, path, status=resp.status_code)
                    if safe_retry and attempt < self.max_retries:
                        time.sleep(max(retry_after, 2 ** attempt))
                        continue
                    raise DaseinUnavailableError(
                        self._extract_detail(resp), retry_after=retry_after)

                if resp.status_code < 300:
                    return resp

                raise DaseinError(self._extract_detail(resp))

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if safe_retry_base and attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise DaseinUnavailableError(f"Connection failed: {e}")

        raise DaseinError(f"Request failed after {self.max_retries} retries: {last_error}")

    def create_index(
        self,
        name: str,
        index_type: str = "dense",
        model: str | None = None,
        dim: int | None = None,
    ) -> Index:
        """
        Create a new index.

        Args:
            name: Human-readable index name
            index_type: "dense" (semantic only) or "hybrid" (semantic + BM25 keyword search).
                You must create a hybrid index to use mode="hybrid" in queries.
            model: Embedding model ID (e.g., "bge-large-en-v1.5"). None for BYOV.
            dim: Override embedding dimension for Matryoshka-capable models. The
                 model's full-dimension embeddings are truncated and renormalized to
                 this size. Pass None (default) to use the model's native dimension.

        Returns:
            Index object for upserting and querying
        """
        body: dict = {"name": name, "model_id": model, "index_type": index_type}
        if dim is not None:
            body["dim"] = dim
        resp = self._request("POST", "/indexes", json=body)
        data = resp.json()
        return Index(
            client=self,
            index_id=data["index_id"],
            model_id=model,
            index_type=data.get("index_type", data.get("plan", index_type)),
            dim=data.get("dim", 1024),
            max_vectors=data.get("max_vectors"),
        )

    def list_indexes(self) -> list[dict]:
        """List all indexes owned by the authenticated user."""
        resp = self._request("GET", "/indexes")
        return resp.json().get("indexes", [])

    def get_index(self, index_id: str) -> Index:
        """Get an existing index by ID."""
        resp = self._request("GET", f"/indexes/{index_id}")
        data = resp.json()
        return Index(
            client=self,
            index_id=data["index_id"],
            model_id=data.get("model_id"),
            index_type=data.get("index_type", data.get("plan", "dense")),
            dim=data.get("dim", 1024),
            max_vectors=data.get("max_vectors"),
        )

    def index_info(self, index_id: str) -> IndexInfo:
        """Get detailed index information."""
        resp = self._request("GET", f"/indexes/{index_id}")
        data = resp.json()
        return IndexInfo.from_dict(data)

    def delete_index(self, index_id: str) -> None:
        """Delete an index permanently."""
        self._request("DELETE", f"/indexes/{index_id}")

    def predict_alpha(
        self,
        text: str,
        query_vector: list[float] | None = None,
        model_id: str | None = None,
    ) -> float:
        """Managed per-query hybrid-fusion weight.

        Call this from *any* search stack — you don't need a Dasein index.
        Dasein returns ``alpha ∈ [0.0, 1.0]`` for the given query: blend
        your own dense and BM25 rankings at that weight and ship the fused
        list.

        - 0.0  →  use only your dense ranking
        - 1.0  →  use only your BM25 ranking
        - 0.5  →  equal blend

        Typical usage::

            qvec = my_encoder.encode("who founded apple?")
            alpha = client.predict_alpha("who founded apple?", query_vector=qvec)
            fused = rrf_fuse(dense_hits, bm25_hits, alpha=alpha)

        Args:
            text: The raw query text.
            query_vector: The dense query embedding from YOUR encoder.
                Strongly recommended — the returned alpha is tied to the
                geometry of this vector, so for alpha to be valid for
                your retriever you must pass the same vector you're about
                to retrieve with. If omitted, Dasein embeds ``text`` with
                its default model and the alpha will only be meaningful
                for that model's geometry. Counts against your embed
                token quota when omitted.
            model_id: Optional override for the embedding model when
                ``query_vector`` is not supplied. Ignored otherwise.

        Returns:
            float in [0.0, 1.0].

        Quota:
            Free plans: 1,000 calls per month.
            Paid hybrid plans: unlimited.
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        payload: dict[str, Any] = {"text": text}
        if query_vector is not None:
            payload["query_vector"] = list(query_vector)
        if model_id is not None:
            payload["model_id"] = model_id
        resp = self._request("POST", "/v1/predict_alpha", json=payload)
        data = resp.json()
        return float(data["alpha"])

    def query_batch(
        self,
        queries: list[dict[str, Any]],
    ) -> list[QueryResponse]:
        """Run many queries in a single HTTP round-trip, fanned out across any
        mix of indexes — same feature surface as ``Index.query()``.

        Each entry is a dict with all the kwargs ``Index.query()`` accepts,
        plus the target index. Vectors may be passed as ``list[float]`` or
        ``numpy.ndarray``; text sub-queries are auto-embedded server-side
        using the index's model. A single request can hit hundreds of
        different indexes — the router buckets by index, fans out to the
        owning pods in parallel, and coalesces embed calls by model so a
        batch of 256 texts across 60 indexes costs O(distinct models) embed
        calls, not 256.

        Args:
            queries: list of per-query dicts. Each dict must include::

                {
                    "index_id":  "<uuid>",          # required
                    "text":      "hello",           # OR "vector"
                    "vector":    [0.1, 0.2, ...],   # OR "text"
                    "top_k":     10,
                    "mode":      "dense" | "hybrid",
                    "filter":    {...},             # optional metadata filter
                    "exact":     True,              # optional keyword match flags
                    "phrase":    True,
                    "fuzzy":     True,
                    "alpha":     0.5,
                    "include_text":     False,
                    "include_metadata": False,
                    "include_vectors":  True,
                }

        Returns:
            list[QueryResponse] — one entry per input query, in request
            order. Per-slot failures (index not loaded, auth failure,
            malformed query) come back as empty result sets with the HTTP
            body's ``error`` field preserved on the response; they do not
            fail the whole batch. Re-run the slot individually to get the
            same error you would have seen from ``query()``.

        Raises:
            DaseinError: if the top-level request itself fails (network,
                auth, 5xx from the router). Per-slot errors do NOT raise.
        """
        if not queries:
            return []
        if len(queries) > 4096:
            raise ValueError("query_batch: max 4096 queries per call")

        import base64 as _b64

        norm: list[dict[str, Any]] = []
        any_wants_vec = False
        for q in queries:
            if not isinstance(q, dict):
                raise ValueError("query_batch: each query must be a dict")
            if not q.get("index_id"):
                raise ValueError("query_batch: each query must include 'index_id'")
            entry: dict[str, Any] = {"index_id": q["index_id"]}

            wire_b64 = False
            v = q.get("vector")
            if v is not None:
                if _HAS_NUMPY:
                    arr = _np.asarray(v, dtype=_np.float32)
                    entry["vector"] = _b64.b64encode(arr.tobytes()).decode("ascii")
                    wire_b64 = True
                else:
                    entry["vector"] = list(v)

            for key in (
                "text", "top_k", "mode", "filter",
                "exact", "phrase", "fuzzy", "alpha",
                "include_text", "include_metadata", "include_vectors",
            ):
                if key in q and q[key] is not None:
                    entry[key] = q[key]

            if entry.get("include_vectors"):
                any_wants_vec = True
                if _HAS_NUMPY:
                    wire_b64 = True
            if wire_b64:
                entry["vector_format"] = "base64"
            norm.append(entry)

        payload = {"queries": norm}

        t0 = time.perf_counter()
        resp = self._request(
            "POST",
            "/v1/batch_query",
            json=payload,
            timeout=300.0,
        )
        round_trip_ms = (time.perf_counter() - t0) * 1000

        data = _resp_json(resp)
        batches = data.get("batches", [])
        if len(batches) != len(queries):
            raise DaseinError(
                f"Batch response mismatch: sent {len(queries)} queries, "
                f"got {len(batches)} result sets"
            )

        h = resp.headers
        search_us = int(h.get("x-search-us", 0))
        total_us  = int(h.get("x-total-us",  0))

        out: list[QueryResponse] = []
        for b in batches:
            results = [
                QueryResult(
                    id=r["id"],
                    score=r.get("score", 0.0),
                    text=r.get("text"),
                    metadata=r.get("metadata"),
                    vector=_decode_vector(r.get("vector")) if any_wants_vec else None,
                )
                for r in b.get("results", [])
            ]
            out.append(QueryResponse(
                results=results,
                round_trip_ms=round_trip_ms,
                server_total_us=total_us,
                search_us=search_us,
                error=b.get("error"),
            ))
        return out

    def multihop_external(self, question: str, retriever_callback,
                          min_hops: int | None = None,
                          max_hops: int | None = None,
                          r_seed: int | None = None) -> dict:
        """Run a fully-external multi-hop search. See dasein.multihop_external."""
        from dasein.index import multihop_external as _mh_ext
        return _mh_ext(self, question, retriever_callback,
                       min_hops=min_hops, max_hops=max_hops, r_seed=r_seed)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
