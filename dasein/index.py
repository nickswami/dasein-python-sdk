"""
Index class: upsert, query, query_batch, delete, build, status.
"""
from __future__ import annotations

import base64
import io
import time
from typing import TYPE_CHECKING, Any

from dasein.types import QueryResult, QueryResponse, IndexInfo, UpsertItem
from dasein.exceptions import DaseinError, DaseinBuildError, DaseinUnavailableError

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# orjson is ~3-5x faster than stdlib json on large dicts/arrays and crucially
# releases the GIL during parsing, which matters when N client threads hammer
# query() concurrently. We fall back to stdlib silently so the SDK stays a
# pure drop-in with zero required extras.
try:
    import orjson as _orjson

    def _loads(data: bytes | str) -> Any:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return _orjson.loads(data)
except ImportError:
    import json as _json

    def _loads(data: bytes | str) -> Any:
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
        return _json.loads(data)


def _resp_json(resp) -> Any:
    """Parse an httpx.Response body using orjson if available, else stdlib.
    Use this on the query hot path; non-latency-sensitive callers can keep
    using resp.json()."""
    return _loads(resp.content)


def _decode_vector(v):
    """Decode the "vector" field of a result into an ndarray (or list) losslessly.

    The server emits either a JSON array of floats (legacy) or a base64 string
    of little-endian fp32 bytes (when the client requested vector_format=base64,
    which we only do when numpy is available). Both paths hit here.
    """
    if v is None:
        return None
    if isinstance(v, str):
        if not _HAS_NUMPY:
            return v
        # base64 -> raw bytes -> ndarray view. np.frombuffer is O(1) copy-free
        # over the decoded buffer and releases the GIL. Returning ndarray
        # instead of list avoids 1024 PyFloat allocations per candidate, which
        # is the difference between ~5 QPS and ~85 QPS per client thread.
        return np.frombuffer(base64.b64decode(v), dtype="<f4")
    if _HAS_NUMPY and isinstance(v, list):
        return np.asarray(v, dtype=np.float32)
    return v


if TYPE_CHECKING:
    from dasein.client import Client


class Index:
    """
    Represents a single Dasein index. Supports upsert, query, delete, build.

    Do not instantiate directly -- use Client.create_index() or Client.get_index().
    """

    def __init__(
        self,
        client: Client,
        index_id: str,
        model_id: str | None = None,
        index_type: str = "dense",
        dim: int = 1024,
        max_vectors: int | None = None,
    ):
        self._client = client
        self.index_id = index_id
        self.model_id = model_id
        self.index_type = index_type
        self.dim = dim
        self.max_vectors = max_vectors

    def upsert(self, documents: list[dict | UpsertItem]) -> dict:
        """
        Upsert documents into the index.

        Each document can contain:
          - id (required): unique document ID
          - text: raw text (we embed it if model is set)
          - vector: pre-computed embedding vector
          - metadata: dict of key-value pairs for filtering (values: str, int, or float)

        Args:
            documents: List of dicts or UpsertItem objects

        Returns:
            {"status": "ok", "count": N, "total": M}
        """
        if not documents:
            return {"status": "ok", "count": 0, "total": 0}

        docs = []
        for d in documents:
            if isinstance(d, UpsertItem):
                entry = {"id": d.id}
                if d.vector is not None:
                    entry["vector"] = d.vector
                if d.text is not None:
                    entry["text"] = d.text
                if d.metadata is not None:
                    entry["metadata"] = d.metadata
            elif isinstance(d, dict):
                entry = d
            else:
                raise ValueError(f"Expected dict or UpsertItem, got {type(d)}")
            docs.append(entry)

        MAX_BATCH = 2000
        results = []
        any_staged = False
        n_batches = (len(docs) + MAX_BATCH - 1) // MAX_BATCH
        for i in range(0, len(docs), MAX_BATCH):
            if i > 0 and n_batches > 1:
                time.sleep(0.15)
            batch = docs[i:i + MAX_BATCH]
            use_binary = _HAS_NUMPY and all("vector" in d for d in batch)
            if use_binary:
                resp = self._send_binary_batch(batch)
            else:
                resp = self._client._request(
                    "POST",
                    f"/indexes/{self.index_id}/upsert",
                    json={"documents": batch},
                    timeout=300.0,
                )
            batch_result = resp.json()
            if batch_result.get("status") == "staged":
                any_staged = True
            results.append(batch_result)

        if len(results) == 1:
            return results[0]

        status = "staged" if any_staged else "ok"
        merged = {
            "status": status,
            "count": sum(r.get("count", 0) for r in results),
            "total": results[-1].get("total", 0),
        }
        if any_staged:
            merged["message"] = "One or more batches required a rebuild — data will be queryable after rebuild completes."
        return merged

    def _send_binary_batch(self, batch: list[dict]):
        """Pack BYOV batch as NPZ and POST as binary — ~3x faster than JSON."""
        ids = np.array([str(d["id"]) for d in batch], dtype=object)
        vectors = np.array([d["vector"] for d in batch], dtype=np.float32)
        kw: dict = {"ids": ids, "vectors": vectors}
        texts = [d.get("text") for d in batch]
        if any(t is not None for t in texts):
            kw["texts"] = np.array(texts, dtype=object)
        metas = [d.get("metadata") for d in batch]
        if any(m is not None for m in metas):
            kw["metadatas"] = np.array(metas, dtype=object)
        buf = io.BytesIO()
        np.savez(buf, **kw)
        return self._client._request(
            "POST",
            f"/indexes/{self.index_id}/upsert-binary",
            content=buf.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=120.0,
        )

    def upsert_and_wait(self, documents: list[dict | UpsertItem],
                        timeout: float = 120.0) -> dict:
        """
        Upsert documents and wait for the index to become queryable (active).

        If upsert returns "staged" (live sync failed, rebuild queued), polls
        until the rebuild completes and the index transitions back to active.
        Also handles building → built → placing → active for first-time upserts.
        """
        result = self.upsert(documents)

        needs_rebuild = result.get("status") == "staged" or result.get("live_sync") is False

        start = time.time()
        seen_non_active = False
        while time.time() - start < timeout:
            info = self.status()
            if info.status != "active":
                seen_non_active = True
            if info.status == "active":
                if needs_rebuild and not seen_non_active:
                    time.sleep(1)
                    continue
                result["index_status"] = "active"
                if result.get("status") == "staged":
                    result["live_sync"] = False
                return result
            if info.status == "requires_build":
                result["index_status"] = "requires_build"
                result["message"] = (
                    "Index has data but needs an explicit build (BYOV with unknown model). "
                    "Call index.build() to start the build."
                )
                return result
            if info.status == "build_failed":
                raise DaseinBuildError(f"Build failed for index {self.index_id}")
            time.sleep(2)

        info = self.status()
        if info.status in ("built", "building", "placing"):
            result["index_status"] = info.status
            result["message"] = (
                f"Index is {info.status} but not yet active. "
                "This usually resolves shortly — try again."
            )
            return result

        raise DaseinUnavailableError(
            f"Index {self.index_id} did not become active within {timeout}s (status: {info.status})"
        )

    def query(
        self,
        text: str | None = None,
        vector: list[float] | None = None,
        top_k: int = 10,
        mode: str = "dense",
        filter: dict[str, Any] | None = None,
        exact: bool = False,
        phrase: bool = False,
        fuzzy: bool = False,
        alpha: float = 0.5,
        include_text: bool = False,
        include_metadata: bool = False,
        include_vectors: bool = False,
    ) -> QueryResponse:
        """
        Query the index.

        Provide either text (we embed it) or vector (pre-computed).

        Args:
            text: Query text (requires model to be set on the index)
            vector: Query vector (list of floats)
            top_k: Number of results to return
            mode: "dense" or "hybrid"
            filter: Metadata filter dict. Supports equality, $ne, $in, $nin,
                $exists, $gt/$gte/$lt/$lte, and $or. Example::

                    {"genre": "sci-fi", "year": {"$gte": 2020}}

            exact: Exact keyword matching -- only return docs containing all query terms
            phrase: Phrase matching -- only return docs containing the query as an exact phrase
            fuzzy: Fuzzy matching -- match keywords with typo tolerance (edit distance 1)
            alpha: Balance between dense and BM25 in hybrid RRF fusion.
                0.0 = all dense, 1.0 = all BM25, 0.5 = equal (default).
            include_text: Return stored text in results (requires SSD read, default False).
            include_metadata: Return stored metadata in results (requires SSD read, default False).
            include_vectors: Return reconstructed approximate vectors (from RAM, default False).

        Returns:
            QueryResponse (iterable like a list of QueryResult, with timing attrs)
        """
        if text is None and vector is None:
            raise ValueError("Either text or vector must be provided")

        payload: dict[str, Any] = {"top_k": top_k, "mode": mode}
        if text is not None:
            payload["text"] = text
        if vector is not None:
            payload["vector"] = vector
        if filter is not None:
            payload["filter"] = filter
        if exact:
            payload["exact"] = True
        if phrase:
            payload["phrase"] = True
        if fuzzy:
            payload["fuzzy"] = True
        if alpha != 0.5:
            payload["alpha"] = alpha
        if include_text:
            payload["include_text"] = True
        if include_metadata:
            payload["include_metadata"] = True
        if include_vectors:
            payload["include_vectors"] = True
            # Ask the server for base64-encoded fp32 vectors when we can
            # decode them natively. This turns a 1.5 MB JSON-float-array
            # response into a ~550 KB string that np.frombuffer decodes
            # outside the GIL — fixes the 20x throughput cliff we hit at
            # high concurrency when include_vectors=True.
            if _HAS_NUMPY:
                payload["vector_format"] = "base64"

        t0 = time.perf_counter()
        resp = self._client._request(
            "POST",
            f"/v1/indexes/{self.index_id}/query",
            json=payload,
        )
        round_trip_ms = (time.perf_counter() - t0) * 1000

        h = resp.headers
        data = _resp_json(resp)
        results = [
            QueryResult(
                id=r["id"],
                score=r.get("score", 0.0),
                text=r.get("text"),
                metadata=r.get("metadata"),
                vector=_decode_vector(r.get("vector")),
            )
            for r in data.get("results", [])
        ]
        return QueryResponse(
            results=results,
            round_trip_ms=round_trip_ms,
            server_total_us=int(h.get("x-total-us", 0)),
            search_us=int(h.get("x-search-us", 0) or h.get("x-daemon-us", 0)),
            embed_us=int(h.get("x-embed-us", 0) or h.get("x-build-us", 0)),
            auth_us=int(h.get("x-auth-us", 0)),
            rate_us=int(h.get("x-rate-us", 0)),
            route_us=int(h.get("x-route-us", 0)),
            resp_us=int(h.get("x-resp-us", 0)),
        )

    def query_batch(
        self,
        queries: list[dict[str, Any]],
    ) -> list[QueryResponse]:
        """Run many queries in one HTTP round-trip.

        ``query_batch`` is functionally identical to calling ``query()`` N
        times — same server-side search path, same RRF hybrid fusion, same
        filter operators, same flags (``mode``, ``exact``, ``phrase``,
        ``fuzzy``, ``alpha``), same ``include_text`` / ``include_metadata`` /
        ``include_vectors`` options. The only difference is that up to
        4096 queries travel on one TCP connection in one JSON payload,
        amortizing HTTP / TLS / router overhead over the whole batch.

        Args:
            queries: list of per-query parameter dicts. Each dict takes
                the same keys as ``Index.query(...)`` — e.g. ``vector``,
                ``text``, ``top_k``, ``mode``, ``filter``, ``exact``,
                ``phrase``, ``fuzzy``, ``alpha``, ``include_text``,
                ``include_metadata``, ``include_vectors``. A single query
                must supply either ``vector`` or ``text`` (hybrid indexes
                with a model can auto-embed ``text`` just like
                ``query()``).

        Returns:
            list[QueryResponse] — one entry per input query, in the same
            order. Malformed entries come back as empty result sets rather
            than failing the whole batch; re-run those slots individually
            to get the 400-class error message.
        """
        if not queries:
            return []
        if len(queries) > 4096:
            raise ValueError("query_batch: max 4096 queries per call")

        norm_queries = []
        for q in queries:
            if not isinstance(q, dict):
                raise ValueError("query_batch: each query must be a dict")
            entry: dict[str, Any] = {}
            # Normalize accepted keys; accept ndarray vectors (fast path).
            if "vector" in q and q["vector"] is not None:
                v = q["vector"]
                if _HAS_NUMPY and isinstance(v, np.ndarray):
                    v = v.astype(np.float32, copy=False).tolist()
                entry["vector"] = v
            for key in (
                "text", "top_k", "mode", "filter",
                "exact", "phrase", "fuzzy", "alpha",
                "include_text", "include_metadata", "include_vectors",
            ):
                if key in q and q[key] is not None:
                    entry[key] = q[key]
            # Ask for base64 vectors whenever numpy can decode them. The
            # server emits one setting for the whole response, so if ANY
            # sub-query wants vectors we opt the entire batch in. Callers
            # that don't set include_vectors won't see any "vector" fields
            # regardless.
            if _HAS_NUMPY and entry.get("include_vectors"):
                entry["vector_format"] = "base64"
            norm_queries.append(entry)

        payload = {"queries": norm_queries}

        t0 = time.perf_counter()
        resp = self._client._request(
            "POST",
            f"/v1/indexes/{self.index_id}/batch_query",
            json=payload,
            timeout=120.0,
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
        total_us = int(h.get("x-total-us", 0))

        out: list[QueryResponse] = []
        for b in batches:
            results = [
                QueryResult(
                    id=r["id"],
                    score=r.get("score", 0.0),
                    text=r.get("text"),
                    metadata=r.get("metadata"),
                    vector=_decode_vector(r.get("vector")),
                )
                for r in b.get("results", [])
            ]
            out.append(QueryResponse(
                results=results,
                round_trip_ms=round_trip_ms,
                server_total_us=total_us,
                search_us=search_us,
            ))
        return out

    def delete(self, ids: list[str | int]) -> dict:
        """Delete documents by ID. Automatically batches if more than 1000 IDs."""
        _BATCH = 1000
        if len(ids) <= _BATCH:
            resp = self._client._request(
                "DELETE",
                f"/indexes/{self.index_id}/documents",
                json={"ids": ids},
            )
            return resp.json()

        total_tombstones = 0
        last_result: dict = {}
        for start in range(0, len(ids), _BATCH):
            batch = ids[start : start + _BATCH]
            resp = self._client._request(
                "DELETE",
                f"/indexes/{self.index_id}/documents",
                json={"ids": batch},
            )
            last_result = resp.json()
            total_tombstones += last_result.get("tombstones_staged", len(batch))
        last_result["tombstones_staged"] = total_tombstones
        return last_result

    def build(self) -> dict:
        """
        Trigger an explicit index build.
        Only needed for BYOV indexes with unknown embedding models.
        """
        resp = self._client._request(
            "POST",
            f"/indexes/{self.index_id}/build",
        )
        return resp.json()

    def compact(self) -> dict:
        """
        Trigger compaction: rebuild graph discarding tombstones.
        """
        resp = self._client._request(
            "POST",
            f"/indexes/{self.index_id}/compact",
        )
        return resp.json()

    def status(self) -> IndexInfo:
        """Get current index status and metadata."""
        return self._client.index_info(self.index_id)

    def __repr__(self) -> str:
        return f"Index(id={self.index_id!r}, index_type={self.index_type!r}, model={self.model_id!r})"
