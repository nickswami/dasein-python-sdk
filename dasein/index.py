"""
Index class: upsert, query, delete, build, status.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from dasein.types import QueryResult, IndexInfo, UpsertItem
from dasein.exceptions import DaseinError, DaseinBuildError, DaseinUnavailableError

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
        plan: str = "dense",
        dim: int = 1024,
    ):
        self._client = client
        self.index_id = index_id
        self.model_id = model_id
        self.plan = plan
        self.dim = dim

    def upsert(self, documents: list[dict | UpsertItem]) -> dict:
        """
        Upsert documents into the index.

        Each document can contain:
          - id (required): unique document ID
          - text: raw text (we embed it if model is set)
          - vector: pre-computed embedding vector
          - metadata: dict of string key-value pairs for filtering

        Args:
            documents: List of dicts or UpsertItem objects

        Returns:
            {"status": "ok", "count": N, "total": M}
        """
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

        MAX_BATCH = 100
        results = []
        for i in range(0, len(docs), MAX_BATCH):
            batch = docs[i:i + MAX_BATCH]
            resp = self._client._request(
                "POST",
                f"/indexes/{self.index_id}/upsert",
                json={"documents": batch},
            )
            results.append(resp.json())

        if len(results) == 1:
            return results[0]
        return {
            "status": "ok",
            "count": sum(r.get("count", 0) for r in results),
            "total": results[-1].get("total", 0),
        }

    def upsert_and_wait(self, documents: list[dict | UpsertItem],
                        timeout: float = 120.0) -> dict:
        """
        Upsert documents and wait for the index to become queryable (active)
        or fully built (built). If the index reaches 'built' but not 'active',
        the user likely needs to activate a subscription first.
        """
        result = self.upsert(documents)

        start = time.time()
        while time.time() - start < timeout:
            info = self.status()
            if info.status == "active":
                return result
            if info.status == "built":
                result["index_status"] = "built"
                result["message"] = "Index built successfully. Activate a subscription to make it queryable."
                return result
            if info.status in ("build_failed",):
                raise DaseinBuildError(f"Build failed for index {self.index_id}")
            time.sleep(2)

        raise DaseinUnavailableError(f"Index {self.index_id} did not finish building within {timeout}s")

    def query(
        self,
        text: str | None = None,
        vector: list[float] | None = None,
        top_k: int = 10,
        mode: str = "dense",
        filter: dict[str, str] | None = None,
        exact_rescore: bool = False,
    ) -> list[QueryResult]:
        """
        Query the index.

        Provide either text (we embed it) or vector (pre-computed).

        Args:
            text: Query text (requires model to be set on the index)
            vector: Query vector (list of floats)
            top_k: Number of results to return
            mode: "dense" or "hybrid"
            filter: Metadata filter dict, e.g. {"tenant": "acme"}
            exact_rescore: If True and mode is hybrid, use exact BM25 rescore

        Returns:
            List of QueryResult objects
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
        if exact_rescore:
            payload["exact_rescore"] = True

        resp = self._client._request(
            "POST",
            f"/indexes/{self.index_id}/query",
            json=payload,
        )
        data = resp.json()
        return [
            QueryResult(
                id=r["id"],
                score=r.get("score", 0.0),
                text=r.get("text"),
                metadata=r.get("metadata"),
            )
            for r in data.get("results", [])
        ]

    def delete(self, ids: list[str | int]) -> dict:
        """Delete documents by ID."""
        resp = self._client._request(
            "DELETE",
            f"/indexes/{self.index_id}/documents",
            json={"ids": ids},
        )
        return resp.json()

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
        return f"Index(id={self.index_id!r}, plan={self.plan!r}, model={self.model_id!r})"
