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
        Upsert documents and wait for the index to become queryable (active).

        Polls through building → built → placing → active. Returns early
        with guidance if the index needs an explicit build (BYOV).
        """
        result = self.upsert(documents)

        if result.get("status") == "staged":
            result["index_status"] = "staged"
            result["live_sync"] = False
            return result

        start = time.time()
        while time.time() - start < timeout:
            info = self.status()
            if info.status == "active":
                result["index_status"] = "active"
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
        if info.status == "built":
            result["index_status"] = "built"
            result["message"] = (
                "Index built but not yet placed on a serving host. "
                "This usually resolves within 60 seconds — try again shortly."
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
        filter: dict[str, str] | None = None,
        exact: bool = False,
        phrase: bool = False,
        fuzzy: bool = False,
        alpha: float = 0.5,
        include_text: bool = False,
        include_metadata: bool = True,
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
            exact: Exact keyword matching -- only return docs containing all query terms
            phrase: Phrase matching -- only return docs containing the query as an exact phrase
            fuzzy: Fuzzy matching -- match keywords with typo tolerance (edit distance 1)
            alpha: Balance between dense and BM25 in hybrid RRF fusion.
                0.0 = all dense, 1.0 = all BM25, 0.5 = equal (default).
            include_text: Return stored text in results (requires SSD read, default False).
            include_metadata: Return stored metadata in results (requires SSD read, default True).

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
        if not include_metadata:
            payload["include_metadata"] = False

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
