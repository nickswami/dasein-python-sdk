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

import time
from typing import Optional

import httpx

from dasein.index import Index
from dasein.types import IndexInfo
from dasein.exceptions import (
    DaseinAuthError,
    DaseinQuotaError,
    DaseinNotFoundError,
    DaseinUnavailableError,
    DaseinRateLimitError,
    DaseinError,
)

DEFAULT_BASE_URL = "https://dasein-api-939340394421.us-central1.run.app"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
__version__ = "0.1.0"


class Client:
    """
    Dasein API client.

    Args:
        api_key: Your Dasein API key (starts with dsk_)
        base_url: API base URL (defaults to Cloud Run URL)
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
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    @staticmethod
    def _extract_detail(resp: httpx.Response) -> str:
        try:
            return resp.json().get("detail", resp.text)
        except Exception:
            return resp.text or f"HTTP {resp.status_code}"

    _IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "DELETE"})
    _SAFE_TO_RETRY_PATHS = frozenset({"/query"})
    _IDEMPOTENT_POST_PATHS = frozenset({"/query", "/upsert"})

    def _is_safe_retry(self, method: str, path: str, status: int = 503) -> bool:
        """503/504/connection retries are only safe for idempotent methods or read-only POSTs.

        Exception: 504 on /upsert is retryable because upserts have replace
        semantics (idempotent by document ID) and 504 indicates the gateway
        timed out before processing began.
        """
        if method.upper() in self._IDEMPOTENT_METHODS:
            return True
        if any(path.endswith(s) for s in self._SAFE_TO_RETRY_PATHS):
            return True
        if status == 504 and any(path.endswith(s) for s in self._IDEMPOTENT_POST_PATHS):
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
                    if any(kw in detail_lower for kw in _QUOTA_KEYWORDS):
                        raise DaseinQuotaError(detail)
                    raise DaseinAuthError(detail)
                if resp.status_code == 404:
                    raise DaseinNotFoundError(self._extract_detail(resp))

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    if attempt < self.max_retries:
                        time.sleep(max(retry_after, 2 ** attempt))
                        continue
                    raise DaseinRateLimitError(
                        self._extract_detail(resp) or "Rate limit exceeded",
                        retry_after=retry_after,
                    )

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
        model: str | None = None,
        plan: str = "dense",
        dim: int | None = None,
    ) -> Index:
        """
        Create a new index.

        Args:
            name: Human-readable index name
            model: Embedding model ID (e.g., "bge-large-en-v1.5"). None for BYOV.
            plan: Index type — "dense" or "hybrid". Trial accounts get trial-tier
                  limits regardless of the requested plan; the returned Index reflects
                  the effective plan assigned by the server.
            dim: Override embedding dimension for Matryoshka-capable models. The
                 model's full-dimension embeddings are truncated and renormalized to
                 this size. Pass None (default) to use the model's native dimension.

        Returns:
            Index object for upserting and querying
        """
        body: dict = {"name": name, "model_id": model, "plan": plan}
        if dim is not None:
            body["dim"] = dim
        resp = self._request("POST", "/indexes", json=body)
        data = resp.json()
        return Index(
            client=self,
            index_id=data["index_id"],
            model_id=model,
            plan=data.get("plan", plan),
            dim=data.get("dim", 1024),
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
            plan=data.get("plan", "dense"),
            dim=data.get("dim", 1024),
        )

    def index_info(self, index_id: str) -> IndexInfo:
        """Get detailed index information."""
        resp = self._request("GET", f"/indexes/{index_id}")
        data = resp.json()
        return IndexInfo.from_dict(data)

    def delete_index(self, index_id: str) -> None:
        """Delete an index permanently."""
        self._request("DELETE", f"/indexes/{index_id}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
