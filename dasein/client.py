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
    DaseinNotFoundError,
    DaseinUnavailableError,
    DaseinRateLimitError,
    DaseinError,
)

DEFAULT_BASE_URL = "https://dasein-api-jf2ghk3bha-uc.a.run.app"
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
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make an HTTP request with retry logic for 429/503."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.request(method, path, **kwargs)

                if resp.status_code == 401:
                    raise DaseinAuthError("Invalid API key")
                if resp.status_code == 403:
                    raise DaseinAuthError("Forbidden")
                if resp.status_code == 404:
                    raise DaseinNotFoundError(resp.json().get("detail", "Not found"))

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    if attempt < self.max_retries:
                        time.sleep(retry_after)
                        continue
                    raise DaseinRateLimitError("Rate limit exceeded", retry_after=retry_after)

                if resp.status_code == 503:
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    if attempt < self.max_retries:
                        time.sleep(min(retry_after, 2 ** attempt))
                        continue
                    raise DaseinUnavailableError("Service unavailable", retry_after=retry_after)

                resp.raise_for_status()
                return resp

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise DaseinUnavailableError(f"Connection failed: {e}")

        raise DaseinError(f"Request failed after {self.max_retries} retries: {last_error}")

    def create_index(
        self,
        name: str,
        model: str | None = None,
        plan: str = "dense",
    ) -> Index:
        """
        Create a new index.

        Args:
            name: Human-readable index name
            model: Embedding model ID (e.g., "bge-large-en-v1.5"). None for BYOV.
            plan: "dense" ($10/mo) or "hybrid" ($15/mo)

        Returns:
            Index object for upserting and querying
        """
        resp = self._request("POST", "/indexes", json={
            "name": name,
            "model_id": model,
            "plan": plan,
        })
        data = resp.json()
        return Index(
            client=self,
            index_id=data["index_id"],
            model_id=model,
            plan=plan,
            dim=data.get("dim", 1024),
        )

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
        return IndexInfo(**data)

    def delete_index(self, index_id: str) -> None:
        """Delete an index permanently."""
        self._request("DELETE", f"/indexes/{index_id}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
