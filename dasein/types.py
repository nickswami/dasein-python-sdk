"""Typed request and response objects for the Dasein SDK."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UpsertItem:
    """A document to upsert into an index."""
    id: str | int
    vector: list[float] | None = None
    text: str | None = None
    metadata: dict[str, int | float | str] | None = None


@dataclass
class QueryResult:
    """A single search result."""
    id: str | int
    score: float
    text: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class QueryResponse:
    """Query results plus server-side timing from X-*-Us response headers.

    Behaves like a list of QueryResult for backward compatibility —
    you can iterate, index, and len() it directly.
    """
    results: list[QueryResult]
    round_trip_ms: float = 0.0
    server_total_us: int = 0
    search_us: int = 0
    embed_us: int = 0
    auth_us: int = 0
    rate_us: int = 0
    route_us: int = 0
    resp_us: int = 0

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        return self.results[idx]


@dataclass
class IndexInfo:
    """Index metadata. Tolerates extra keys from API for forward compat."""
    index_id: str
    status: str
    index_type: str | None = None
    vector_count: int = 0
    model_id: str | None = None
    has_text: bool = False
    dim: int = 1024
    max_vectors: int | None = None
    index_mode: str | None = None
    ram_bytes: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "IndexInfo":
        mapped = dict(data)
        if "plan" in mapped and "index_type" not in mapped:
            mapped["index_type"] = mapped.pop("plan")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in mapped.items() if k in known})
