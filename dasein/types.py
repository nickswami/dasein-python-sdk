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
    metadata: dict[str, str] | None = None


@dataclass
class QueryResult:
    """A single search result."""
    id: str | int
    score: float
    text: str | None = None
    metadata: dict[str, Any] | None = None
    dense_score: float | None = None
    sparse_score: float | None = None


@dataclass
class IndexInfo:
    """Index metadata."""
    index_id: str
    status: str
    plan: str | None = None
    vector_count: int = 0
    model_id: str | None = None
    has_text: bool = False
    dim: int = 1024
