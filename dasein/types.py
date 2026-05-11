"""Typed request and response objects for the Dasein SDK."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Hybrid-fusion alpha convention.
#
# Public / SDK convention (matches Pinecone, Weaviate, the standard hybrid
# search literature):
#
#     score = alpha * dense + (1 - alpha) * sparse
#     alpha = 1.0  →  pure dense
#     alpha = 0.0  →  pure BM25 / sparse
#     alpha = 0.5  →  even blend
#
# The server (engine/csrc/serve.c, the dynamic-hybrid alpha head, and the
# dynamic-hybrid checkpoints under gs://.../dh/) was originally wired with
# the *complement* — alpha = BM25 weight — and the trained DH heads emit
# their scalars in that internal frame. Rather than retrain the heads and
# break every deployed serving pod, the SDK transparently flips on the
# wire: user → server on send, server → user on receive. Server stays
# untouched; users see the standard convention end to end.
#
# The flip is symmetric (1 - x), so one helper is enough.
# ---------------------------------------------------------------------------

def _alpha_user_to_server(alpha: float) -> float:
    """Convert a public-convention alpha (1=dense) to the server's internal
    alpha (1=BM25). Used everywhere the SDK forwards a user-supplied
    alpha to the API."""
    return 1.0 - float(alpha)


def _alpha_server_to_user(alpha: float) -> float:
    """Convert a server-side alpha (1=BM25) into the public convention
    (1=dense). Used when surfacing the alpha scalar from
    /v1/predict_dynamic_top_k (the unified predict_dynamic backend)
    to the caller."""
    return 1.0 - float(alpha)


@dataclass
class UpsertItem:
    """A document to upsert into an index."""
    id: str | int
    vector: list[float] | None = None
    text: str | None = None
    metadata: dict[str, int | float | str] | None = None


@dataclass
class QueryResult:
    """A single search result.

    `vector` is either a numpy.ndarray (float32, preferred — returned when the
    SDK can request and decode the base64 wire format), a list[float] (legacy
    JSON path when numpy isn't installed), or None (when include_vectors=False).
    Callers should tolerate both. np.asarray(v) normalizes either form.
    """
    id: str | int
    score: float
    text: str | None = None
    metadata: dict[str, Any] | None = None
    vector: Any | None = None


@dataclass
class QueryResponse:
    """Query results plus server-side timing from X-*-Us response headers.

    Behaves like a list of QueryResult for backward compatibility —
    you can iterate, index, and len() it directly.

    When the call was an ``agentic_search`` query (multi-hop), ``results``
    holds the **final-hop fused ranking** — the documents the system
    surfaced after working through the chain of sub-questions. This is
    a retrieval system: the deliverable is the ranked list, same shape
    as ``index.query()``.

    Optional, off by default:

    * ``final_answer`` — populated only if the caller passed
      ``include_answer=True``. The reader's parsed answer to the
      original question, surfaced as a convenience for callers who want
      a one-liner. The ranking is still the source of truth.
    * ``chain`` / ``n_hops`` / ``hops`` — populated when the caller
      passed ``return_hops=True``; otherwise None. Exposes the
      intermediate sub-questions and per-hop hits for debugging / UI.
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
    # Populated by Client.query_batch() when a single sub-query failed
    # (auth, index-not-loaded, malformed, backend error). None on success.
    # Per-slot failures do NOT raise — the caller iterates and inspects
    # `error` to decide whether to retry that slot.
    error: str | None = None
    # Multi-hop / agentic_search fields. None on a single-hop query.
    final_answer: str | None = None
    chain: list[str] | None = None
    n_hops: int | None = None
    hops: list[dict[str, Any]] | None = None

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        return self.results[idx]


@dataclass
class DynamicPrediction:
    """Per-query retrieval-plan prediction (BYO-retriever surface).

    Returned by :meth:`Client.predict_dynamic`. One GPU forward, one
    HTTP call, three scalars: the per-query fusion weight Dasein would
    use, plus the per-query top-K cutoffs to apply downstream. Apply
    whichever subset matches your stack — pure-dense callers ignore
    ``alpha`` and use ``top_k_dense``; hybrid callers fuse with
    ``alpha`` and clip to ``top_k_hybrid``.

    * ``alpha`` — per-query fusion weight in ``[0.0, 1.0]``. 1.0 = pure
      dense, 0.0 = pure BM25, 0.5 = even blend (Pinecone / Weaviate
      convention). Use this in your RRF / convex-combination fusion.
    * ``top_k_dense`` — smallest top-K to retrieve when using the
      **dense ranking only**.
    * ``top_k_hybrid`` — smallest top-K to retrieve when using the
      **alpha-fused dense + BM25 ranking** (paired with ``alpha``
      above; the K head was trained against that fused ranking).

    Both K values are integers in ``[1, 10]`` (the keep heads were
    trained against that budget). They're an **upper-bound suggestion**
    — i.e. ``effective_k = min(your_top_k, top_k_*)``.
    """
    alpha: float
    top_k_dense: int
    top_k_hybrid: int


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
