"""
End-to-end latency test for `Index.query(agentic_search=True)`.

Hits a real warm production index via the public API path:

    SDK → Cloudflare → API (Cloud Run) → multihop SSE proxy → embed pod
        → T5 decompose → per-hop encode → C-server :8080 /batch_query
        → reader → final ranking + answer

Skipped unless both env vars are set:

    DASEIN_AGENTIC_E2E_API_KEY   = a production API key (dsk_…)
    DASEIN_AGENTIC_E2E_INDEX_ID  = a warm hybrid index owned by that key

The harness runs **two** calls:

  1. *warmup*  — pays any cold-path costs (resolver cache miss, JIT
                 compile, etc). Result discarded.
  2. *measured* — the latency we actually assert on.

Assertions match the warm band we observed on the live demo:

  * total wall-time < 2.5 s   (hard ceiling — fail on regression)
  * total wall-time < 1.8 s   (soft target — emits a warning)
  * we got a non-empty fused ranking AND a parsed final_answer

Override the budget with DASEIN_AGENTIC_E2E_MAX_S if you need to be
forgiving on a degraded host.
"""
from __future__ import annotations

import os
import time
import warnings

import pytest

from dasein import Client


_API_KEY_ENV = "DASEIN_AGENTIC_E2E_API_KEY"
_INDEX_ENV = "DASEIN_AGENTIC_E2E_INDEX_ID"
_QUESTION_ENV = "DASEIN_AGENTIC_E2E_QUESTION"
_BASE_URL_ENV = "DASEIN_AGENTIC_E2E_BASE_URL"
_MAX_S_ENV = "DASEIN_AGENTIC_E2E_MAX_S"

_DEFAULT_QUESTION = (
    "What 2010 dream-heist movie was directed by the filmmaker who made "
    "the space wormhole movie starring the actor who played the 'Alright, "
    "alright, alright' guy in Dazed and Confused?"
)
_DEFAULT_BASE_URL = "https://api.daseinai.ai"
_DEFAULT_MAX_S = 2.5     # hard fail above this
_SOFT_TARGET_S = 1.8     # warn above this, fail above _DEFAULT_MAX_S


pytestmark = pytest.mark.skipif(
    not (os.environ.get(_API_KEY_ENV) and os.environ.get(_INDEX_ENV)),
    reason=(
        f"set {_API_KEY_ENV} and {_INDEX_ENV} to run agentic-search E2E "
        "latency test"
    ),
)


def _client_and_index():
    api_key = os.environ[_API_KEY_ENV]
    base_url = os.environ.get(_BASE_URL_ENV, _DEFAULT_BASE_URL)
    client = Client(api_key=api_key, base_url=base_url)
    return client, client.get_index(os.environ[_INDEX_ENV])


def test_agentic_search_warm_latency_within_budget():
    """End-to-end latency on a warm production index.

    Production multihop p50 (3-4 hops) is ~1.0-1.6 s. We pay the cold-path
    open_session + resolver cache miss on the first call, then assert on
    the second. Hard ceiling 2.5 s — anything above that is a regression.
    """
    question = os.environ.get(_QUESTION_ENV, _DEFAULT_QUESTION)
    max_s = float(os.environ.get(_MAX_S_ENV, _DEFAULT_MAX_S))
    _, idx = _client_and_index()

    # Warmup: discard timing.
    _ = idx.query(question, top_k=10, agentic_search=True)

    t0 = time.perf_counter()
    resp = idx.query(question, top_k=10, agentic_search=True,
                     return_hops=True)
    elapsed = time.perf_counter() - t0

    assert resp.results, "warm agentic query returned an empty ranking"
    assert resp.final_answer is not None, (
        "warm agentic query returned no final_answer"
    )
    assert resp.n_hops and resp.n_hops >= 1, (
        f"expected >=1 hop, got n_hops={resp.n_hops!r}"
    )

    if elapsed > _SOFT_TARGET_S:
        warnings.warn(
            f"agentic_search warm latency {elapsed:.2f}s exceeded soft "
            f"target {_SOFT_TARGET_S:.2f}s (hard ceiling {max_s:.2f}s)",
            stacklevel=2,
        )
    assert elapsed < max_s, (
        f"agentic_search warm latency regression: {elapsed:.2f}s "
        f">= {max_s:.2f}s ceiling. response={{"
        f"n_hops={resp.n_hops}, "
        f"final_answer={resp.final_answer!r}}}"
    )


def test_agentic_search_respects_dynamic_hybrid_per_hop():
    """Smoke: a multi-hop run with dynamic_hybrid=True should still finish
    inside the budget. This exercises the full server-side fan-out
    (per-hop alpha-predictor → C-server fast path)."""
    question = os.environ.get(_QUESTION_ENV, _DEFAULT_QUESTION)
    max_s = float(os.environ.get(_MAX_S_ENV, _DEFAULT_MAX_S))
    _, idx = _client_and_index()

    # Warmup.
    _ = idx.query(question, top_k=10, agentic_search=True,
                  mode="hybrid", dynamic_hybrid=True)

    t0 = time.perf_counter()
    resp = idx.query(question, top_k=10, agentic_search=True,
                     mode="hybrid", dynamic_hybrid=True)
    elapsed = time.perf_counter() - t0

    assert resp.results, "warm agentic+dh query returned empty ranking"
    assert elapsed < max_s, (
        f"agentic_search+dynamic_hybrid warm latency {elapsed:.2f}s "
        f">= {max_s:.2f}s ceiling"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
