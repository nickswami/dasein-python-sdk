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

  * total wall-time < 1.5 s   (hard ceiling — fail on regression)
  * total wall-time < 1.2 s   (soft target — emits a warning)
  * we got a non-empty fused ranking back

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
# Demo runs 3-4 hops in ~1 s warm. We give a small buffer and fail
# anything beyond — agentic_search is supposed to be ~demo-speed.
_DEFAULT_MAX_S = 1.5     # hard fail above this
_SOFT_TARGET_S = 1.2     # warn above this, fail above _DEFAULT_MAX_S


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


def _check_latency(elapsed: float, *, label: str, max_s: float):
    """Common latency assertion — print elapsed for visibility in -s mode,
    warn above soft target, hard-fail above ceiling."""
    print(f"\n[e2e] {label}: warm latency = {elapsed*1000:.0f} ms "
          f"(soft target {_SOFT_TARGET_S*1000:.0f} ms, "
          f"hard ceiling {max_s*1000:.0f} ms)")
    if elapsed > _SOFT_TARGET_S:
        warnings.warn(
            f"agentic_search [{label}] warm latency {elapsed:.2f}s "
            f"exceeded soft target {_SOFT_TARGET_S:.2f}s "
            f"(hard ceiling {max_s:.2f}s)",
            stacklevel=2,
        )
    assert elapsed < max_s, (
        f"agentic_search [{label}] warm latency regression: "
        f"{elapsed:.2f}s >= {max_s:.2f}s ceiling"
    )


def test_agentic_search_warm_latency_within_budget():
    """End-to-end latency on a warm production index, static hybrid α=0.5.

    Demo p50 (3-4 hops) is ~1 s. We pay the cold-path open_session +
    resolver cache miss on the warmup call, then assert on the measured
    call.
    """
    question = os.environ.get(_QUESTION_ENV, _DEFAULT_QUESTION)
    max_s = float(os.environ.get(_MAX_S_ENV, _DEFAULT_MAX_S))
    _, idx = _client_and_index()

    _ = idx.query(question, top_k=10, agentic_search=True)

    t0 = time.perf_counter()
    resp = idx.query(question, top_k=10, agentic_search=True,
                     return_hops=True)
    elapsed = time.perf_counter() - t0

    assert resp.results, "warm agentic query returned an empty ranking"
    assert resp.n_hops and resp.n_hops >= 1, (
        f"expected >=1 hop, got n_hops={resp.n_hops!r}"
    )
    _check_latency(elapsed, label="static-hybrid", max_s=max_s)


def test_agentic_search_respects_dynamic_hybrid_per_hop():
    """Smoke: a multi-hop run with dynamic_hybrid=True should still finish
    inside the budget. This exercises the full server-side fan-out
    (per-hop alpha-predictor → C-server fast path)."""
    question = os.environ.get(_QUESTION_ENV, _DEFAULT_QUESTION)
    max_s = float(os.environ.get(_MAX_S_ENV, _DEFAULT_MAX_S))
    _, idx = _client_and_index()

    _ = idx.query(question, top_k=10, agentic_search=True,
                  mode="hybrid", dynamic_hybrid=True)

    t0 = time.perf_counter()
    resp = idx.query(question, top_k=10, agentic_search=True,
                     mode="hybrid", dynamic_hybrid=True)
    elapsed = time.perf_counter() - t0

    assert resp.results, "warm agentic+dh query returned empty ranking"
    _check_latency(elapsed, label="dynamic-hybrid", max_s=max_s)


def test_agentic_search_with_dynamic_top_k_under_budget():
    """Multi-hop with dynamic_hybrid + dynamic_top_k. Verifies the
    Dynamic Top-K cutoff doesn't blow the latency budget — the K head
    runs in the same forward as the alpha head, so the only added cost
    is the per-hop slice. Result set may be SHORTER than top_k (that's
    the point); we just assert non-empty and within budget."""
    question = os.environ.get(_QUESTION_ENV, _DEFAULT_QUESTION)
    max_s = float(os.environ.get(_MAX_S_ENV, _DEFAULT_MAX_S))
    _, idx = _client_and_index()

    _ = idx.query(question, top_k=10, agentic_search=True,
                  mode="hybrid", dynamic_hybrid=True, dynamic_top_k=True)

    t0 = time.perf_counter()
    resp = idx.query(question, top_k=10, agentic_search=True,
                     mode="hybrid", dynamic_hybrid=True, dynamic_top_k=True)
    elapsed = time.perf_counter() - t0

    assert resp.results, "warm agentic+dh+dyntopk query returned empty ranking"
    # K_pred ∈ [1, 10] by training; can't exceed the caller's top_k=10.
    assert len(resp.results) <= 10, (
        f"dynamic_top_k must not exceed caller's top_k=10, got "
        f"{len(resp.results)}"
    )
    _check_latency(elapsed, label="dynamic-top-k", max_s=max_s)


def test_agentic_search_dynamic_top_k_requires_dynamic_hybrid_client_side():
    """SDK enforces the pairing before any wire call — fast-fail."""
    _, idx = _client_and_index()
    with pytest.raises(ValueError, match="dynamic_hybrid"):
        idx.query("anything", agentic_search=True, dynamic_top_k=True)


def test_agentic_search_include_answer_opt_in():
    """`include_answer=True` populates response.final_answer; default off."""
    question = os.environ.get(_QUESTION_ENV, _DEFAULT_QUESTION)
    _, idx = _client_and_index()

    off = idx.query(question, top_k=5, agentic_search=True)
    assert off.final_answer is None, (
        f"final_answer should be None by default, got {off.final_answer!r}"
    )

    on = idx.query(question, top_k=5, agentic_search=True,
                   include_answer=True)
    # When the reader extracted something, final_answer is a non-None
    # string (possibly empty if the reader skipped the final hop, which
    # is intentional — it's a search ranking, not a QA system). The
    # contract is: opt-in => field present, default => None.
    assert on.final_answer is not None, (
        "final_answer should be a string when include_answer=True"
    )
    assert isinstance(on.final_answer, str)
    print(f"\n[e2e] include_answer=True -> final_answer={on.final_answer!r}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
