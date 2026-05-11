"""
End-to-end test for ``Client.predict_dynamic`` against a live API.

(File name retained from the pre-merge ``predict_dynamic_top_k`` era;
the surface itself has been collapsed into a single ``predict_dynamic``
call that returns ``alpha + top_k_dense + top_k_hybrid``.)

Hits the full path:

    SDK → Cloudflare → API (Cloud Run) → embed pod
        (BGE-large encoder + AlphaKeepMLP triple-head)
        → response

Skipped unless ``DASEIN_AGENTIC_E2E_API_KEY`` is set (we reuse the same
env var as the agentic e2e harness — it's the same backing key).
"""
from __future__ import annotations

import os
import time

import pytest

from dasein import Client, DynamicPrediction


_API_KEY_ENV = "DASEIN_AGENTIC_E2E_API_KEY"
_BASE_URL_ENV = "DASEIN_AGENTIC_E2E_BASE_URL"
_DEFAULT_BASE_URL = "https://api.daseinai.ai"
# predict_dynamic is one warm GPU forward over an L4. p50 should be well
# under 100ms; we budget 1s to absorb a Cloud Run cold start on the
# first call.
_BUDGET_S = 1.0


pytestmark = pytest.mark.skipif(
    not os.environ.get(_API_KEY_ENV),
    reason=f"set {_API_KEY_ENV} to run predict_dynamic E2E",
)


def _client():
    api_key = os.environ[_API_KEY_ENV]
    base_url = os.environ.get(_BASE_URL_ENV, _DEFAULT_BASE_URL)
    return Client(api_key=api_key, base_url=base_url)


def test_predict_dynamic_returns_valid_result():
    """Happy path: text-only call returns a DynamicPrediction with both
    Ks in [1, 10] and alpha in [0, 1] (public dense-weight convention).
    Latency under budget."""
    client = _client()

    # Warmup — absorbs Cloud Run cold start + BGE encoder warmup.
    _ = client.predict_dynamic("warmup probe — anything")

    t0 = time.perf_counter()
    r = client.predict_dynamic("who founded apple computer?")
    elapsed = time.perf_counter() - t0

    assert isinstance(r, DynamicPrediction)
    assert 1 <= r.top_k_dense <= 10, f"top_k_dense out of [1,10]: {r.top_k_dense}"
    assert 1 <= r.top_k_hybrid <= 10, f"top_k_hybrid out of [1,10]: {r.top_k_hybrid}"
    assert 0.0 <= r.alpha <= 1.0, f"alpha out of [0,1]: {r.alpha}"
    assert elapsed < _BUDGET_S, (
        f"warm predict_dynamic {elapsed:.2f}s >= {_BUDGET_S:.2f}s"
    )
    print(f"\n[e2e] predict_dynamic: {elapsed*1000:.0f} ms "
          f"alpha={r.alpha:.3f} K_dense={r.top_k_dense} K_hybrid={r.top_k_hybrid}")


def test_predict_dynamic_respects_query_vector():
    """Caller-supplied query_vector skips the embed step. Same shape of
    response, lower latency (no encoder hop)."""
    pytest.importorskip("numpy")
    import numpy as np

    client = _client()
    qvec = np.zeros(1024, dtype=np.float32)
    qvec[0] = 1.0  # arbitrary unit vector — exercising the API surface.

    r = client.predict_dynamic(
        "test with caller-supplied vector", query_vector=qvec.tolist(),
    )
    assert isinstance(r, DynamicPrediction)
    assert 1 <= r.top_k_dense <= 10
    assert 1 <= r.top_k_hybrid <= 10
    assert 0.0 <= r.alpha <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
