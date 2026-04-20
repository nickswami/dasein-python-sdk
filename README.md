# Dasein

Python SDK for the [Dasein](https://www.daseinai.ai/) managed vector index service.

Low-latency vector search with hybrid retrieval. Send raw text and get back ranked results — Dasein handles embedding, indexing, and serving.

See our <a href="https://results.daseinai.ai/results" target="_blank">VectorDBBench results</a> for latency and recall benchmarks.

## Install

```bash
pip install dasein-ai  # not "dasein" — the package name is dasein-ai
```

## Quick Start

```python
from dasein import Client

client = Client(api_key="dsk_...")  # get a free key at https://api.daseinai.ai/auth/github

# Create a hybrid index (semantic + keyword search)
index = client.create_index("my-docs", index_type="hybrid", model="bge-large-en-v1.5")

# Upsert documents — metadata values can be strings, ints, or floats
index.upsert([
    {"id": "doc1", "text": "SpaceX launched Starship on its 5th test flight",
     "metadata": {"source": "reuters", "category": "space", "year": 2025, "priority": 1}},
    {"id": "doc2", "text": "GPT-5 achieves superhuman reasoning on ARC-AGI",
     "metadata": {"source": "arxiv", "category": "ai", "year": 2025, "priority": 2}},
    {"id": "doc3", "text": "Fed holds rates steady amid cooling inflation",
     "metadata": {"source": "bloomberg", "category": "finance", "year": 2025, "priority": 3}},
    {"id": "doc4", "text": "Python 3.13 ships with a JIT compiler",
     "metadata": {"source": "pep", "category": "code", "year": 2024, "priority": 1}},
])

# Hybrid search — semantic similarity + BM25 keyword matching
results = index.query("what is machine learning?", top_k=5, mode="hybrid")

# Filter by metadata — all operators are true pre-filters (no recall penalty)
results = index.query("recent breakthroughs", top_k=5, filter={"year": {"$gte": 2025}}, include_metadata=True)
results = index.query("top stories", top_k=5, filter={"source": {"$in": ["reuters", "bloomberg"]}, "priority": 1}, include_metadata=True)
results = index.query("tech news", top_k=5, filter={"$or": [{"category": "ai"}, {"category": "code"}]}, include_metadata=True)

for r in results:
    print(f"{r.id}: {r.score:.4f} — {r.metadata}")
```

## Choosing an Index Type

You choose the index type at creation time. This determines what search modes are available.

| `index_type` | What it builds | Query modes available |
|---|---|---|
| `"hybrid"` | Dense vectors + BM25 inverted index | `mode="hybrid"` and `mode="dense"` |
| `"dense"` | Dense vectors only | `mode="dense"` only |

**Use `"hybrid"` unless you have a reason not to.** Hybrid indexes support both dense and hybrid queries — you choose per query. Dense indexes are smaller in RAM but cannot use keyword search.

```python
# Hybrid index — supports both query modes
index = client.create_index("my-docs", index_type="hybrid", model="bge-large-en-v1.5")

# Dense-only index — only supports mode="dense"
index = client.create_index("my-docs", index_type="dense", model="bge-large-en-v1.5")
```

## Hybrid Search

Hybrid indexes support per-query toggling between dense-only and hybrid retrieval — no reindexing, no separate BM25 pipeline.

```python
# Dense: pure semantic similarity
results = index.query("financial derivatives risk models", top_k=10, mode="dense")

# Hybrid: semantic + BM25 keyword matching, fused and re-ranked
results = index.query("AAPL earnings Q3 2025", top_k=10, mode="hybrid")

# Exact keyword matching — only docs that contain all your terms
results = index.query("AAPL earnings Q3 2025", top_k=10, mode="hybrid", exact=True)

# Phrase matching — only docs containing "machine learning" as an exact phrase
results = index.query("machine learning", top_k=10, mode="hybrid", phrase=True)

# Fuzzy matching — handles typos (edit distance 1)
results = index.query("machin lerning", top_k=10, mode="hybrid", fuzzy=True)

# Tune the dense vs BM25 balance (0.0 = all dense, 1.0 = all BM25, default 0.5)
results = index.query("AAPL earnings", top_k=10, mode="hybrid", alpha=0.7)  # lean keyword-heavy
```

Hybrid mode is strongest on queries with specific keywords, entity names, or codes where pure semantic search loses signal. Dense mode is better for abstract, conceptual queries. You choose per query. The keyword features (`exact`, `phrase`, `fuzzy`) refine hybrid results — use them when you need precise keyword control. The `alpha` parameter lets you tune the balance between dense and BM25 ranking in the fusion step.

### Dynamic hybrid — let Dasein pick the balance

Tuning `alpha` per query is tedious and fragile. On hybrid indexes you can
hand the decision off to Dasein:

```python
results = index.query("AAPL earnings Q3 2025",
                      top_k=10, mode="hybrid", dynamic_hybrid=True)
```

With `dynamic_hybrid=True`, Dasein picks the dense/BM25 balance for each
query individually and returns the final ranking directly — `alpha` is
ignored. Only available on hybrid indexes. `top_k` must be `<= 100`.

No extra setup. No retraining on your data. Works across encoders.

### Managed fusion weight for your own stack — `client.predict_alpha`

If you run your own dense + BM25 pipeline and just want a better per-query
fusion weight, call Dasein directly — no index required:

```python
alpha = client.predict_alpha("who founded apple?")
# blend your own rankings at this alpha
fused = rrf_fuse(my_dense_hits, my_bm25_hits, alpha=alpha)
```

Returns a `float` in `[0.0, 1.0]` (0 = all dense, 1 = all BM25). Free
plans get 1,000 calls per month; paid hybrid plans are unlimited. Pass an
already-embedded `query_vector` if you have one to skip the embed cost.

## Metadata

Attach key-value metadata to documents for filtering at query time. Values can be strings, integers, or floats.

```python
index.upsert([
    {
        "id": "doc1",
        "text": "SpaceX launched Starship",
        "metadata": {
            "source": "reuters",
            "category": "space",
            "year": 2025,
            "priority": 1,
            "rating": 9.2,
        },
    },
])

# Simple equality
results = index.query("rocket launch", top_k=10, filter={"source": "reuters"})
results = index.query("rocket launch", top_k=10, filter={"category": "space", "year": 2025})
```

### Filtering

Filters are true pre-filters — candidates that don't match are never touched. No recall penalty.

```python
# Equality (default — bare values are $eq)
filter={"genre": "sci-fi"}
filter={"genre": {"$eq": "sci-fi"}}  # equivalent explicit form

# Not equal
filter={"status": {"$ne": "archived"}}

# In set
filter={"category": {"$in": ["ai", "finance", "health"]}}

# Not in set
filter={"source": {"$nin": ["spam", "test"]}}

# Exists / not exists
filter={"author": {"$exists": True}}

# Numeric range
filter={"year": {"$gte": 2020, "$lte": 2025}}
filter={"rating": {"$gt": 7.5}}

# OR across keys
filter={"$or": [{"category": "ai"}, {"priority": 1}]}

# Combine (AND by default)
filter={"source": "reuters", "year": {"$gte": 2024}, "category": {"$in": ["tech", "science"]}}
```

All filter operators work with both dense and hybrid queries. Pass `include_metadata=True` to return metadata with results.

## Get an API Key

**Web:** Sign up with GitHub at [api.daseinai.ai/auth/github](https://api.daseinai.ai/auth/github) — no credit card required. You'll get an API key instantly.

**CLI / Agents:**

```python
import httpx, time

resp = httpx.post("https://api.daseinai.ai/auth/device/start").json()
print(f"Go to {resp['verification_uri']} and enter code: {resp['user_code']}")

while True:
    time.sleep(resp.get("interval", 5))
    poll = httpx.post(
        "https://api.daseinai.ai/auth/device/poll",
        json={"device_code": resp["device_code"]},
    ).json()
    if poll.get("api_key"):
        print(f"API key: {poll['api_key']}")
        break
```

## Features

**Managed embedding** — Pass raw text, we embed with open-source models (BGE, Nomic, E5, GTE). No embedding infrastructure to manage.

**Bring your own vectors** — Already have embeddings? Pass them directly with any dimension.

**Hybrid search** — Switch between dense and hybrid retrieval per query. No reindexing, no separate BM25 infrastructure.

**Dynamic hybrid** — Let Dasein pick the dense/BM25 balance per query on hybrid indexes (`dynamic_hybrid=True`), or call `client.predict_alpha(text)` to get the same weight for your own hybrid stack. Works across encoders, no retraining required.

**Metadata filtering** — Attach metadata to documents and filter at query time with operators like `$in`, `$ne`, `$gte`, `$lte`, and `$or`. True pre-filters with no recall penalty.

**Automatic retries** — The SDK retries with exponential backoff:

| Error | Read / query | Upsert | Build / delete |
|-------|-------------|--------|----------------|
| 429 (rate limit) | Retried (up to `max_retries`) | Retried | Retried |
| 503 (transient) | Retried | Retried (upserts are idempotent by doc ID) | Not retried |
| 504 (gateway timeout) | Retried | Retried | Not retried |
| Connection error | Retried | Retried | Not retried |

## Embedding Models

| Model | Dimensions | Matryoshka dims | Notes |
|-------|-----------|----------------|-------|
| `bge-large-en-v1.5` | 1024 | 512, 256, 128, 64 | Strong general-purpose English model |
| `nomic-embed-text-v1.5` | 768 | 512, 384, 256, 128, 64 | Good balance of speed and quality |
| `e5-large-v2` | 1024 | — | Microsoft's E5 family (no MRL support) |
| `gte-large-en-v1.5` | 1024 | 512, 256, 128, 64 | Alibaba's GTE family |

Or skip the model parameter and pass your own vectors of any dimension.

### Matryoshka Dimension Truncation

Models trained with Matryoshka Representation Learning (MRL) can be truncated to lower dimensions with minimal recall loss, cutting RAM and storage proportionally. Pass `dim` at index creation:

```python
index = client.create_index("my-docs", index_type="hybrid", model="bge-large-en-v1.5", dim=256)
```

Embeddings are generated at full dimension and truncated + L2-renormalized before indexing. Queries are truncated the same way automatically. The first build for a truncated dimension uses an initial one-time build (slightly slower) since pretrained models are only available for native dimensions.

## API Reference

### Client

```python
from dasein import Client

client = Client(
    api_key="dsk_...",       # required
    base_url=None,           # override API URL (default: Dasein Cloud)
    timeout=30.0,            # request timeout in seconds
    max_retries=3,           # retries on 429/503
)
```

### Create Index

```python
index = client.create_index(
    name="my-index",
    index_type="hybrid",             # REQUIRED CHOICE: "dense" or "hybrid"
    model="bge-large-en-v1.5",      # None for bring-your-own-vectors
    dim=None,                        # truncate to lower dim for MRL models (e.g., 256)
)
```

`index_type` determines what search capabilities the index has:
- `"hybrid"` — builds both a dense vector index and a BM25 inverted index. Supports `mode="dense"` and `mode="hybrid"` queries.
- `"dense"` — builds a dense vector index only. Supports `mode="dense"` queries only.

### List Indexes

```python
indexes = client.list_indexes()
for idx in indexes:
    print(idx["index_id"], idx["name"], idx["status"], idx["vector_count"])
```

### Get Existing Index

```python
index = client.get_index("index_id")
```

### Delete Index

```python
client.delete_index("index_id")
```

### Predict Alpha (managed fusion weight)

```python
alpha = client.predict_alpha(
    text="who founded apple?",
    query_vector=None,    # optional — pre-embedded dense vector
    model_id=None,        # optional — embedding model if query_vector is None
)
```

Returns a `float` in `[0.0, 1.0]` to use as the dense/BM25 blend for your
own hybrid stack. No Dasein index required. See
[Managed fusion weight](#managed-fusion-weight-for-your-own-stack--clientpredict_alpha)
for usage and quotas.

### Cross-Index Query Batch

```python
responses = client.query_batch([
    {"index_id": "abc", "text":   "hello",       "top_k": 10},
    {"index_id": "def", "vector": my_vec,        "top_k": 5, "include_vectors": True},
    {"index_id": "ghi", "text":   "rate limit",  "top_k": 20, "mode": "hybrid"},
])
```

See [Query Batch](#query-batch) below for the full feature surface, per-slot error semantics, and limits.

### Upsert Documents

```python
index.upsert([
    {"id": "doc1", "text": "Hello world", "metadata": {"type": "greeting"}},
    {"id": "doc2", "text": "Goodbye world", "metadata": {"type": "farewell"}},
])
```

Each document can have:
- `id` (required) — unique document ID (string or int)
- `text` — raw text (embedded automatically if the index has a model)
- `vector` — pre-computed embedding (list of floats)
- `metadata` — `dict[str, str | int | float]` for filtering

Max 5,000 documents per call for model-backed indexes (10,000 for bring-your-own-vectors). The SDK automatically batches larger lists.

You can also use the typed `UpsertItem` class instead of raw dicts:

```python
from dasein import UpsertItem

index.upsert([
    UpsertItem(id="doc1", text="Hello world", metadata={"type": "greeting"}),
    UpsertItem(id="doc2", vector=[0.1, 0.2, ...]),
])
```

### Query

```python
results = index.query(
    text="search query",         # or vector=[0.1, 0.2, ...]
    top_k=10,
    mode="hybrid",               # "dense" or "hybrid" (hybrid requires index_type="hybrid")
    filter={"key": "value"},     # optional metadata filter (supports operators — see Filtering)
    exact=False,                 # exact keyword matching (hybrid only)
    phrase=False,                # exact phrase matching (hybrid only)
    fuzzy=False,                 # typo-tolerant matching (hybrid only)
    alpha=0.5,                   # dense vs BM25 balance (0=dense, 1=BM25)
    dynamic_hybrid=False,        # let Dasein pick alpha per query (hybrid only, top_k<=100)
    include_text=False,          # return stored text (off by default)
    include_metadata=False,      # return stored metadata (off by default)
    include_vectors=False,       # return approximate vectors (off by default)
)
```

**What you get back** depends on your settings:

| Setting | Returns | I/O cost |
|---------|---------|----------|
| Default | `id`, `score` | Zero — pure RAM, no SSD reads |
| `include_metadata=True` | + `metadata` | Small SSD read per result (page-cached for hot indexes) |
| `include_text=True` | + `text` | Larger SSD read per result |
| `include_vectors=True` | + `vector` | Zero — reconstructed from RAM (approximate) |

```python
# Default — IDs and scores only, pure RAM, maximum QPS
results = index.query("quarterly earnings", top_k=10)
for r in results:
    print(r.id, r.score)

# Include metadata
results = index.query("quarterly earnings", top_k=10, include_metadata=True)
for r in results:
    print(r.id, r.score, r.metadata)

# Full hydration — metadata + original text
results = index.query("quarterly earnings", top_k=10, include_text=True, include_metadata=True)
for r in results:
    print(r.id, r.score, r.text, r.metadata)

# Include approximate vectors (reconstructed from RAM, no disk I/O)
results = index.query("quarterly earnings", top_k=10, include_vectors=True)
for r in results:
    # r.vector is a numpy.ndarray (float32) when numpy is installed,
    # or a list[float] otherwise. np.asarray(r.vector) works for both.
    print(r.id, r.score, len(r.vector))
```

Text and metadata are stored on SSD and only fetched when you opt in. Vectors are reconstructed from compact codes already in RAM — no disk I/O. The default query path is entirely RAM-resident.

#### Wire format for `include_vectors`

When numpy is available, the SDK automatically asks the server for vectors as base64-encoded little-endian `float32` bytes, then decodes them with `np.frombuffer` outside the GIL. This avoids allocating thousands of Python `float` objects per response and is the path that unlocks high throughput under concurrent use. If numpy isn't installed, the SDK falls back to the legacy JSON-array-of-floats wire format transparently.

### Query Batch

For workloads that run many queries back-to-back — training loops, evaluation suites, mining — use batch queries to amortize HTTP / TLS / router overhead across a single round-trip. Two flavors:

- `index.query_batch(queries)` — many queries, **one index**.
- `client.query_batch(queries)` — many queries, **many indexes** in one request.

Both return `list[QueryResponse]` in request order, both accept the full set of `query()` kwargs per entry, and both cap out at 4096 sub-queries per call.

#### Single-index: `index.query_batch`

```python
# Each entry takes the same keys as Index.query(...)
batch = [
    {"vector": v, "top_k": 10, "include_vectors": True}
    for v in my_query_vectors
]  # up to 4096
responses = index.query_batch(batch)

for q_idx, resp in enumerate(responses):
    for r in resp:
        print(q_idx, r.id, r.score)
```

`index.query_batch` is **functionally identical to calling `query()` N times** — same server-side search path, same hybrid RRF fusion, same filter operators, same flags. The only difference is that many queries travel on one TCP connection in one JSON payload. You can mix dense and hybrid queries, different `top_k`, different `filter`, different `include_*` choices in the same batch.

```python
# Every key that query() takes works inside query_batch() entries:
batch = [
    {"text": "rocket launch",    "top_k": 5,  "mode": "hybrid"},
    {"text": "quarterly earnings","top_k": 10, "filter": {"year": {"$gte": 2024}},
     "include_metadata": True},
    {"vector": my_vec,           "top_k": 20, "include_vectors": True},
]
responses = index.query_batch(batch)
```

#### Multi-index: `client.query_batch`

`client.query_batch` takes a list where **each entry carries its own `index_id`** and fans out across every index it touches inside the router. Same feature surface as `Index.query()` per entry — text / vector, dense / hybrid, filters, `exact` / `phrase` / `fuzzy` / `alpha`, `include_text` / `include_metadata` / `include_vectors`.

```python
# 256 queries scattered across many indexes in one round-trip.
batch = []
for idx_id, qvec in zip(index_ids, query_vectors):
    batch.append({
        "index_id":        idx_id,
        "vector":          qvec,
        "top_k":           10,
        "mode":            "hybrid",
        "include_vectors": True,
    })
responses = client.query_batch(batch)

for sent, resp in zip(batch, responses):
    if resp.error:              # per-slot failure — doesn't fail the batch
        print(sent["index_id"], "FAILED:", resp.error)
        continue
    for r in resp:
        print(sent["index_id"], r.id, r.score, r.vector[:4])
```

Text auto-embeds just like `query()` — the router looks up each index's model, coalesces all sub-queries that share a model into one embed call, and splices the resulting vectors back into their slots. A batch of 256 texts across 60 indexes that all use `bge-large-en-v1.5` costs **one** embed round-trip, not 256.

```python
batch = [
    {"index_id": "abc-001", "text": "climate policy", "top_k": 10, "mode": "hybrid"},
    {"index_id": "def-002", "text": "interest rates", "top_k": 5,  "mode": "dense"},
    {"index_id": "ghi-003", "vector": pre_embedded,   "top_k": 20, "include_vectors": True},
]
responses = client.query_batch(batch)
```

Under the hood the router:

1. Authenticates the API key once, then checks per-slot authorization against each `index_id`.
2. Groups text sub-queries by the index's `model_id` and fires one parallel `/embed` call per distinct model.
3. Groups every (index_id, host) pair into a bucket and fires one pod-side `/batch_query` per bucket — in parallel across up to 24 pods.
4. Reassembles the response in original slot order.

#### Per-slot errors (multi-index only)

Multi-index batches never throw for one bad slot — the whole batch always comes back 200 if the envelope made it. Bad slots come back as an empty `results` list with `resp.error` set:

| `resp.error`                                   | Meaning                                 |
| ---------------------------------------------- | --------------------------------------- |
| `"missing index_id"` / `"missing text or vector"` | Malformed entry                      |
| `"invalid api key"` / `"api key not authorized for this index"` | Auth failure            |
| `"index not loaded"`                           | Index is placing/migrating or unknown   |
| `"embed failed"` / `"embed service not configured"` | Embed path failure                 |
| `"backend error (status N)"`                   | Pod returned non-2xx                    |

Single-index `index.query_batch` uses the same per-slot model — malformed sub-queries come back as empty result sets without `error` set.

Response ordering always matches request ordering: `responses[i]` corresponds to `batch[i]`.

#### Limits

- Max 4096 queries per call (either flavor).
- Request body capped at 16 MiB on the router's inbound side. With 1024-dim JSON-encoded query vectors, that's roughly 1500 queries per batch before you need to split; bring-your-own-vector batches at smaller `top_k` and fewer `include_*` can stretch further.
- Multi-index: one batch can span up to 1024 distinct (index_id, host) buckets and up to 32 distinct embedding models.

### Optional: faster JSON parsing

If `orjson` is importable the SDK will use it for query / query_batch response parsing automatically. It's strictly optional — no changes to your code — but installing it noticeably reduces CPU on the query hot path, especially for large batch responses:

```bash
pip install orjson
```

### Delete Documents

```python
index.delete(["doc1", "doc2"])
```

### Upsert and Wait

```python
result = index.upsert_and_wait(documents, timeout=120.0)
```

Upserts documents and polls until the index becomes queryable. Useful for scripts where you want to upsert and immediately query.

### Build (BYOV only)

```python
index.build()
```

Only needed for bring-your-own-vectors with unrecognized models. Known-model indexes build automatically after the first upsert.

### Compact

```python
index.compact()
```

Triggers a compaction rebuild that removes deleted document tombstones from the graph. Run this after large batch deletions to reclaim performance.

### Index Status

```python
info = index.status()
print(info.status)        # created, building, built, active, etc.
print(info.vector_count)
```

## Exceptions

```python
from dasein.exceptions import (
    DaseinError,             # base — catch-all for any Dasein error, including plain 403 Forbidden
    DaseinAuthError,         # 401, or 403 mentioning credentials / API key / revoked
    DaseinQuotaError,        # 403 — billing/plan/trial/subscription/embed limit
    DaseinNotFoundError,     # 404 — index doesn't exist
    DaseinRateLimitError,    # 429 — transient rate limit exceeded (has retry_after)
    DaseinUnavailableError,  # 503/504 — service temporarily unavailable (has retry_after)
    DaseinBuildError,        # build failed
)
```

`DaseinAuthError` is raised only for credential issues (bad API key, revoked key, authentication failure). `DaseinQuotaError` covers trial limits, plan vector caps, expired/past-due subscriptions, and embed token quotas (including 429s that indicate a non-transient monthly embed cap). `DaseinRateLimitError` is raised for transient per-second rate limits that the SDK retries automatically. A generic 403 (e.g., accessing a resource you don't own) raises `DaseinError` — catch it separately if you need to distinguish resource authorization from credential errors.

## License

MIT
