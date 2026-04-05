# Dasein

Python SDK for the [Dasein](https://daseinai.ai/index) managed vector index service.

Low-latency vector search with hybrid retrieval as a one-line toggle. Send raw text and get back ranked results — Dasein handles embedding, indexing, and serving.

See our <a href="http://results.daseinai.ai" target="_blank">VectorDBBench results</a> for latency and recall benchmarks.

## Install

```bash
pip install dasein-ai
```

## Quick Start

```python
from dasein import Client

client = Client(api_key="dsk_...")

# Create an index — we embed your text automatically
index = client.create_index("my-docs", model="bge-large-en-v1.5")

# Upsert documents
index.upsert([
    {"id": "doc1", "text": "Machine learning is a subset of AI", "metadata": {"topic": "ai"}},
    {"id": "doc2", "text": "Python is great for data science", "metadata": {"topic": "code"}},
    {"id": "doc3", "text": "The stock market rallied today", "metadata": {"topic": "finance"}},
])

# Dense search — returns id, score, metadata (all from RAM, no SSD)
results = index.query("what is machine learning?", top_k=5)

# Flip to hybrid — combines dense vectors with BM25 in a single call
results = index.query("what is machine learning?", top_k=5, mode="hybrid")

# Need the original text back? Opt in (adds SSD read per result)
results = index.query("what is machine learning?", top_k=5, include_text=True)

for r in results:
    print(f"{r.id}: {r.score:.4f} — {r.metadata}")
```

## Hybrid Search

Toggle between dense-only and hybrid retrieval per query — no config changes, no reindexing, no separate BM25 pipeline.

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

## Get an API Key

**Web:** Sign up with GitHub at [daseinai.ai/auth](https://dasein-api-939340394421.us-central1.run.app/auth/github) — no credit card required. You'll get an API key instantly.

**CLI / Agents:**

```python
import httpx, time

resp = httpx.post("https://dasein-api-939340394421.us-central1.run.app/auth/device/start").json()
print(f"Go to {resp['verification_uri']} and enter code: {resp['user_code']}")

while True:
    time.sleep(resp.get("interval", 5))
    poll = httpx.post(
        "https://dasein-api-939340394421.us-central1.run.app/auth/device/poll",
        json={"device_code": resp["device_code"]},
    ).json()
    if poll.get("api_key"):
        print(f"API key: {poll['api_key']}")
        break
```

## Features

**Managed embedding** — Pass raw text, we embed with open-source models (BGE, Nomic, E5, GTE). No embedding infrastructure to manage.

**Bring your own vectors** — Already have embeddings? Pass them directly with any dimension.

**Hybrid search as a toggle** — Switch between dense and hybrid retrieval per query. No reindexing, no separate BM25 infrastructure.

**Metadata filtering** — Attach key-value metadata to documents and filter at query time.

**Automatic retries** — The SDK retries on 429 and 503 with exponential backoff.

## Embedding Models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `bge-large-en-v1.5` | 1024 | Strong general-purpose English model |
| `nomic-embed-text-v1.5` | 768 | Good balance of speed and quality |
| `e5-large-v2` | 1024 | Microsoft's E5 family |
| `gte-large-en-v1.5` | 1024 | Alibaba's GTE family |

Or skip the model parameter and pass your own vectors of any dimension.

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
    model="bge-large-en-v1.5",  # None for bring-your-own-vectors
    plan="dense",                # "dense" or "hybrid"
)
```

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

### Upsert Documents

```python
index.upsert([
    {"id": "doc1", "text": "Hello world"},
    {"id": "doc2", "vector": [0.1, 0.2, ...], "metadata": {"type": "example"}},
])
```

Max 100 documents per call. The SDK automatically batches larger lists.

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
    mode="dense",                # "dense" or "hybrid"
    filter={"key": "value"},     # optional metadata filter
    exact=False,                 # exact keyword matching (hybrid only)
    phrase=False,                # exact phrase matching (hybrid only)
    fuzzy=False,                 # typo-tolerant matching (hybrid only)
    alpha=0.5,                   # dense vs BM25 balance (0=dense, 1=BM25)
    include_text=False,          # return stored text (off by default)
    include_metadata=True,       # return stored metadata (on by default)
)
```

**What you get back** depends on your settings:

| Setting | Returns | I/O cost |
|---------|---------|----------|
| Default | `id`, `score`, `metadata` | RAM only (dense) or RAM only (hybrid) |
| `include_text=True` | + `text` | Adds SSD read per result |
| `include_metadata=False` | `id`, `score` only | Fastest — pure RAM, zero SSD |

```python
# Maximum QPS — IDs and scores only, pure RAM
results = index.query("quarterly earnings", top_k=10, include_metadata=False)
for r in results:
    print(r.id, r.score)

# Standard — IDs, scores, and metadata (default)
results = index.query("quarterly earnings", top_k=10)
for r in results:
    print(r.id, r.score, r.metadata)

# Full hydration — include original text
results = index.query("quarterly earnings", top_k=10, include_text=True)
for r in results:
    print(r.id, r.score, r.text, r.metadata)
```

Text is stored on SSD and only fetched when you ask for it. This means the default dense and hybrid search paths are entirely RAM-resident — no disk I/O in the query hot path.

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
    DaseinError,             # base exception
    DaseinAuthError,         # 401/403 — bad or missing API key
    DaseinNotFoundError,     # 404 — index doesn't exist
    DaseinRateLimitError,    # 429 — rate limit exceeded (has retry_after)
    DaseinUnavailableError,  # 503 — service temporarily unavailable (has retry_after)
    DaseinBuildError,        # build failed
)
```

## License

MIT
