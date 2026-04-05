# Dasein

Python SDK for the [Dasein](https://daseinai.ai) managed vector index service.

Low-latency vector search with hybrid retrieval as a one-line toggle. Send raw text and get back ranked results — Dasein handles embedding, indexing, and serving.

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

# Dense search
results = index.query("what is machine learning?", top_k=5)

# Flip to hybrid — combines dense vectors with BM25 in a single call
results = index.query("what is machine learning?", top_k=5, mode="hybrid")

for r in results:
    print(f"{r.id}: {r.score:.4f} — {r.text}")
```

## Hybrid Search

Toggle between dense-only and hybrid retrieval per query — no config changes, no reindexing, no separate BM25 pipeline.

```python
# Dense: pure semantic similarity
results = index.query("financial derivatives risk models", top_k=10, mode="dense")

# Hybrid: semantic + BM25 keyword matching, fused and re-ranked
results = index.query("AAPL earnings Q3 2025", top_k=10, mode="hybrid")

# Hybrid with exact BM25 rescore for maximum precision
results = index.query("AAPL earnings Q3 2025", top_k=10, mode="hybrid", exact_rescore=True)
```

Hybrid mode is strongest on queries with specific keywords, entity names, or codes where pure semantic search loses signal. Dense mode is better for abstract, conceptual queries. You choose per query.

## Get an API Key

Sign up with GitHub at [daseinai.ai](https://daseinai.ai) — no credit card required.

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

### Get Existing Index

```python
index = client.get_index("index_id")
```

### Upsert Documents

```python
index.upsert([
    {"id": "doc1", "text": "Hello world"},
    {"id": "doc2", "vector": [0.1, 0.2, ...], "metadata": {"type": "example"}},
])
```

Max 100 documents per call. The SDK automatically batches larger lists.

### Query

```python
results = index.query(
    text="search query",      # or vector=[0.1, 0.2, ...]
    top_k=10,
    mode="dense",              # "dense" or "hybrid"
    filter={"key": "value"},   # optional metadata filter
    exact_rescore=False,       # exact BM25 rescore (hybrid only)
)

for r in results:
    print(r.id, r.score, r.text, r.metadata)
```

### Delete Documents

```python
index.delete(["doc1", "doc2"])
```

### Build (BYOV only)

```python
index.build()
```

Only needed for bring-your-own-vectors with unrecognized models. Known-model indexes build automatically after the first upsert.

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
