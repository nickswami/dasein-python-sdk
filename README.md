# Dasein

Python SDK for the [Dasein](https://daseinai.ai) managed vector index service.

Dasein delivers up to 1M vectors at $10/month (dense) and $15/month (hybrid) with low latency, high recall, and zero configuration. Bring your own vectors or let us embed your text with open-source models.

## Install

```bash
pip install dasein-ai
```

## Quick Start

```python
from dasein import Client

client = Client(api_key="dsk_...")

# Create an index with automatic embedding
index = client.create_index("my-docs", model="bge-large-en-v1.5")

# Upsert documents — we handle embedding
index.upsert([
    {"id": "doc1", "text": "Machine learning is a subset of AI", "metadata": {"topic": "ai"}},
    {"id": "doc2", "text": "Python is great for data science", "metadata": {"topic": "code"}},
    {"id": "doc3", "text": "The stock market rallied today", "metadata": {"topic": "finance"}},
])

# Query with text
results = index.query("what is machine learning?", top_k=5)
for r in results:
    print(f"{r.id}: {r.score:.4f} — {r.text}")

# Filter by metadata
results = index.query("programming", top_k=5, filter={"topic": "code"})
```

## Get an API Key

Sign up with GitHub at [daseinai.ai](https://daseinai.ai) — no credit card required. Your free trial includes 1 index, 100K vectors, and 1M embedding tokens.

## Features

**Managed embedding** — Pass raw text, we embed it with open-source models (BGE, Nomic, E5, GTE). No embedding infrastructure to manage.

**Bring your own vectors** — Already have embeddings? Pass them directly.

**Metadata filtering** — Attach key-value metadata to documents and filter at query time. Up to 1,000 unique filter values per index.

**Hybrid search** — Combine dense vector search with BM25 for better retrieval on keyword-heavy queries. Opt-in per query with `mode="hybrid"`.

**Automatic retries** — The SDK retries on 429 (rate limit) and 503 (temporarily unavailable) with exponential backoff.

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

Only needed if you're using bring-your-own-vectors with a model we don't recognize. Indexes with a known model build automatically after the first upsert.

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

## Pricing

| Plan | Price | Includes |
|------|-------|----------|
| Free trial | $0 for 7 days | 1 index, 100K vectors, 1M embedding tokens/month |
| Dense | $10/month | 3 indexes, 1M vectors, 50M embedding tokens/month |
| Hybrid | $15/month | 3 indexes, 1M vectors, 50M embedding tokens/month, BM25 + dense fusion |

## License

MIT
