"""
Dasein quickstart example.

Shows index creation, upsert with typed metadata, filtered queries, and hybrid search.
"""
from dasein import Client

client = Client(api_key="dsk_your_api_key_here")

# Create a hybrid index — supports both dense (semantic) and hybrid (semantic + keyword) queries
index = client.create_index("my-rag-docs", index_type="hybrid", model="bge-large-en-v1.5")

# Upsert documents with rich metadata (str, int, float values)
index.upsert([
    {"id": "doc1", "text": "SpaceX launched Starship on its 5th test flight",
     "metadata": {"source": "reuters", "category": "space", "year": 2025, "priority": 1, "rating": 9.2}},
    {"id": "doc2", "text": "GPT-5 achieves superhuman reasoning on ARC-AGI",
     "metadata": {"source": "arxiv", "category": "ai", "year": 2025, "priority": 2, "rating": 8.7}},
    {"id": "doc3", "text": "Fed holds rates steady amid cooling inflation",
     "metadata": {"source": "bloomberg", "category": "finance", "year": 2025, "priority": 3, "rating": 7.1}},
    {"id": "doc4", "text": "PostgreSQL 17 ships with incremental backup support",
     "metadata": {"source": "pgsql-announce", "category": "code", "year": 2024, "priority": 1, "rating": 8.0}},
])

# --- Hybrid search: semantic + BM25 keyword matching ---
print("=== Hybrid search ===")
results = index.query("what is machine learning?", top_k=5, mode="hybrid", include_text=True)
for r in results:
    print(f"  {r.id}: {r.score:.4f} - {r.text}")

# --- Metadata filtering (all operators) ---
print("\n=== Filter: numeric range ===")
results = index.query("recent news", top_k=5, filter={"year": {"$gte": 2025}, "rating": {"$gt": 8.0}}, include_metadata=True)
for r in results:
    print(f"  {r.id}: {r.score:.4f} [{r.metadata}]")

print("\n=== Filter: $in + equality ===")
results = index.query("top stories", top_k=5, filter={"source": {"$in": ["reuters", "bloomberg"]}, "priority": 1}, include_metadata=True)
for r in results:
    print(f"  {r.id}: {r.score:.4f} [{r.metadata}]")

print("\n=== Filter: $or ===")
results = index.query("tech news", top_k=5, filter={"$or": [{"category": "ai"}, {"category": "code"}]}, include_metadata=True)
for r in results:
    print(f"  {r.id}: {r.score:.4f} [{r.metadata}]")

print("\n=== Filter: $ne + $exists ===")
results = index.query("everything except finance", top_k=5, filter={"category": {"$ne": "finance"}}, include_metadata=True)
for r in results:
    print(f"  {r.id}: {r.score:.4f} [{r.metadata}]")

# --- Bring your own vectors (no hydration — fastest path) ---
print("\n=== BYOV query ===")
results = index.query(vector=[0.12, -0.03, 0.45] + [0.0] * 1021, top_k=3)
for r in results:
    print(f"  {r.id}: {r.score:.4f}")

# --- Index status ---
info = index.status()
print(f"\nIndex {info.index_id}: {info.status}, {info.vector_count} vectors")
