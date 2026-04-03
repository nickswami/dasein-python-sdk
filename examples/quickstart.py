"""
Dasein quickstart example.

Shows how to create an index, upsert documents, and query.
"""
from dasein import Client

client = Client(api_key="dsk_your_api_key_here")

index = client.create_index("my-rag-docs", model="bge-large-en-v1.5")

index.upsert([
    {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog", "metadata": {"tenant": "acme", "type": "faq"}},
    {"id": "doc2", "text": "Machine learning is a subset of artificial intelligence", "metadata": {"tenant": "acme", "type": "article"}},
    {"id": "doc3", "text": "PostgreSQL supports full text search natively", "metadata": {"tenant": "globex", "type": "docs"}},
])

results = index.query("what is machine learning?", top_k=5)
for r in results:
    print(f"  {r.id}: {r.score:.4f} - {r.text}")

results = index.query("machine learning", top_k=5, filter={"tenant": "acme"})
for r in results:
    print(f"  {r.id}: {r.score:.4f} - {r.text} [{r.metadata}]")

results = index.query(vector=[0.12, -0.03, 0.45] + [0.0] * 1021, top_k=3)
for r in results:
    print(f"  {r.id}: {r.score:.4f}")

info = index.status()
print(f"Index {info.index_id}: {info.status}, {info.vector_count} vectors")
