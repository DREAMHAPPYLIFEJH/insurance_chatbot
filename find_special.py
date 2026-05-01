import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("insurance_products")

# 특약 청크만 필터링
result = collection.get(
    where={"category": "약관"},
    limit=10,
    include=["documents", "metadatas"]
)

for doc, meta in zip(result["documents"], result["metadatas"]):
    print(f"[{meta['source']}] {meta['category']}")
    print(doc[:200])
    print()