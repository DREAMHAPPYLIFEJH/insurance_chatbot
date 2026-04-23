import chromadb

client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("insurance_products")

# 1개만 꺼내서 확인
result = collection.get(
    limit=1,
    include=["documents", "metadatas", "embeddings"]
)

print("텍스트:", result["documents"][0][:80])
print("메타데이터:", result["metadatas"][0])
print("벡터 차원 수:", len(result["embeddings"][0]))  # ← 이게 핵심
print("벡터 앞 5개:", result["embeddings"][0][:5])
