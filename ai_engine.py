"""
AI 엔진
- EXAONE 3.5 2.4B (HuggingFace Transformers)
- ChromaDB RAG (보험 상품 + 가입설계 이력 검색)

실행 전 설치:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers>=4.43.0 accelerate chromadb sentence-transformers
"""

import torch
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
LLM_MODEL_NAME   = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
EMBED_MODEL_NAME = "BAAI/bge-m3"
CHROMA_PATH      = "./chroma_db"

# ─────────────────────────────────────────
# 모델 로드 (최초 1회만 로드 → 캐싱)
# ─────────────────────────────────────────
_llm_model     = None
_llm_tokenizer = None
_embed_model   = None
_chroma_client = None


def _load_llm():
    """EXAONE 3.5 모델 로드 (최초 1회)"""
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        print(f"[AI 엔진] LLM 모델 로딩: {LLM_MODEL_NAME}")
        _llm_tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            trust_remote_code=True
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            _llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=device,
                attn_implementation="flash_attention_2",
            )
            print("[AI 엔진] Flash Attention 2 활성화")
        except Exception:
            _llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=device,
            )
            print("[AI 엔진] 기본 Attention으로 로드")
        print("[AI 엔진] LLM 로드 완료")
    return _llm_model, _llm_tokenizer


def _load_embed():
    """임베딩 모델 로드 (최초 1회)"""
    global _embed_model
    if _embed_model is None:
        print(f"[AI 엔진] 임베딩 모델 로딩: {EMBED_MODEL_NAME}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
        print(f"[AI 엔진] 임베딩 모델 로드 완료 (device={device})")
    return _embed_model


def _load_chroma():
    """ChromaDB 클라이언트 로드 (최초 1회)"""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


# ─────────────────────────────────────────
# RAG: ChromaDB 검색
# ─────────────────────────────────────────
def search_products(query: str, n_results: int = 3, source: str = None, query_embedding: list = None) -> list[dict]:
    """보험 상품 ChromaDB에서 유사 상품 검색."""
    try:
        client     = _load_chroma()
        collection = client.get_collection("insurance_products")

        if query_embedding is None:
            embed_model     = _load_embed()
            query_embedding = embed_model.encode(
                [query], normalize_embeddings=True
            ).tolist()

        where = {"source": source} if source else None

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
            **({"where": where} if where else {})
        )

        items = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            items.append({
                "text":       doc,
                "metadata":   meta,
                "similarity": round(1 - dist, 4)
            })
        return items

    except Exception as e:
        print(f"[RAG] 상품 검색 실패: {e}")
        return []


def search_history(query: str, n_results: int = 3, query_embedding: list = None) -> list[dict]:
    """가입설계 이력 ChromaDB에서 유사 케이스 검색."""
    try:
        client     = _load_chroma()
        collection = client.get_collection("design_history")

        if query_embedding is None:
            embed_model     = _load_embed()
            query_embedding = embed_model.encode(
                [query], normalize_embeddings=True
            ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        items = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            items.append({
                "text":       doc,
                "metadata":   meta,
                "similarity": round(1 - dist, 4)
            })
        return items

    except Exception as e:
        print(f"[RAG] 이력 검색 실패: {e}")
        return []


def _get_full_chunks_by_source(source: str) -> list[dict]:
    """특정 source(PDF 파일명)의 모든 chunk를 chunk_index 순서로 반환."""
    client     = _load_chroma()
    collection = client.get_collection("insurance_products")

    results = collection.get(
        where={"source": source},
        include=["documents", "metadatas"]
    )

    items = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        items.append({"text": doc, "metadata": meta})

    items.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
    return items


# ─────────────────────────────────────────
# EXAONE 3.5 추론
# ─────────────────────────────────────────
def _run_llm(prompt: str, max_new_tokens: int = 256) -> str:
    """EXAONE 3.5로 추론 실행"""
    model, tokenizer = _load_llm()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful insurance design assistant specializing in Korean life insurance products. Always respond in Korean."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs    = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            use_cache=True,
        )

    generated = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ─────────────────────────────────────────
# Warm-up
# ─────────────────────────────────────────
def warmup_models():
    """서버 시작 시 모든 AI 모델을 미리 로드한다."""
    import time

    t = time.time()
    print("[Warmup] Embedding 모델 로드 중...")
    _load_embed()
    print(f"[Warmup] Embedding 완료 ({time.time()-t:.1f}s)")

    t = time.time()
    print("[Warmup] ChromaDB 로드 중...")
    _load_chroma()
    print(f"[Warmup] ChromaDB 완료 ({time.time()-t:.1f}s)")

    t = time.time()
    print("[Warmup] LLM 로드 중...")
    _load_llm()
    print(f"[Warmup] LLM 완료 ({time.time()-t:.1f}s)")
