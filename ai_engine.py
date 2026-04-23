"""
AI 추천 엔진
- EXAONE 3.5 2.4B (HuggingFace Transformers)
- ChromaDB RAG (보험 상품 + 가입설계 이력 검색)

실행 전 설치:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers>=4.43.0 accelerate chromadb sentence-transformers
"""

import re
import torch
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
LLM_MODEL_NAME  = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
EMBED_MODEL_NAME = "BAAI/bge-m3"
CHROMA_PATH     = "./chroma_db"

# ─────────────────────────────────────────
# 모델 로드 (최초 1회만 로드 → 캐싱)
# ─────────────────────────────────────────
_llm_model     = None
_llm_tokenizer = None
_embed_model   = None
_chroma_client = None
_lc_llm        = None   # LangChain LLM 래퍼 캐시

def _load_llm():
    """EXAONE 3.5 모델 로드 (최초 1회)"""
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        print(f"[AI 엔진] LLM 모델 로딩: {LLM_MODEL_NAME}")
        _llm_tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            trust_remote_code=True
        )
        # Flash Attention 2 시도 (미지원 GPU면 기본 attention으로 fallback)
        try:
            _llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            print("[AI 엔진] Flash Attention 2 활성화")
        except Exception:
            _llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
            print("[AI 엔진] 기본 Attention으로 로드")
        print("[AI 엔진] LLM 로드 완료")
    return _llm_model, _llm_tokenizer


def load_lc_llm():
    """EXAONE을 LangChain HuggingFacePipeline으로 래핑 (최초 1회)"""
    global _lc_llm
    if _lc_llm is None:
        from langchain_community.llms import HuggingFacePipeline
        model, tokenizer = _load_llm()
        pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        _lc_llm = HuggingFacePipeline(pipeline=pipe)
        print("[AI 엔진] LangChain LLM 래퍼 완료")
    return _lc_llm


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
    """
    보험 상품 ChromaDB에서 유사 상품 검색.

    Parameters:
        source          : PDF 파일명 지정 시 해당 상품 내에서만 검색
        query_embedding : 미리 계산된 임베딩 (없으면 내부에서 계산)

    반환: [{"text": str, "metadata": dict, "similarity": float}, ...]
    """
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
    """
    가입설계 이력 ChromaDB에서 유사 케이스 검색.

    Parameters:
        query_embedding : 미리 계산된 임베딩 (없으면 내부에서 계산)

    반환: [{"text": str, "metadata": dict}, ...]
    """
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


# ─────────────────────────────────────────
# PDF 파일명 → 상품명 정제
# ─────────────────────────────────────────
def _clean_source_name(source: str) -> str:
    """
    '상품요약서_무배당수호천사NEW실속플러스하나로암보험_20251001.pdf'
    → '무배당수호천사NEW실속플러스하나로암보험'
    """
    name = source.replace("상품요약서_", "").replace(".pdf", "")
    name = re.sub(r'[_\s]\d{8}.*$', '', name)   # _YYYYMMDD 이후 제거
    name = re.sub(r'\s*\(\d+\)$', '', name)      # 중복 파일 번호 (1) 제거
    return name.strip()


# ─────────────────────────────────────────
# 상품 전체 chunk 수집 (source 기준)
# ─────────────────────────────────────────
def _get_full_chunks_by_source(source: str) -> list[dict]:
    """
    특정 source(PDF 파일명)의 모든 chunk를 chunk_index 순서로 반환.
    반환: [{"text": str, "metadata": dict}, ...]
    """
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
# 프롬프트 생성
# ─────────────────────────────────────────
def _build_prompt(user_input: str, product_texts: list, history: list) -> str:
    """RAG 검색 결과를 포함한 프롬프트 생성"""

    # 상품 컨텍스트 (전체 내용)
    product_ctx = ""
    if product_texts:
        product_ctx = "\n[관련 보험 상품 전체 내용]\n"
        for i, text in enumerate(product_texts, 1):
            product_ctx += f"\n--- 상품 {i} ---\n{text}\n"

    # 이력 컨텍스트
    history_ctx = ""
    if history:
        history_ctx = "\n[유사 가입 이력]\n"
        for i, h in enumerate(history, 1):
            history_ctx += f"{i}. {h['text']} (유사도: {h['similarity']})\n"

    prompt = f"""당신은 동양생명 보험 가입설계 전문 AI 어시스턴트입니다.
설계사가 입력한 고객 상황을 분석하여 최적의 보험 상품을 추천해주세요.

{product_ctx}
{history_ctx}

[설계사 입력]
{user_input}

[응답 형식 - 반드시 아래 형식으로 응답하세요]
추천상품: (상품명)
상품군: (생명보험/건강보험/연금보험/치아보험/암보험 중 하나)
상품유형: (종신형/정기형/저축형/변액형 중 하나)
납입기간: (10년납/20년납/전기납 중 하나)
보험기간: (10년/20년/30년/종신 중 하나)
납입주기: (월납/분기납/연납 중 하나)
가입금액: (숫자, 만원 단위)
예상월보험료: (숫자, 원 단위)
추천정확도: (0-100 사이 숫자)
추천근거: (2-3문장으로 추천 이유 설명)
주요보장: (보장항목1, 보장항목2, 보장항목3)"""
    
    # ↓ 여기 추가
    print(f"[프롬프트] 총 길이: {len(prompt)}자")
    print(f"[프롬프트] 상품 {len(product_texts)}개, 각 길이: {[len(t) for t in product_texts]}")
    
    # 토큰 수도 보고 싶으면
    tokens = _llm_tokenizer(prompt)["input_ids"]
    print(f"[프롬프트] 토큰 수: {len(tokens)}")

    return prompt


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
# 응답 파싱
# ─────────────────────────────────────────
def _parse_response(response: str) -> dict:
    """
    EXAONE 응답 텍스트를 구조화된 dict로 파싱.
    파싱 실패 시 기본값 반환.
    """
    result = {
        "product_name":     "추천 상품",
        "product_group":    "건강보험",
        "product_type":     "정기형",
        "payment_period":   "20년납",
        "insurance_period": "30년",
        "payment_cycle":    "월납",
        "amount":           3000,
        "monthly_premium":  90000,
        "ai_accuracy":      85,
        "reason":           response[:200] if response else "AI 분석 결과",
        "coverage":         ["사망보험금", "입원일당", "수술비"],
        "raw_response":     response
    }

    lines = response.strip().split("\n")
    for line in lines:
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()

        if "추천상품"   in key: result["product_name"]     = val
        elif "상품군"   in key: result["product_group"]    = val
        elif "상품유형" in key: result["product_type"]     = val
        elif "납입기간" in key: result["payment_period"]   = val
        elif "보험기간" in key: result["insurance_period"] = val
        elif "납입주기" in key: result["payment_cycle"]    = val
        elif "추천근거" in key: result["reason"]           = val
        elif "가입금액" in key:
            try: result["amount"] = int("".join(filter(str.isdigit, val)))
            except: pass
        elif "예상월보험료" in key:
            try: result["monthly_premium"] = int("".join(filter(str.isdigit, val)))
            except: pass
        elif "추천정확도" in key:
            try: result["ai_accuracy"] = int("".join(filter(str.isdigit, val)))
            except: pass
        elif "주요보장" in key:
            result["coverage"] = [c.strip() for c in val.split(",") if c.strip()]

    return result


# ─────────────────────────────────────────
# 상품 요약 (신규 추가)
# ─────────────────────────────────────────
def get_product_summary(query: str) -> dict:
    """
    질의와 가장 유사한 상품의 PDF 전체 내용을 보험 상품 형식으로 정리.

    흐름:
        query → top 1 chunk 검색 → source 추출
        → 해당 PDF 전체 chunk 수집 → LLM 정리 → dict 반환

    반환:
        {
            "product_name":     str,
            "source":           str,   # PDF 파일명
            "coverage":         str,   # 보장내용
            "payment_period":   str,   # 납입기간
            "insurance_period": str,   # 보험기간
            "payment_cycle":    str,   # 납입주기
            "amount":           str,   # 가입금액
            "special_terms":    str,   # 주요 특약
            "summary":          str,   # 상품 전체 요약
        }
    """
    print(f"\n[AI 엔진] 상품 요약 시작: '{query}'")

    # 1. 관련 chunk 검색 (top 1으로 source 특정)
    chunks = search_products(query, n_results=1)
    if not chunks:
        raise ValueError(f"[get_product_summary] 관련 상품 없음: '{query}'")

    top_source = chunks[0]["metadata"].get("source", "")
    if not top_source:
        raise ValueError("[get_product_summary] source 메타데이터 없음 — rag_pipeline 재실행 필요")

    print(f"  대상 PDF: {top_source}")

    # 2. 해당 PDF의 전체 chunk 수집
    full_chunks = _get_full_chunks_by_source(top_source)
    if not full_chunks:
        raise ValueError(f"[get_product_summary] chunk 없음: {top_source}")

    print(f"  chunk 수: {len(full_chunks)}개")

    # 3. chunk 합치기 (토큰 초과 방지: 3000자 제한)
    full_text = "\n".join(c["text"] for c in full_chunks)[:3000]

    # 4. LLM으로 보험 상품 형식 정리
    prompt = f"""아래는 동양생명 보험 상품 약관 내용입니다.
이 내용을 바탕으로 보험 상품을 아래 형식에 맞게 정리해주세요.

[약관 내용]
{full_text}

[응답 형식 - 반드시 아래 형식으로 응답]
상품명: (상품명)
보장내용: (주요 보장 내용 요약)
납입기간: (납입기간 옵션)
보험기간: (보험기간 옵션)
납입주기: (월납/분기납/연납)
가입금액: (가입금액 범위)
주요특약: (주요 특약 목록)
상품요약: (2~3문장으로 상품 전체 요약)"""

    print("[AI 엔진] LLM 상품 정리 중...")
    response = _run_llm(prompt, max_new_tokens=512)

    # 5. 파싱
    result = {
        "product_name":     "",
        "source":           top_source,
        "coverage":         "",
        "payment_period":   "",
        "insurance_period": "",
        "payment_cycle":    "",
        "amount":           "",
        "special_terms":    "",
        "summary":          response,
    }

    for line in response.strip().split("\n"):
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()

        if   "상품명"   in key: result["product_name"]     = val
        elif "보장내용" in key: result["coverage"]         = val
        elif "납입기간" in key: result["payment_period"]   = val
        elif "보험기간" in key: result["insurance_period"] = val
        elif "납입주기" in key: result["payment_cycle"]    = val
        elif "가입금액" in key: result["amount"]           = val
        elif "주요특약" in key: result["special_terms"]    = val
        elif "상품요약" in key: result["summary"]          = val

    print(f"[AI 엔진] 상품 요약 완료: {result['product_name']}")
    return result


# ─────────────────────────────────────────
# 메인 함수 (chatbot_ui.py에서 호출)
# ─────────────────────────────────────────
def get_recommendation(user_input: str) -> dict:
    """
    설계사 자연어 입력 → AI 보험 상품 추천.

    Parameters:
        user_input: 설계사가 입력한 고객 상황
                    예) "45세 남성, 암 걱정, 월 15만원 이내"

    Returns:
        {
            "product_name":     str,
            "product_group":    str,
            "product_type":     str,
            "payment_period":   str,
            "insurance_period": str,
            "payment_cycle":    str,
            "amount":           int,
            "monthly_premium":  int,
            "ai_accuracy":      int,
            "reason":           str,
            "coverage":         list[str],
            "raw_response":     str
        }
    """
    print(f"\n[AI 엔진] 추천 시작: '{user_input}'")

    # 1. RAG 검색 (임베딩 1회 계산 후 재사용)
    print("[AI 엔진] RAG 검색 중...")
    embed_model     = _load_embed()
    query_embedding = embed_model.encode([user_input], normalize_embeddings=True).tolist()
    raw_products    = search_products(user_input, n_results=15, query_embedding=query_embedding)
    history         = search_history(user_input,  n_results=3,  query_embedding=query_embedding)
    print(f"  상품 chunk {len(raw_products)}개, 이력 {len(history)}개 검색됨")

    # source 기준 중복 제거 → 상위 3개 고유 상품
    seen_sources = {}
    for p in raw_products:
        source = p["metadata"].get("source", "")
        if source and source not in seen_sources:
            seen_sources[source] = p
        if len(seen_sources) == 3:
            break
    products = list(seen_sources.values())
    print(f"  고유 상품 {len(products)}개")

    # 2. 상품 내용 수집 (1등: 전체, 2~3등: 매칭 청크만)
    print("[AI 엔진] 상품 내용 수집 중...")
    product_texts = []
    for i, p in enumerate(products):
        source = p["metadata"].get("source", "")
        if not source:
            continue
        if i == 0:
            full_chunks = _get_full_chunks_by_source(source)
            text = "\n".join(c["text"] for c in full_chunks)[:1500]
            print(f"  [1등] 전체 수집: {source} ({len(text)}자)")
        else:
            text = p["text"][:400]
            print(f"  [{i+1}등] 청크만 수집: {source} ({len(text)}자)")
        product_texts.append(text)
    print(f"  상품 내용 {len(product_texts)}개 수집됨")

    # 3. 프롬프트 생성
    prompt = _build_prompt(user_input, product_texts, history)

    # 4. EXAONE 추론
    print("[AI 엔진] EXAONE 3.5 추론 중...")
    response = _run_llm(prompt)
    print(f"[AI 엔진] 추론 완료:\n{response[:200]}...")

    # 5. 응답 파싱
    result = _parse_response(response)
    print(f"[AI 엔진] 추천 상품: {result['product_name']}")

    # 6. RAG Top3 상품 정보 추가 (UI 표시용)
    result["rag_products"] = [
        {
            "product_name": _clean_source_name(p["metadata"].get("source", "")),
            "source":       p["metadata"].get("source", ""),
            "similarity":   p["similarity"],
            "category":     p["metadata"].get("category", ""),
        }
        for p in products
    ]

    return result


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

    t = time.time()
    print("[Warmup] LangChain 래퍼 로드 중...")
    load_lc_llm()
    print(f"[Warmup] LangChain 완료 ({time.time()-t:.1f}s)")


# ─────────────────────────────────────────
# 단독 테스트
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_input = "45세 남성 고객입니다. 암 가족력이 있고 월 15만원 이내로 암 보험 들고 싶어합니다."
    result = get_recommendation(test_input)

    print("\n=== 추천 결과 ===")
    for k, v in result.items():
        if k != "raw_response":
            print(f"  {k}: {v}")

    print("\n=== 상품 요약 테스트 ===")
    summary = get_product_summary(test_input)
    for k, v in summary.items():
        print(f"  {k}: {v}")