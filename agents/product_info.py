"""
ProductInfoAgent
특정 상품 보장내용 / 약관 조회
"""
from ai_engine import search_products, _get_full_chunks_by_source, _run_llm

def run(user_input: str) -> dict:
    result = get_product_summary(user_input)
    return {
        "intent":       "product_info",
        "product_name": result.get("product_name", ""),
        "source":       result.get("source", ""),
        "coverage":     result.get("coverage", ""),
        "payment_period":   result.get("payment_period", ""),
        "insurance_period": result.get("insurance_period", ""),
        "payment_cycle":    result.get("payment_cycle", ""),
        "amount":       result.get("amount", ""),
        "special_terms":result.get("special_terms", ""),
        "summary":      result.get("summary", ""),
    }

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

    print(f"  [프롬프트] 총 길이: {len(prompt)}자")
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