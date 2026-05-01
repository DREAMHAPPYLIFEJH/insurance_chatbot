"""
RecommendationAgent
고객 상황 → 보험 상품 추천
"""

import re
from ai_engine import _load_embed, search_products, search_history, _get_full_chunks_by_source, _run_llm


def _clean_source_name(source: str) -> str:
    name = source.replace("상품요약서_", "").replace(".pdf", "")
    name = re.sub(r'[_\s]\d{8}.*$', '', name)
    name = re.sub(r'\s*\(\d+\)$', '', name)
    return name.strip()


def _build_prompt(user_input: str, product_texts: list, history: list) -> str:
    product_ctx = ""
    if product_texts:
        product_ctx = "\n[관련 보험 상품 전체 내용]\n"
        for i, text in enumerate(product_texts, 1):
            product_ctx += f"\n--- 상품 {i} ---\n{text}\n"

    history_ctx = ""
    if history:
        history_ctx = "\n[유사 가입 이력]\n"
        for i, h in enumerate(history, 1):
            history_ctx += f"{i}. {h['text']} (유사도: {h['similarity']})\n"

    return f"""당신은 동양생명 보험 가입설계 전문 AI 어시스턴트입니다.
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


def _parse_response(response: str) -> dict:
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

    for line in response.strip().split("\n"):
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()

        if   "추천상품"   in key: result["product_name"]     = val
        elif "상품군"     in key: result["product_group"]    = val
        elif "상품유형"   in key: result["product_type"]     = val
        elif "납입기간"   in key: result["payment_period"]   = val
        elif "보험기간"   in key: result["insurance_period"] = val
        elif "납입주기"   in key: result["payment_cycle"]    = val
        elif "추천근거"   in key: result["reason"]           = val
        elif "가입금액"   in key:
            try: result["amount"] = int("".join(filter(str.isdigit, val)))
            except: pass
        elif "예상월보험료" in key:
            try: result["monthly_premium"] = int("".join(filter(str.isdigit, val)))
            except: pass
        elif "추천정확도" in key:
            try: result["ai_accuracy"] = int("".join(filter(str.isdigit, val)))
            except: pass
        elif "주요보장"   in key:
            result["coverage"] = [c.strip() for c in val.split(",") if c.strip()]

    return result


def run(user_input: str) -> dict:
    print(f"\n[추천 에이전트] 추천 시작: '{user_input}'")

    # 1. RAG 검색 (임베딩 1회 계산 후 재사용)
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
    product_texts = []
    for i, p in enumerate(products):
        source = p["metadata"].get("source", "")
        if not source:
            continue
        if i == 0:
            full_chunks = _get_full_chunks_by_source(source)
            text = "\n".join(c["text"] for c in full_chunks)[:1000]
        else:
            text = p["text"][:400]
        product_texts.append(text)

    # 3. 프롬프트 생성 → 추론 → 파싱
    prompt = _build_prompt(user_input, product_texts, history)
    print(f"  [프롬프트] 총 길이: {len(prompt)}자")
    response = _run_llm(prompt)
    result = _parse_response(response)
    print(f"[추천 에이전트] 추천 상품: {result['product_name']}")

    # 4. RAG Top3 상품 정보 추가 (UI 표시용)
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
