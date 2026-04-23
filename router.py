"""
Router
사용자 입력을 분류하여 적절한 에이전트로 라우팅.

분류 방식 (하이브리드):
  1단계: 키워드 매칭 → 즉시 반환 (LLM 호출 없음)
  2단계: klue/roberta-base 추론 → 키워드 미감지 시

session_ctx (session_state에서 전달):
  current_source  : 현재 대화 중인 상품 PDF 파일명
  current_intent  : 직전 intent
  current_terms   : 현재 선택된 특약 목록
  chat_history    : ChatMessageHistory (special_terms 전용)
"""

from transformers import pipeline

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
ROUTER_MODEL = "./router_model"

KEYWORDS = {
    "recommendation": ["추천", "가입", "들고싶", "뭐가 좋", "골라줘", "뭘 팔", "뭐 팔", "설계해줘", "뭐가 필요"],
    "product_info":   ["보장내용", "보장 내용", "알려줘", "설명해줘", "어떻게 돼", "조건", "갱신", "수령"],
    "special_terms":  ["특약", "특별약관", "넣어야", "추가해야", "빼줘", "추가해줘", "넣어줘"],
}

# ─────────────────────────────────────────
# 모델 로드 (최초 1회)
# ─────────────────────────────────────────
_clf = None

def _load_clf():
    global _clf
    if _clf is None:
        print("[라우터] 분류 모델 로딩...")
        _clf = pipeline("text-classification", model=ROUTER_MODEL)
        print("[라우터] 분류 모델 로드 완료")
    return _clf


# ─────────────────────────────────────────
# 분류
# ─────────────────────────────────────────
def classify(user_input: str, current_intent: str = None) -> tuple[str, str]:
    """
    입력 문장을 분류하여 intent 반환.

    special_terms 대화 중 애매한 입력("빼줘", "추가해줘")은
    current_intent 가 special_terms면 그대로 유지.

    Returns:
        (intent, method)
    """
    # 특약 대화 중 수정 요청 → intent 유지
    CONTINUATION_KEYWORDS = ["빼줘", "추가해줘", "넣어줘", "바꿔줘", "제거해줘"]
    if current_intent == "special_terms" and any(kw in user_input for kw in CONTINUATION_KEYWORDS):
        print(f"[라우터] 대화 연속 감지 → special_terms 유지")
        return "special_terms", "continuation"

    # 1단계: 키워드 매칭
    for intent, keywords in KEYWORDS.items():
        if any(kw in user_input for kw in keywords):
            print(f"[라우터] 키워드 분류: {intent}")
            return intent, "keyword"

    # 2단계: klue/roberta-base 추론
    clf    = _load_clf()
    result = clf(user_input)[0]
    intent = result["label"]
    score  = result["score"]
    print(f"[라우터] 모델 분류: {intent} ({score:.2%})")
    return intent, "model"


# ─────────────────────────────────────────
# source 추출 (RAG 검색으로 상품 특정)
# ─────────────────────────────────────────
def _resolve_source(user_input: str, current_source: str, intent: str) -> str:
    """
    recommendation → source 초기화 (새 상품 찾는 중)
    product_info / special_terms → RAG top1으로 source 특정
    general_qa → 기존 source 유지 or None
    """
    if intent == "recommendation":
        return None

    if intent in ("product_info", "special_terms"):
        from ai_engine import search_products
        results = search_products(user_input, n_results=1)
        if results:
            return results[0]["metadata"].get("source", current_source)

    return current_source  # general_qa or 변경 없음


# ─────────────────────────────────────────
# 라우팅
# ─────────────────────────────────────────
def route(user_input: str, session_ctx: dict) -> tuple[dict, dict]:
    """
    입력 문장을 분류하고 해당 에이전트 실행.

    Parameters:
        user_input  : 설계사 입력
        session_ctx : {
            current_source  : str | None,
            current_intent  : str | None,
            current_terms   : list,
            chat_history    : ChatMessageHistory,
        }

    Returns:
        (result, updated_ctx)
        result      : 에이전트 반환 dict
        updated_ctx : 업데이트된 session_ctx
    """
    from agents import recommendation, product_info, special_terms, general_qa

    current_source = session_ctx.get("current_source")
    current_intent = session_ctx.get("current_intent")
    current_terms  = session_ctx.get("current_terms", [])
    chat_history   = session_ctx.get("chat_history")

    # 1. 분류
    intent, method = classify(user_input, current_intent)

    # 2. source 추적
    new_source = _resolve_source(user_input, current_source, intent)

    # 3. 에이전트 실행
    print(f"[라우터] → {intent} 에이전트 실행 (방식: {method}, source: {new_source})")

    if intent == "recommendation":
        result = recommendation.run(user_input)

    elif intent == "product_info":
        result = product_info.run(user_input)

    elif intent == "special_terms":
        result = special_terms.run(
            user_input    = user_input,
            chat_history  = chat_history,
            current_source= new_source,
            current_terms = current_terms,
        )
        # 특약 목록 업데이트
        current_terms = result.get("recommended", current_terms)

    elif intent == "general_qa":
        result = general_qa.run(user_input)

    result["intent"]       = intent
    result["route_method"] = method

    # 4. 컨텍스트 업데이트
    updated_ctx = {
        "current_source": new_source,
        "current_intent": intent,
        "current_terms":  current_terms,
        "chat_history":   chat_history,
    }

    return result, updated_ctx


# ─────────────────────────────────────────
# Warm-up
# ─────────────────────────────────────────
def warmup():
    """서버 시작 시 라우터 모델을 미리 로드한다."""
    import time
    t = time.time()
    print("[Warmup] 라우터(klue/roberta) 로드 중...")
    _load_clf()
    print(f"[Warmup] 라우터 완료 ({time.time()-t:.1f}s)")


# ─────────────────────────────────────────
# 단독 테스트
# ─────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "45세 남성 암보험 추천해줘",
        "수호천사 암보험 보장내용 알려줘",
        "특약 어떻게 구성해야 해?",
        "실손보험이 뭐야?",
        "이 고객한테 뭐가 좋을까",
    ]

    for t in tests:
        intent, method = classify(t)
        print(f"[{method:12s}] {intent:20s} | {t}")
