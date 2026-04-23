"""
SpecialTermsAgent
특약 구성 / 선택 상담 — LangChain RunnableWithMessageHistory 사용

대화 흐름 예시:
  1턴: "암보험 특약 추천해줘"    → 추천 특약 목록 반환 + 히스토리 저장
  2턴: "입원일당은 빼줘"        → 이전 맥락 보고 수정
  3턴: "골절특약도 추가해줘"    → 누적 상태 반영
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from ai_engine import load_lc_llm, search_products


# ─────────────────────────────────────────
# 프롬프트
# ─────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 동양생명 보험 특약 전문 상담사입니다.
이전 대화를 참고하여 설계사의 요청에 답변하세요.

[관련 약관 내용]
{context}

[현재 선택된 특약 목록]
{current_terms}

[응답 형식]
추천특약: (최종 전체 특약 목록을 콤마로 나열)
변경사항: (추가/제거된 특약 설명)
추천이유: (2문장 이내)
주의사항: (1문장 이내, 없으면 없음)"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def run(
    user_input:     str,
    chat_history:   ChatMessageHistory,
    current_source: str  = None,
    current_terms:  list = None,
) -> dict:
    """
    Parameters:
        user_input     : 설계사 입력
        chat_history   : ChatMessageHistory (session_state에서 전달)
        current_source : 현재 대화 중인 상품 PDF 파일명 (향후 where 필터용)
        current_terms  : 현재 선택된 특약 목록

    Returns:
        {
            intent       : "special_terms"
            recommended  : [특약명, ...]   ← 최종 전체 목록
            added        : [추가된 특약]
            removed      : [제거된 특약]
            reason       : str
            caution      : str
            raw_response : str
        }
    """
    current_terms = current_terms or []

    # 1. RAG 검색 (current_source 있으면 해당 상품 내에서만 검색)
    chunks  = search_products(user_input, n_results=5, source=current_source)
    context = "\n".join(c["text"] for c in chunks)[:2000]
    print(f"[특약 에이전트] source={current_source}, chunk {len(chunks)}개 검색됨")

    # 2. 현재 특약 목록 문자열 변환
    terms_str = ", ".join(current_terms) if current_terms else "없음"

    # 3. 체인 구성
    llm   = load_lc_llm()
    chain = PROMPT | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=lambda _: chat_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 4. 추론
    response = chain_with_history.invoke(
        {
            "input":         user_input,
            "context":       context,
            "current_terms": terms_str,
        },
        config={"configurable": {"session_id": "special_terms"}},
    )

    # HuggingFacePipeline은 str 반환
    response_text = response if isinstance(response, str) else response.content

    # 5. 파싱
    result = {
        "intent":       "special_terms",
        "recommended":  [],
        "added":        [],
        "removed":      [],
        "reason":       "",
        "caution":      "",
        "raw_response": response_text,
    }

    for line in response_text.strip().split("\n"):
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()

        if "추천특약" in key:
            new_terms = [t.strip() for t in val.split(",") if t.strip() and t.strip() != "없음"]
            result["recommended"] = new_terms
            result["added"]       = [t for t in new_terms if t not in current_terms]
            result["removed"]     = [t for t in current_terms if t not in new_terms]
        elif "추천이유" in key: result["reason"]  = val
        elif "주의사항" in key: result["caution"] = val

    return result
