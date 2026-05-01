"""
SpecialTermsAgent
특약 구성 / 선택 상담

대화 흐름 예시:
  1턴: "암보험 특약 추천해줘"    → 추천 특약 목록 반환 + 히스토리 저장
  2턴: "입원일당은 빼줘"        → 이전 맥락 보고 수정
  3턴: "골절특약도 추가해줘"    → 누적 상태 반영
"""

from ai_engine import _run_llm, _load_llm, search_products


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def run(
    user_input:     str,
    chat_history,
    current_source: str  = None,
    current_terms:  list = None,
) -> dict:
    current_terms = current_terms or []

    # 1. RAG 검색
    query   = f"{user_input} 특약 보장금액 지급사유"
    chunks  = search_products(query, n_results=5, source=current_source)
    context = "\n".join(c["text"] for c in chunks)[:2000]
    print(f"[특약 에이전트] source={current_source}, chunk {len(chunks)}개 검색됨")

    # 2. 현재 특약 목록 문자열 변환
    terms_str = ", ".join(current_terms) if current_terms else "없음"

    # 3. EXAONE chat template 적용 메시지 구성
    system_content = f"""당신은 동양생명 보험 특약 전문 상담사입니다.
이전 대화를 참고하여 설계사의 요청에 답변하세요.

[관련 약관 내용]
{context}

[현재 선택된 특약 목록]
{terms_str}

반드시 아래 형식으로만 답변하세요:
추천특약: 특약명1, 특약명2, 특약명3
추천이유: (2문장 이내로 추천 이유 설명)
주의사항: (1문장 이내, 없으면 없음)"""

    messages = [{"role": "system", "content": system_content}]

    # 이전 대화 히스토리 삽입
    if chat_history and hasattr(chat_history, "messages"):
        for msg in chat_history.messages:
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
            messages.append({"role": role, "content": msg.content})

    messages.append({"role": "user", "content": user_input})

    # 4. EXAONE chat template → _run_llm 직접 호출
    _, tokenizer = _load_llm()
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"  [프롬프트] 총 길이: {len(prompt_text)}자")
    response_text = _run_llm(prompt_text, max_new_tokens=256)

    # 5. 히스토리 저장
    if chat_history:
        chat_history.add_user_message(user_input)
        chat_history.add_ai_message(response_text)

    print(f"[특약 에이전트] 응답:\n{response_text[:200]}")

    # 6. 파싱
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
