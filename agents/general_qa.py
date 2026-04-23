"""
GeneralQAAgent
보험 일반 지식 질문 답변
"""
from ai_engine import search_products, _run_llm

def run(user_input: str) -> dict:
    # 관련 chunk 검색
    chunks = search_products(user_input, n_results=3)
    context = "\n".join(c["text"] for c in chunks)[:1500]

    prompt = f"""당신은 동양생명 보험 전문 상담사입니다.
아래 참고 내용을 활용하여 설계사의 질문에 쉽고 명확하게 답변해주세요.

[참고 내용]
{context}

[설계사 질문]
{user_input}

[답변] (3~5문장으로 간결하게)"""

    response = _run_llm(prompt, max_new_tokens=256)

    return {
        "intent":       "general_qa",
        "answer":       response,
        "raw_response": response,
    }
