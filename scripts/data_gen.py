"""
라우터 학습 데이터 생성
  1단계: 수동 시드 데이터 (클래스당 20개) → router_train.csv 저장
  2단계: EXAONE으로 합성 데이터 (클래스당 80개) 추가 생성
  출력: router_train.csv (text, label, source)
"""

import csv
import re
import sys
sys.path.append(".")
from ai_engine import _load_llm

# ─────────────────────────────────────────
# 클래스 정의
# ─────────────────────────────────────────
CLASSES = {
    "recommendation": "보험 상품 추천 요청 (고객 나이/성별/예산/희망 보장 등을 말하며 추천 요청)",
    "product_info":   "특정 보험 상품의 보장내용/약관/조건 조회",
    "special_terms":  "특약 구성/선택/추가에 관한 상담",
    "general_qa":     "보험 일반 지식 질문 (용어, 개념, 제도 설명 등)",
}

# ─────────────────────────────────────────
# 시드 데이터 (클래스당 20개, 수동 작성)
# ─────────────────────────────────────────
SEED_DATA = {
    "recommendation": [
        "45세 남성 고객인데 암보험 추천해줘",
        "30대 여성 직장인, 월 10만원 이내로 건강보험 뭐가 좋아?",
        "60세 남성 부모님 보험 들어드리고 싶어",
        "어린이 보험 추천해줘, 5세 남아야",
        "40대 여성 고객, 암 가족력 있어. 월 15만원 이내",
        "35세 남성, 운전 많이 해서 상해보험 필요해",
        "50대 부부 종신보험 뭐가 좋아?",
        "20대 직장인 첫 보험, 실손 위주로 추천해줘",
        "당뇨 있는 55세 남성 고객, 간편심사 상품 있어?",
        "연금보험 알아보는 40대 여성 고객이야",
        "치아보험 추천해줘, 50대 여성",
        "아이 태어났는데 태아보험 추천 부탁해",
        "월 20만원 예산으로 최대 보장 받을 수 있는 상품이 뭐야",
        "65세 어머니 치매 보험 추천해줘",
        "30대 초반 남성, 사망보장 위주로 추천해줘",
        "고혈압 있는 고객인데 가입 가능한 보험 있어?",
        "저축성 보험 알아보는 고객이야",
        "암 치료 후 완치된 고객인데 가입 가능한 보험 추천해줘",
        "퇴직 앞둔 58세 남성 고객, 연금 설계해줘",
        "월 5만원으로 들 수 있는 보험이 뭐가 있어?",
    ],
    "product_info": [
        "수호천사 암보험 보장 내용이 뭐야?",
        "우리WON 건강보험 어떤 거 보장해줘?",
        "엔젤 하이브리드 연금보험 설명해줘",
        "실손보험 보장 범위가 어디까지야?",
        "수호천사 종신보험 납입기간 옵션이 어떻게 돼?",
        "다이렉트 보장보험 가입조건 알려줘",
        "치아보험 임플란트 보장이 있어?",
        "우리WON 미니상해보험 보험료 어느 정도야?",
        "어린이보험 보장 항목 알려줘",
        "간편보장보험이랑 일반 보장보험 차이가 뭐야?",
        "유니버셜 종신보험 특징이 뭐야?",
        "암보험 면책기간이 어떻게 돼?",
        "수호천사 실손 보장보험 갱신 주기 알려줘",
        "누구나행복연금보험 수령 방식이 어떻게 돼?",
        "독특한암치료보험 주요 보장 내용이 뭐야?",
        "꿈나무 우리아이보험 가입 가능 나이가 어떻게 돼?",
        "치매간병보험 지급 조건이 뭐야?",
        "VIP플러스 정기보험 사망보험금이 얼마야?",
        "건강검진 수술보험 어떤 수술 보장해줘?",
        "온라인 저축보험 금리가 어떻게 돼?",
    ],
    "special_terms": [
        "암보험에 어떤 특약 넣는 게 좋아?",
        "종신보험 특약 구성 어떻게 하면 좋을까?",
        "실손 특약 추가하면 보험료 얼마나 올라?",
        "입원일당 특약 추가해야 할지 모르겠어",
        "뇌혈관질환 특약이 필요할까?",
        "암 직접치료비 특약 설명해줘",
        "수술비 특약이랑 입원비 특약 중 뭐가 더 나아?",
        "3대 질병 특약이 어떤 거야?",
        "재해 관련 특약 어떤 게 있어?",
        "소득보상 특약 추가할만해?",
        "특약 너무 많이 넣으면 어떤 문제가 있어?",
        "기본계약이랑 특약 차이가 뭐야?",
        "어린이보험 특약 뭐 넣어야 해?",
        "치아 특약 추가 가능한 보험이 있어?",
        "간병인 지원 특약 설명해줘",
        "골절 특약 필요한 고객 유형이 어떻게 돼?",
        "운전자 특약 추가했을 때 보장 범위가 어떻게 돼?",
        "특약 중도 해지 가능해?",
        "주계약 없이 특약만 가입 가능해?",
        "납입면제 특약이 뭐야?",
    ],
    "general_qa": [
        "실손보험이 뭐야?",
        "종신보험이랑 정기보험 차이가 뭐야?",
        "보험 갱신형이 뭔지 설명해줘",
        "면책기간이 뭐야?",
        "보험료 납입면제 조건이 뭐야?",
        "해지환급금이 뭐야?",
        "보험 계약 이전이 뭐야?",
        "보험 청약 철회 어떻게 해?",
        "표준형이랑 실손형 차이가 뭐야?",
        "보험사 지급여력비율이 뭐야?",
        "예정이율이 뭐야?",
        "순수보장형이랑 환급형 차이가 뭐야?",
        "보험 계약 부활이 뭐야?",
        "실손보험 자기부담금이 뭐야?",
        "보험 고지의무가 뭐야?",
        "갱신형 보험 갱신 거절 가능해?",
        "보험금 청구할 때 필요한 서류가 뭐야?",
        "중복보험 보장이 어떻게 돼?",
        "암의 정의가 보험에서 어떻게 돼?",
        "변액보험이 뭐야?",
    ],
}

OUTPUT_CSV = "./router_train.csv"


# ─────────────────────────────────────────
# 1단계: 시드 데이터 CSV 저장
# ─────────────────────────────────────────
def save_seed_data():
    rows = []
    for label, texts in SEED_DATA.items():
        for text in texts:
            rows.append({"text": text, "label": label, "source": "manual"})

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[1단계] 시드 데이터 저장 완료: {len(rows)}개 → {OUTPUT_CSV}")
    return rows


# ─────────────────────────────────────────
# 2단계: EXAONE으로 합성 데이터 생성
# ─────────────────────────────────────────
def generate_synthetic(label: str, description: str, n: int = 80) -> list[str]:
    """EXAONE에게 특정 클래스 질문 n개 생성 요청"""
    seed_examples = "\n".join(f"- {t}" for t in SEED_DATA[label][:5])

    prompt = f"""보험 설계사가 사용하는 챗봇에 입력할 법한 질문을 생성해주세요.

질문 유형: {description}

예시:
{seed_examples}

위 예시와 비슷한 스타일로, 다양한 표현을 사용해 질문 {n}개를 생성해주세요.
반드시 번호 목록 형식으로 출력하세요.
1. (질문)
2. (질문)
...{n}. (질문)"""

    model, tokenizer = _load_llm()

    messages = [
        {"role": "system", "content": "당신은 보험 챗봇 학습 데이터 생성 전문가입니다. 요청한 형식을 정확히 따르세요."},
        {"role": "user",   "content": prompt},
    ]

    import torch
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    # "숫자. 질문" 패턴 파싱
    lines = re.findall(r'^\d+\.\s+(.+)', text, re.MULTILINE)
    return [l.strip() for l in lines if len(l.strip()) > 5]


def augment_with_llm(n_per_class: int = 80):
    """EXAONE으로 합성 데이터 생성 후 CSV에 추가"""
    print(f"\n[2단계] EXAONE 합성 데이터 생성 시작 (클래스당 {n_per_class}개 목표)")

    new_rows = []
    for label, description in CLASSES.items():
        print(f"  [{label}] 생성 중...")
        generated = generate_synthetic(label, description, n=n_per_class)
        print(f"  [{label}] {len(generated)}개 생성됨")
        for text in generated:
            new_rows.append({"text": text, "label": label, "source": "llm"})

    # 기존 CSV에 추가
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source"])
        writer.writerows(new_rows)

    print(f"\n[2단계] 합성 데이터 추가 완료: {len(new_rows)}개 → {OUTPUT_CSV}")
    return new_rows


# ─────────────────────────────────────────
# 통계 출력
# ─────────────────────────────────────────
def print_stats():
    from collections import Counter
    counts = Counter()
    with open(OUTPUT_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row["label"]] += 1

    print("\n" + "="*40)
    print("  학습 데이터 통계")
    print("="*40)
    for label, count in counts.items():
        print(f"  {label:20s}: {count}개")
    print(f"  {'합계':20s}: {sum(counts.values())}개")
    print("="*40)


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-only", action="store_true",
                        help="시드 데이터만 생성 (EXAONE 호출 없음)")
    parser.add_argument("--n", type=int, default=80,
                        help="클래스당 합성 데이터 수 (기본: 80)")
    args = parser.parse_args()

    # 1단계: 시드 저장 (항상 실행)
    save_seed_data()

    # 2단계: EXAONE 합성 (--seed-only 아닐 때만)
    if not args.seed_only:
        augment_with_llm(n_per_class=args.n)

    print_stats()
