from transformers import pipeline

clf = pipeline('text-classification', model='./router_model')

tests = [
    ('이 고객한테 뭐 팔면 좋을까요?',         'recommendation'),
    ('우리WON 암보험 갱신 조건 좀 알려줘',     'product_info'),
    ('납입면제 특약이 있는지 확인해줘',         'special_terms'),
    ('보험에서 고지의무 위반하면 어떻게 돼?',   'general_qa'),
    ('50대 여성인데 뭐가 필요할까',            'recommendation'),
    ('실손보험이랑 건강보험 차이가 뭐야?',      'general_qa'),
    ('수호천사 암보험 특약 구성 어떻게 해?',    'special_terms'),
    ('이 고객 나이가 45세인데 뭘 추천해줘',    'recommendation'),
    ('엔젤 연금보험 수령 방식 알려줘',          'product_info'),
    ('변액보험이 뭔지 설명해줘',               'general_qa'),
]

print(f"\n{'예측':20s} {'확률':8s} {'정답':20s} {'일치':4s}  질문")
print("─" * 90)

correct = 0
for text, answer in tests:
    r      = clf(text)[0]
    pred   = r['label']
    score  = r['score']
    match  = '✅' if pred == answer else '❌'
    if pred == answer:
        correct += 1
    print(f"{pred:20s} {score:.2%}  {answer:20s} {match}   {text}")

print("─" * 90)
print(f"정확도: {correct}/{len(tests)} ({correct/len(tests):.0%})")
