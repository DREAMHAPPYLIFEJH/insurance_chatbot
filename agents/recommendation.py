"""
RecommendationAgent
고객 상황 → 보험 상품 추천
"""
from ai_engine import get_recommendation

def run(user_input: str) -> dict:
    return get_recommendation(user_input)
