"""
ProductInfoAgent
특정 상품 보장내용 / 약관 조회
"""
from ai_engine import get_product_summary

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
