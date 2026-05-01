"""
FastAPI 백엔드 서버

엔드포인트:
  POST /api/chat             — 라우터 자동 분기 (Streamlit UI 연동)
  POST /api/recommend        — 상품 추천 직접 호출
  POST /api/product_info     — 상품 정보 직접 호출
  POST /api/special_terms    — 특약 상담 직접 호출
  POST /api/general_qa       — 일반 질의 직접 호출
  GET  /api/health           — 서버 상태 확인

API 문서: http://localhost:8000/docs

실행:
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import asyncio
import concurrent.futures
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from router import route
from routes import recommend, product_info, special_terms, general_qa

try:
    from langchain_community.chat_message_histories import ChatMessageHistory
except ImportError:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage

    class ChatMessageHistory(BaseChatMessageHistory):
        def __init__(self): self.messages = []
        def add_user_message(self, msg): self.messages.append(HumanMessage(content=msg))
        def add_ai_message(self, msg):   self.messages.append(AIMessage(content=msg))
        def clear(self):                 self.messages = []

# ─────────────────────────────────────────
# 모델 사전 로딩 (Warm-up)
# ─────────────────────────────────────────
def _warmup_all():
    """동기 블로킹 함수 — ThreadPoolExecutor에서 실행"""
    from router import warmup as router_warmup
    from ai_engine import warmup_models
    router_warmup()
    warmup_models()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[서버] 모델 사전 로딩 시작...")
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        await loop.run_in_executor(pool, _warmup_all)
    print("[서버] 모델 사전 로딩 완료 — 요청 수신 시작")
    try:
        yield
    except asyncio.CancelledError:
        pass


# ─────────────────────────────────────────
# 앱 초기화
# ─────────────────────────────────────────
app = FastAPI(
    lifespan=lifespan,
    title="동양생명 가입설계 챗봇 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 에이전트별 라우터 등록
app.include_router(recommend.router)
app.include_router(product_info.router)
app.include_router(special_terms.router)
app.include_router(general_qa.router)

# ─────────────────────────────────────────
# 서버 메모리 세션 저장소
# session_id → session_ctx
# ─────────────────────────────────────────
_sessions: dict[str, dict] = {}

def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "current_source": None,
            "current_intent": None,
            "current_terms":  [],
            "chat_history":   ChatMessageHistory(),
        }
    return _sessions[session_id]

def _clear_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]


# ─────────────────────────────────────────
# 요청/응답 스키마
# ─────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

class ChatResponse(BaseModel):
    intent:        str
    route_method:  str

    # recommendation 전용
    product_name:     Optional[str]  = None
    product_group:    Optional[str]  = None
    product_type:     Optional[str]  = None
    payment_period:   Optional[str]  = None
    insurance_period: Optional[str]  = None
    payment_cycle:    Optional[str]  = None
    amount:           Optional[int]  = None
    monthly_premium:  Optional[int]  = None
    ai_accuracy:      Optional[int]  = None
    reason:           Optional[str]  = None
    coverage:         Optional[list] = None
    rag_products:     Optional[list] = None

    # product_info 전용
    source:             Optional[str]  = None
    coverage_info:      Optional[list] = None
    summary:            Optional[str]  = None
    special_terms_info: Optional[list] = None

    # special_terms 전용
    recommended: Optional[list] = None
    added:       Optional[list] = None
    removed:     Optional[list] = None
    caution:     Optional[str]  = None

    # general_qa 전용
    answer: Optional[str] = None

    # 공통
    current_source: Optional[str]  = None
    current_terms:  Optional[list] = None
    raw_response:   Optional[str]  = None


# ─────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "sessions": len(_sessions)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    import traceback
    session_ctx = _get_session(req.session_id)

    try:
        result, updated_ctx = route(req.user_input, session_ctx)
    except Exception as e:
        traceback.print_exc()   # uvicorn 터미널에 전체 스택 출력
        raise HTTPException(status_code=500, detail=str(e))

    # 세션 업데이트
    _sessions[req.session_id] = updated_ctx

    def _list(val):
        """빈 문자열/None → None, list면 그대로"""
        if isinstance(val, list): return val
        return None

    def _int(val):
        """빈 문자열/None → None, 숫자면 int로"""
        if val is None or val == "": return None
        try: return int(val)
        except: return None

    def _str(val):
        """빈 문자열 → None"""
        return val if val else None

    # 응답 구성
    return ChatResponse(
        intent       = result.get("intent", ""),
        route_method = result.get("route_method", ""),

        # recommendation
        product_name     = _str(result.get("product_name")),
        product_group    = _str(result.get("product_group")),
        product_type     = _str(result.get("product_type")),
        payment_period   = _str(result.get("payment_period")),
        insurance_period = _str(result.get("insurance_period")),
        payment_cycle    = _str(result.get("payment_cycle")),
        amount           = _int(result.get("amount")),
        monthly_premium  = _int(result.get("monthly_premium")),
        ai_accuracy      = _int(result.get("ai_accuracy")),
        reason           = _str(result.get("reason")),
        coverage         = _list(result.get("coverage")),
        rag_products     = _list(result.get("rag_products")),

        # product_info
        source             = _str(result.get("source")),
        coverage_info      = _list(result.get("coverage")),
        summary            = _str(result.get("summary")),
        special_terms_info = _list(result.get("special_terms")),

        # special_terms
        recommended = _list(result.get("recommended")),
        added       = _list(result.get("added")),
        removed     = _list(result.get("removed")),
        caution     = _str(result.get("caution")),

        # general_qa
        answer = _str(result.get("answer")),

        # 공통 컨텍스트
        current_source = _str(updated_ctx["current_source"]),
        current_terms  = updated_ctx["current_terms"] or [],
        raw_response   = _str(result.get("raw_response")),
    )


@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):
    _clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
