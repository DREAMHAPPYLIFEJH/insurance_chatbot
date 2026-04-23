import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st

# ─────────────────────────────────────────
# 페이지 설정 (반드시 첫 번째 Streamlit 명령어)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="동양생명 가입설계 챗봇",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import sqlite3
import jwt
import time
import httpx
from datetime import datetime, timedelta
from utils.pdf_generator import generate_pdf

API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api")

def call_chat_api(session_id: str, user_input: str) -> dict:
    """FastAPI /api/chat 호출"""
    resp = httpx.post(
        f"{API_BASE}/chat",
        json={"session_id": session_id, "user_input": user_input},
        timeout=300.0,   # EXAONE 추론 시간 고려
    )
    resp.raise_for_status()
    return resp.json()

def clear_session_api(session_id: str):
    """FastAPI 세션 초기화"""
    try:
        httpx.delete(f"{API_BASE}/session/{session_id}", timeout=5.0)
    except Exception:
        pass

# ─────────────────────────────────────────
# JWT 설정
# ─────────────────────────────────────────
JWT_SECRET       = "dongyang_life_secret_key"
JWT_ALGORITHM    = "HS256"
JWT_EXPIRE_HOURS = 8

def create_token(fc_id, fc_name):
    payload = {
        "fc_id":   fc_id,
        "fc_name": fc_name,
        "exp":     datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat":     datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except:
        return None

def login_with_db(fc_id, password):
    try:
        conn = sqlite3.connect("insurance.db")
        cur  = conn.cursor()
        cur.execute(
            "SELECT fc_id, fc_name, branch FROM fc WHERE fc_id=? AND password=?",
            (fc_id, password)
        )
        row = cur.fetchone()
        conn.close()
        return {"fc_id": row[0], "fc_name": row[1], "branch": row[2]} if row else None
    except Exception as e:
        st.error(f"DB 연결 오류: {e}")
        return None

def check_auth():
    token = st.session_state.get("jwt_token")
    if not token:
        return False
    payload = verify_token(token)
    if not payload:
        st.session_state.clear()
        return False
    st.session_state.fc_id   = payload["fc_id"]
    st.session_state.fc_name = payload["fc_name"]
    return True

def require_auth():
    if not check_auth():
        st.warning("세션이 만료되었습니다. 다시 로그인해주세요.")
        st.session_state.screen = "login"
        st.rerun()

# ─────────────────────────────────────────
# 세션 초기화
# ─────────────────────────────────────────
for key, val in {
    "screen":           "login",
    "jwt_token":        None,
    "fc_id":            "",
    "fc_name":          "",
    "messages":         [],
    "design_data":      None,
    # 대화 컨텍스트
    "current_source":   None,    # 현재 대화 중인 상품 PDF 파일명
    "current_intent":   None,    # 직전 intent
    "current_terms":    [],      # 현재 선택된 특약 목록
    "chat_history":     None,    # ChatMessageHistory (special_terms 전용)
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
* { font-family: 'Noto Sans KR', sans-serif; }

.header {
    background: linear-gradient(135deg, #0d2b5e, #1a4a9e);
    color: white; padding: 18px 24px; border-radius: 12px;
    margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center;
}
.header-left h2 { margin: 0; font-size: 1.2rem; font-weight: 700; }
.header-left p  { margin: 4px 0 0; font-size: 0.8rem; opacity: 0.8; }
.header-right   { font-size: 0.8rem; opacity: 0.85; }

.card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin-bottom: 16px; }
.card-title { font-size: 1rem; font-weight: 700; color: #0d2b5e; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 2px solid #e8edf5; }

/* 챗 말풍선 */
.msg-bot {
    background: #f0f4ff; border-left: 4px solid #1a4a9e;
    border-radius: 0 12px 12px 0; padding: 12px 16px;
    margin-bottom: 10px; font-size: 0.9rem; line-height: 1.6; color: #2c3e50;
    max-width: 85%;
}
.msg-user {
    background: #1a4a9e; color: white;
    border-radius: 12px 0 0 12px; padding: 12px 16px;
    margin-bottom: 10px; font-size: 0.9rem; line-height: 1.6;
    max-width: 85%; margin-left: auto; text-align: right;
}
.msg-wrap-bot  { display: flex; justify-content: flex-start; }
.msg-wrap-user { display: flex; justify-content: flex-end; }

/* 추천 결과 */
.rec-card {
    background: linear-gradient(135deg, #f0f4ff, #e8f0fe);
    border: 1px solid #c5d5f5; border-radius: 12px; padding: 20px; margin-bottom: 12px;
}
.rec-tag { display: inline-block; background: #1a4a9e; color: white; border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; margin: 2px 4px 6px 0; }
.info-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #dde5f5; font-size: 0.88rem; }
.info-label { color: #666; }
.info-value { font-weight: 600; color: #0d2b5e; }

.agree-box { background: #fffbf0; border: 1px solid #ffc107; border-radius: 8px; padding: 12px 16px; font-size: 0.83rem; color: #5a4000; margin-bottom: 12px; }

.success-box { background: linear-gradient(135deg, #e8f5e9, #f1f8e9); border: 1px solid #81c784; border-radius: 12px; padding: 28px; text-align: center; }
.success-icon  { font-size: 3rem; margin-bottom: 10px; }
.success-title { font-size: 1.2rem; font-weight: 700; color: #1b5e20; }
.success-sub   { color: #388e3c; font-size: 0.88rem; margin-top: 6px; }

.stButton > button {
    background: linear-gradient(135deg, #0d2b5e, #1a4a9e);
    color: white !important; border: none; border-radius: 8px;
    padding: 10px 20px; font-weight: 600; width: 100%;
}
.stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 공통 헤더
# ─────────────────────────────────────────
def render_header(title, subtitle=""):
    fc = st.session_state.get("fc_name", "")
    right = f"설계사: {fc}" if fc else ""
    st.markdown(f"""
    <div class="header">
        <div class="header-left">
            <h2>🛡️ {title}</h2>
            {"<p>" + subtitle + "</p>" if subtitle else ""}
        </div>
        <div class="header-right">{right}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 화면 1 — 로그인 / 약관출처(프롬프트) / Multi Agent(약관, ...) / 검색 기능(검색의 범위 감소 - 질의 키워드 추출 & 검색 로직) 
# / ROUTER 기능
# 1. 40대 남성 암보험 가입하고싶어. 월 15만원 이하로.
# 2. 특약 어떤 걸 넣을지를 모르겠어.
# https://smithery.ai/ -> mcp server 다운받아서 구현
# ─────────────────────────────────────────
def screen_login():
    # 유효 토큰 있으면 바로 챗 화면으로
    if check_auth():
        st.session_state.screen = "chat"
        st.rerun()

    render_header("동양생명 가입설계 챗봇", "AI 기반 맞춤형 보험 설계 서비스")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔐 설계사 로그인</div>', unsafe_allow_html=True)

    fc_id = st.text_input("설계사 ID", placeholder="사번을 입력하세요 (예: FC001)")
    fc_pw = st.text_input("비밀번호", type="password", placeholder="비밀번호 (더미: 1234)")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("로그인"):
        if not fc_id or not fc_pw:
            st.error("ID와 비밀번호를 입력해주세요.")
            return
        fc_info = login_with_db(fc_id, fc_pw)
        if fc_info:
            token = create_token(fc_info["fc_id"], fc_info["fc_name"])
            st.session_state.jwt_token      = token
            st.session_state.fc_id          = fc_info["fc_id"]
            st.session_state.fc_name        = fc_info["fc_name"]
            st.session_state.branch         = fc_info.get("branch", "")
            st.session_state.current_source = None
            st.session_state.current_intent = None
            st.session_state.current_terms  = []
            st.session_state.messages  = [{
                "role": "bot",
                "text": f"안녕하세요 {fc_info['fc_name']}님! 👋\n\n고객 상황을 자유롭게 말씀해 주세요.\n\n예시:\n- \"40대 남성, 암이 걱정되고 월 10만원 이내로 들고 싶어요\"\n- \"30대 여자 고객, 어린이 보험 찾는데 예산은 5만원\""}
            ]
            st.session_state.screen = "chat"
            st.rerun()
        else:
            st.error("ID 또는 비밀번호가 올바르지 않습니다.")

# ─────────────────────────────────────────
# 화면 2 — 챗 서비스
# ─────────────────────────────────────────
def screen_chat():
    require_auth()
    render_header("챗 서비스", "고객 상황을 자유롭게 입력하세요")

    # 상단 탭 네비게이션
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("📋 가입설계 화면으로 →"):
            if st.session_state.design_data:
                st.session_state.screen = "design"
                st.rerun()
            else:
                st.warning("먼저 AI 추천을 받아주세요.")
    with col3:
        if st.button("🚪 로그아웃"):
            st.session_state.clear()
            st.rerun()

    # 대화 이력 출력
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "bot":
            rag = msg.get("rag_products", [])
            if not rag:
                text = (msg.get("text") or "").replace(chr(10), "<br>")
                st.markdown(f'<div class="msg-wrap-bot"><div class="msg-bot">🤖 {text}</div></div>', unsafe_allow_html=True)
            if rag:
                rows = ""
                for i, p in enumerate(rag, 1):
                    bar_width = int(p["similarity"] * 100)
                    rows += (
                        f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #e8edf5;font-size:0.82rem;">'
                        f'<span style="min-width:16px;font-weight:700;color:#1a4a9e;">#{i}</span>'
                        f'<span style="flex:1;color:#2c3e50;">{p["product_name"]}</span>'
                        f'<span style="min-width:60px;background:#e8edf5;border-radius:4px;overflow:hidden;">'
                        f'<span style="display:block;width:{bar_width}%;height:8px;background:#1a4a9e;border-radius:4px;"></span>'
                        f'</span>'
                        f'<span style="min-width:42px;color:#1a4a9e;font-weight:600;">{p["similarity"]:.0%}</span>'
                        f'</div>'
                    )
                st.markdown(f"""
                <div style="background:#f7f9ff; border:1px solid #d0daf5; border-radius:10px;
                            padding:12px 16px; margin:4px 0 10px 0; max-width:85%;">
                    <div style="font-size:0.8rem; font-weight:700; color:#0d2b5e; margin-bottom:8px;">
                        🔍 RAG 유사 상품 Top {len(rag)}
                    </div>
                    {rows}
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-wrap-user"><div class="msg-user">{msg["text"]} 👤</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 입력창
    st.markdown('<div class="card">', unsafe_allow_html=True)
    user_input = st.text_area(
        "고객 상황 입력",
        placeholder="예: 45세 남성, 당뇨 있고 암 보장 원함. 월 납입 15만원 이내.",
        height=80,
        label_visibility="collapsed"
    )
    col_a, col_b = st.columns([4, 1])
    with col_b:
        send = st.button("전송 →")
    st.markdown('</div>', unsafe_allow_html=True)

    if send and user_input.strip():
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "text": user_input})

        # 실제 AI 추천 (FastAPI 호출)
        with st.spinner("🤖 AI가 분석 중입니다..."):
            try:
                session_id = st.session_state.fc_id
                design = call_chat_api(session_id, user_input)

                # 서버에서 관리하는 컨텍스트 동기화
                st.session_state.current_source = design.get("current_source")
                st.session_state.current_terms  = design.get("current_terms", [])
                st.session_state.current_intent = design.get("intent")

            except Exception as e:
                st.error(f"AI 추천 오류: {e}")
                st.session_state.messages.append({
                    "role": "bot",
                    "text": "죄송합니다. AI 분석 중 오류가 발생했습니다. 다시 시도해주세요."
                })
                st.rerun()
                return

        st.session_state.design_data = design
        intent = design.get("intent", "")

        # intent별 봇 응답 텍스트 생성
        if intent == "recommendation":
            bot_text = ""  # RAG 카드로 대체
        elif intent == "product_info":
            bot_text = design.get("summary") or design.get("raw_response", "")
        elif intent == "special_terms":
            recommended = design.get("recommended") or []
            added       = design.get("added") or []
            removed     = design.get("removed") or []
            reason      = design.get("reason", "")
            caution     = design.get("caution", "")
            lines = []
            if recommended:
                lines.append(f"📋 최종 특약 목록: {', '.join(recommended)}")
            if added:
                lines.append(f"➕ 추가: {', '.join(added)}")
            if removed:
                lines.append(f"➖ 제거: {', '.join(removed)}")
            if reason:
                lines.append(f"\n💡 {reason}")
            if caution and caution != "없음":
                lines.append(f"⚠️ {caution}")
            bot_text = "\n".join(lines) if lines else design.get("raw_response", "")
        elif intent == "general_qa":
            bot_text = design.get("answer") or design.get("raw_response", "")
        else:
            bot_text = design.get("raw_response", "")

        st.session_state.messages.append({
            "role":         "bot",
            "text":         bot_text,
            "rag_products": design.get("rag_products", []),
        })
        st.rerun()

# ─────────────────────────────────────────
# 화면 3 — 가입설계 화면
# ─────────────────────────────────────────
def screen_design():
    require_auth()
    render_header("가입설계 화면", "AI 추천 결과 확인 및 저장")

    # 상단 네비게이션
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← 챗 서비스로"):
            st.session_state.screen = "chat"
            st.rerun()
    with col3:
        if st.button("🚪 로그아웃"):
            st.session_state.clear()
            st.rerun()

    d = st.session_state.design_data
    if not d:
        st.info("챗 서비스에서 먼저 AI 추천을 받아주세요.")
        return

    # 추천 상품 카드
    st.markdown(f"""
    <div class="rec-card">
        <div style="font-weight:700; color:#0d2b5e; font-size:1rem; margin-bottom:10px">🏆 AI 추천 상품</div>
        <span class="rec-tag">추천 정확도 {d['ai_accuracy']}%</span>
        <span class="rec-tag">{d['product_group']}</span>
        <span class="rec-tag">{d['product_type']}</span>
        <br>
        <div class="info-row"><span class="info-label">상품명</span><span class="info-value">{d['product_name']}</span></div>
        <div class="info-row"><span class="info-label">납입기간</span><span class="info-value">{d['payment_period']}</span></div>
        <div class="info-row"><span class="info-label">보험기간</span><span class="info-value">{d['insurance_period']}</span></div>
        <div class="info-row"><span class="info-label">납입주기</span><span class="info-value">{d['payment_cycle']}</span></div>
        <div class="info-row"><span class="info-label">가입금액</span><span class="info-value">{d['amount']:,}만원</span></div>
        <div class="info-row"><span class="info-label">예상 월납보험료</span><span class="info-value" style="color:#e53e3e">{d['monthly_premium']:,}원</span></div>
    </div>""", unsafe_allow_html=True)

    # 추천 근거
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📊 AI 추천 근거</div>
        <div class="msg-bot">🤖 {d['reason']}</div>
        <b>주요 보장 내역</b><br><br>
        {"".join([f'<span class="rec-tag">✓ {item}</span>' for item in d['coverage']])}
    </div>""", unsafe_allow_html=True)

    # 고객 정보 입력 (가입설계 저장용)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👤 고객 정보 확인</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        customer_name = st.text_input("고객 이름", placeholder="홍길동")
        gender        = st.selectbox("성별", ["남성", "여성"])
    with col2:
        birth         = st.text_input("생년월일", placeholder="1980-01-01")
        customer_id   = st.text_input("고객 ID (없으면 자동 생성)", placeholder="C0001")

    st.markdown('</div>', unsafe_allow_html=True)

    # 동의 및 저장
    st.markdown('<div class="agree-box">⚠️ 위 추천 결과는 AI 분석 기반이며, 실제 보험료는 심사 결과에 따라 달라질 수 있습니다.</div>', unsafe_allow_html=True)

    agree = st.checkbox("고객이 위 내용을 확인하고 동의합니다.")

    if "design_saved" not in st.session_state:
        st.session_state.design_saved = False

    if not st.session_state.design_saved:
        if st.button("✅ 가입설계 저장"):
            if not agree:
                st.error("고객 동의 확인이 필요합니다.")
            elif not birth:
                st.error("고객 생년월일을 입력해주세요.")
            else:
                design_no = f"DY-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                cid = customer_id.strip() if customer_id.strip() else f"C-{datetime.now().strftime('%H%M%S')}"

                with st.spinner("💾 DB에 저장 중..."):
                    # SQLite 저장 (실제: db_setup.save_design() 호출)
                    try:
                        conn = sqlite3.connect("insurance.db")
                        cur  = conn.cursor()

                        # 고객 저장
                        cur.execute("""
                            INSERT OR IGNORE INTO customer
                            (customer_id, name, gender, birth, is_virtual, fc_id)
                            VALUES (?, ?, ?, ?, 0, ?)
                        """, (cid, customer_name or "미입력", gender, birth, st.session_state.fc_id))

                        # 가입설계 정보 저장
                        cur.execute("""
                            INSERT OR IGNORE INTO design_info
                            (design_no, customer_id, fc_id, product_name, product_group,
                             product_type, payment_period, insurance_period, payment_cycle,
                             amount, monthly_premium, ai_accuracy, ai_reason)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            design_no, cid, st.session_state.fc_id,
                            d["product_name"], d["product_group"], d["product_type"],
                            d["payment_period"], d["insurance_period"], d["payment_cycle"],
                            d["amount"], d["monthly_premium"], d["ai_accuracy"], d["reason"]
                        ))
                        conn.commit()
                        conn.close()
                        time.sleep(0.5)
                    except Exception as e:
                        st.error(f"저장 실패: {e}")
                        return

                st.session_state.design_no      = design_no
                st.session_state.design_saved   = True
                st.session_state.customer_info  = {
                    "name":   customer_name or "미입력",
                    "birth":  birth,
                    "gender": gender,
                }
                st.rerun()
    else:
        # 완료 화면
        design_no = st.session_state.get("design_no", "")
        now       = datetime.now().strftime("%Y-%m-%d %H:%M")

        st.markdown(f"""
        <div class="success-box">
            <div class="success-icon">✅</div>
            <div class="success-title">가입설계가 완료되었습니다!</div>
            <div class="success-sub">설계번호: {design_no}</div>
            <div class="success-sub">{now} 저장 완료</div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            pdf_bytes = generate_pdf(
                design_no     = design_no,
                design_data   = st.session_state.design_data,
                customer_info = st.session_state.get("customer_info", {}),
                fc_name       = st.session_state.get("fc_name", ""),
            )
            st.download_button(
                label     = "📥 PDF 다운로드",
                data      = pdf_bytes,
                file_name = f"가입설계_{design_no}.pdf",
                mime      = "application/pdf",
            )
        with col2:
            if st.button("🔄 새 고객 설계"):
                # 로그인 유지, 설계 데이터 초기화
                token, fc_id, fc_name, branch = (
                    st.session_state.jwt_token,
                    st.session_state.fc_id,
                    st.session_state.fc_name,
                    st.session_state.get("branch", "")
                )
                clear_session_api(fc_id)   # 서버 세션 초기화
                st.session_state.clear()
                st.session_state.update({
                    "jwt_token":      token,
                    "fc_id":          fc_id,
                    "fc_name":        fc_name,
                    "branch":         branch,
                    "screen":         "chat",
                    "current_source": None,
                    "current_intent": None,
                    "current_terms":  [],
                    "messages": [{
                        "role": "bot",
                        "text": "새 고객 상담을 시작합니다! 고객 상황을 자유롭게 말씀해 주세요. 😊"
                    }]
                })
                st.rerun()

# ─────────────────────────────────────────
# 라우팅
# ─────────────────────────────────────────
screen = st.session_state.get("screen", "login")

if   screen == "login":  screen_login()
elif screen == "chat":   screen_chat()
elif screen == "design": screen_design()