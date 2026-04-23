"""
동양생명 챗봇 DB 세팅 스크립트
실행: python db_setup.py

구성:
1. SQLite  - 설계사 DB, 고객 DB, 가입설계 정보 DB (더미 데이터)
2. ChromaDB - 가입설계 이력 DB (더미 데이터 + 임베딩 적용)

※ 보험 상품 DB는 rag_pipeline.py에서 별도 처리
"""

import sqlite3
import chromadb
import random
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

CHROMA_PATH     = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"

# ─────────────────────────────────────────
# 1. SQLite DB 초기화
# ─────────────────────────────────────────

def setup_sqlite():
    print("\n[1] SQLite DB 세팅 시작...")
    conn = sqlite3.connect("insurance.db")
    cur  = conn.cursor()

    # ── 설계사 테이블
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fc (
        fc_id       TEXT PRIMARY KEY,
        fc_name     TEXT NOT NULL,
        password    TEXT NOT NULL,
        branch      TEXT,
        phone       TEXT,
        created_at  TEXT DEFAULT (datetime('now', 'localtime'))
    )""")

    # ── 고객 테이블
    cur.execute("""
    CREATE TABLE IF NOT EXISTS customer (
        customer_id   TEXT PRIMARY KEY,
        name          TEXT,
        gender        TEXT,
        birth         TEXT,
        phone         TEXT,
        is_virtual    INTEGER DEFAULT 0,
        fc_id         TEXT,
        created_at    TEXT DEFAULT (datetime('now', 'localtime')),
        FOREIGN KEY (fc_id) REFERENCES fc(fc_id)
    )""")

    # ── 가입설계 정보 테이블
    cur.execute("""
    CREATE TABLE IF NOT EXISTS design_info (
        design_no        TEXT PRIMARY KEY,
        customer_id      TEXT,
        fc_id            TEXT,
        product_name     TEXT,
        product_group    TEXT,
        product_type     TEXT,
        payment_period   TEXT,
        insurance_period TEXT,
        payment_cycle    TEXT,
        amount           INTEGER,
        monthly_premium  INTEGER,
        ai_accuracy      INTEGER,
        ai_reason        TEXT,
        status           TEXT DEFAULT '완료',
        created_at       TEXT DEFAULT (datetime('now', 'localtime')),
        FOREIGN KEY (customer_id) REFERENCES customer(customer_id),
        FOREIGN KEY (fc_id)       REFERENCES fc(fc_id)
    )""")

    conn.commit()

    # ── 더미 설계사 데이터
    fc_data = [
        ("FC001", "김철수", "1234", "서울본부", "010-1111-2222"),
        ("FC002", "이영희", "1234", "부산지점", "010-3333-4444"),
        ("FC003", "박민준", "1234", "대구지점", "010-5555-6666"),
        ("FC004", "최지은", "1234", "인천지점", "010-7777-8888"),
        ("FC005", "정수현", "1234", "광주지점", "010-9999-0000"),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO fc (fc_id, fc_name, password, branch, phone)
        VALUES (?, ?, ?, ?, ?)
    """, fc_data)

    # ── 더미 고객 데이터
    names   = ["홍길동", "김민수", "이서연", "박준호", "최유진", "정다은", "강태양", "윤소희", "임재원", "조하늘"]
    genders = ["남성", "여성"]
    customer_data = []
    for i, name in enumerate(names):
        cid    = f"C{str(i+1).zfill(4)}"
        gender = random.choice(genders)
        year   = random.randint(1960, 2000)
        month  = random.randint(1, 12)
        day    = random.randint(1, 28)
        birth  = f"{year}-{month:02d}-{day:02d}"
        fc_id  = random.choice([fc[0] for fc in fc_data])
        customer_data.append((
            cid, name, gender, birth,
            f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
            0, fc_id
        ))

    cur.executemany("""
        INSERT OR IGNORE INTO customer (customer_id, name, gender, birth, phone, is_virtual, fc_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, customer_data)

    conn.commit()
    conn.close()
    print("  ✅ SQLite DB 완료 (설계사 5명, 고객 10명 더미 생성)")


# ─────────────────────────────────────────
# 2. ChromaDB - 가입설계 이력 (더미 + 임베딩)
# ─────────────────────────────────────────

def setup_design_history_chroma():
    print("\n[2] ChromaDB 가입설계 이력 세팅 시작...")

    print(f"  임베딩 모델 로딩: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 기존 컬렉션 삭제 후 재생성 (재실행 시 중복 방지)
    try:
        client.delete_collection("design_history")
    except Exception:
        pass

    collection = client.create_collection(
        name="design_history",
        metadata={"hnsw:space": "cosine"}
    )

    products = [
        "무배당우리WON하는치아보험(갱신형)",
        "무배당우리WON하는암보험",
        "무배당우리WON하는종신보험",
        "무배당우리WON하는건강보험",
        "무배당우리WON하는어린이보험",
    ]
    genders         = ["남성", "여성"]
    payment_periods = ["10년납", "20년납", "전기납"]
    product_types   = ["종신형", "정기형", "저축형"]
    payment_cycles  = ["월납", "분기납", "연납"]

    docs, metas, ids = [], [], []

    for i in range(50):
        gender         = random.choice(genders)
        age            = random.randint(25, 65)
        product        = random.choice(products)
        payment_period = random.choice(payment_periods)
        product_type   = random.choice(product_types)
        payment_cycle  = random.choice(payment_cycles)
        amount         = random.choice([1000, 2000, 3000, 5000])
        premium        = int(amount * random.uniform(0.002, 0.005))
        days_ago       = random.randint(1, 365)
        date           = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        text = (
            f"{gender} {age}세 고객이 {product}에 가입했습니다. "
            f"상품유형은 {product_type}이며, {payment_period} {payment_cycle}로 "
            f"가입금액 {amount}만원, 월 보험료 {premium}원입니다."
        )

        docs.append(text)
        metas.append({
            "gender":         gender,
            "age":            age,
            "product":        product,
            "product_type":   product_type,
            "payment_period": payment_period,
            "payment_cycle":  payment_cycle,
            "amount":         amount,
            "premium":        premium,
            "date":           date
        })
        ids.append(f"history_{i+1}")

    print(f"  임베딩 중... (50건)")
    embeddings = model.encode(
        docs,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=metas,
        ids=ids
    )
    print(f"  ✅ 가입설계 이력 ChromaDB 완료 (더미 50건 + 임베딩 적용)")


# ─────────────────────────────────────────
# 3. 가입설계 완료 시 SQLite + ChromaDB 동시 저장
#    → Streamlit에서 가입설계 완료 버튼 누를 때 호출
# ─────────────────────────────────────────

def save_design(design: dict, model: SentenceTransformer = None):
    """
    가입설계 완료 시 호출.
    SQLite design_info + ChromaDB design_history 동시 저장.

    Parameters:
        design = {
            "design_no":        str,
            "customer_id":      str,
            "fc_id":            str,
            "customer_name":    str,
            "gender":           str,
            "age":              int,
            "product_name":     str,
            "product_group":    str,
            "product_type":     str,
            "payment_period":   str,
            "insurance_period": str,
            "payment_cycle":    str,
            "amount":           int,
            "monthly_premium":  int,
            "ai_accuracy":      int,
            "ai_reason":        str,
        }
    """
    # ── SQLite 저장
    conn = sqlite3.connect("insurance.db")
    cur  = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO design_info (
            design_no, customer_id, fc_id,
            product_name, product_group, product_type,
            payment_period, insurance_period, payment_cycle,
            amount, monthly_premium, ai_accuracy, ai_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        design["design_no"],
        design["customer_id"],
        design["fc_id"],
        design["product_name"],
        design["product_group"],
        design["product_type"],
        design["payment_period"],
        design["insurance_period"],
        design["payment_cycle"],
        design["amount"],
        design["monthly_premium"],
        design["ai_accuracy"],
        design["ai_reason"],
    ))
    conn.commit()
    conn.close()

    # ── ChromaDB 저장 (임베딩 적용)
    text = (
        f"{design['gender']} {design['age']}세 고객이 {design['product_name']}에 가입했습니다. "
        f"상품유형은 {design['product_type']}이며, {design['payment_period']} {design['payment_cycle']}로 "
        f"가입금액 {design['amount']}만원, 월 보험료 {design['monthly_premium']}원입니다."
    )

    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)

    embedding = model.encode(
        [text], normalize_embeddings=True
    ).tolist()

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("design_history")
    collection.add(
        documents=[text],
        embeddings=embedding,
        metadatas=[{
            "design_no":      design["design_no"],
            "gender":         design["gender"],
            "age":            design["age"],
            "product":        design["product_name"],
            "product_type":   design["product_type"],
            "payment_period": design["payment_period"],
            "payment_cycle":  design["payment_cycle"],
            "amount":         design["amount"],
            "premium":        design["monthly_premium"],
            "date":           datetime.now().strftime("%Y-%m-%d"),
        }],
        ids=[f"history_{design['design_no']}"]
    )

    print(f"  ✅ 가입설계 저장 완료 (SQLite + ChromaDB): {design['design_no']}")


# ─────────────────────────────────────────
# 4. 세팅 확인
# ─────────────────────────────────────────

def verify_setup():
    print("\n[3] 세팅 확인 중...")

    conn = sqlite3.connect("insurance.db")
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM fc")
    fc_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM customer")
    customer_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM design_info")
    design_count = cur.fetchone()[0]
    conn.close()

    client  = chromadb.PersistentClient(path=CHROMA_PATH)
    history = client.get_collection("design_history")

    print("\n" + "="*45)
    print("  📊 DB 세팅 완료 요약")
    print("="*45)
    print(f"  [SQLite - insurance.db]")
    print(f"    설계사 DB      : {fc_count}명")
    print(f"    고객 DB        : {customer_count}명")
    print(f"    가입설계 정보  : {design_count}건")
    print(f"  [ChromaDB - ./chroma_db]")
    print(f"    가입설계 이력  : {history.count()}건 (임베딩 적용)")
    print("="*45)
    print("  ✅ 모든 DB 세팅 완료!")
    print("="*45)


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("="*45)
    print("  동양생명 챗봇 DB 세팅 시작")
    print("="*45)

    setup_sqlite()
    setup_design_history_chroma()
    verify_setup()