"""
RAG Pipeline - 보험 PDF 다중 문서 처리
아키텍처:
  1. 텍스트 추출 (pdfplumber)
  2. 청킹 (Q+A / 용어 / 표기호(-) 기반 의미단위)
     - 일반 텍스트 → 의미단위로 자르기
     - 표 데이터 → 자연어 문장으로 변환 후 자르기
  3. 메타데이터 태깅
  4. 임베딩 (BAAI/bge-m3)
  5. ChromaDB 저장
"""

import re
import csv
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
PDF_DIR         = "./pdfs"           # 크롤링한 PDF 폴더
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "insurance_products"
EMBEDDING_MODEL = "BAAI/bge-m3"


# ─────────────────────────────────────────────
# STEP 1: 텍스트 추출 (pdfplumber)
# ─────────────────────────────────────────────
def extract_from_pdf(pdf_path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            table_bboxes = []
            for table_obj in page.find_tables():
                table_bboxes.append(table_obj.bbox)

            # 표 → 자연어 변환
            for table_obj in page.find_tables():
                rows = table_obj.extract()
                if not rows:
                    continue
                nl_sentences = table_to_natural_language(rows)
                if nl_sentences:
                    pages.append({
                        "page": page_num,
                        "type": "table",
                        "content": nl_sentences,
                    })

            # 표 영역 제외한 순수 텍스트 추출
            if table_bboxes:
                text_page = page
                for bbox in table_bboxes:
                    try:
                        text_page = text_page.outside_bbox(bbox)
                    except Exception:
                        pass
                text = text_page.extract_text() or ""
            else:
                text = page.extract_text() or ""

            text = text.strip()
            if text:
                pages.append({
                    "page": page_num,
                    "type": "text",
                    "content": text,
                })

    return pages


def table_to_natural_language(rows: list[list]) -> str:
    if not rows:
        return ""
    headers = [str(h).strip() if h else "" for h in rows[0]]
    sentences = []
    prev_row = [""] * len(headers)
    for row in rows[1:]:
        filled = []
        for i, cell in enumerate(row):
            cell_str = str(cell).strip() if cell else ""
            filled.append(cell_str if cell_str else prev_row[i])
        prev_row = filled
        parts = []
        for header, cell_str in zip(headers, filled):
            if cell_str:
                parts.append(f"{header}: {cell_str}" if header else cell_str)
        if parts:
            sentences.append(", ".join(parts) + ".")
    return " ".join(sentences)


# ─────────────────────────────────────────────
# STEP 2: 청킹 (의미단위)
# ─────────────────────────────────────────────
def _split_by_bullets(text: str) -> list[str]:
    text = re.sub(r'(?<!\n)■', '\n■', text)
    lines = text.split('\n')
    chunks, current = [], []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^[-•·※■]\s*\S', stripped) or re.match(r'^\d+[.)]\s+', stripped):
            if current:
                joined = ' '.join(current).strip()
                if joined:
                    chunks.append(joined)
            current = [stripped]
        else:
            current.append(stripped)
    if current:
        joined = ' '.join(current).strip()
        if joined:
            chunks.append(joined)
    return chunks if len(chunks) > 1 else [text.strip()]


def chunk_text(text: str) -> list[str]:
    chunks = []

    # 1) Q&A 패턴
    qa_pattern = re.compile(r'(?=(?:Q\.|질문|Q\s*:))', re.IGNORECASE)
    qa_parts = qa_pattern.split(text)
    if len(qa_parts) > 1:
        for part in qa_parts:
            part = part.strip()
            if not part or len(part) <= 10:
                continue
            sub = _split_by_bullets(part)
            chunks.extend(sub)
        return chunks

    # 2) 항목 기호(-■) 분리
    text = re.sub(r'(?<!\n)■', '\n■', text)
    lines = text.split('\n')
    current_chunk = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^[-•·※■]\s*\S', stripped) or re.match(r'^\d+[.)]\s+', stripped):
            if current_chunk:
                joined = ' '.join(current_chunk).strip()
                if joined:
                    chunks.append(joined)
            current_chunk = [stripped]
        else:
            current_chunk.append(stripped)

    if current_chunk:
        joined = ' '.join(current_chunk).strip()
        if joined:
            chunks.append(joined)

    if len(chunks) <= 1:
        chunks = split_by_sentence(text)

    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 10:
            continue
        if len(chunk) > 800:
            result.extend(split_long_chunk(chunk, max_len=500, overlap=50))
        else:
            result.append(chunk)

    return result


def split_by_sentence(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_long_chunk(text: str, max_len: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        chunks.append(text[start:end].strip())
        start += max_len - overlap
    return [c for c in chunks if c]


# ─────────────────────────────────────────────
# STEP 3: 메타데이터 태깅
# ─────────────────────────────────────────────
def tag_metadata(chunk: str, page: int, chunk_type: str, chunk_index: int, source: str) -> dict:
    CATEGORY_KEYWORDS = {
        "보장내용":  ["보장", "지급", "보험금", "급여", "치료비", "수술"],
        "보험료":    ["보험료", "납입", "월납", "연납", "할인"],
        "가입조건":  ["가입", "나이", "연령", "자격", "조건", "심사"],
        "면부책":    ["면책", "부책", "지급제한", "제외", "면제"],
        "갱신조건":  ["갱신", "갱신형", "갱신주기", "자동갱신"],
        "약관":      ["약관", "특약", "주계약", "특별약관"],
        "치아":      ["치아", "충치", "임플란트", "크라운", "보철", "스케일링"],
    }

    category = "일반"
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in chunk for kw in keywords):
            category = cat
            break

    return {
        "page":        page,
        "chunk_type":  chunk_type,
        "chunk_index": chunk_index,
        "category":    category,
        "char_count":  len(chunk),
        "source":      source,       # ← PDF 파일명으로 상품 구분
    }


# ─────────────────────────────────────────────
# STEP 4 & 5: 임베딩 + ChromaDB 저장
# ─────────────────────────────────────────────
def embed_and_store(chunks_with_meta: list[dict], collection):
    texts     = [item["text"]     for item in chunks_with_meta]
    metadatas = [item["metadata"] for item in chunks_with_meta]

    # 기존 저장된 chunk_index와 충돌 방지를 위해 전체 카운트 기반 ID 생성
    existing_count = collection.count()
    ids = [f"chunk_{existing_count + i}" for i in range(len(texts))]

    print(f"    임베딩 중... ({len(texts)}개)")
    model      = embed_and_store._model
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )


# ─────────────────────────────────────────────
# 단일 PDF 처리
# ─────────────────────────────────────────────
def process_pdf(pdf_path: Path, collection) -> int:
    source = pdf_path.name

    # STEP 1: 추출
    try:
        pages = extract_from_pdf(str(pdf_path))
    except Exception as e:
        print(f"  ⚠️  추출 실패 ({source}): {e}")
        return 0

    # STEP 2 + 3: 청킹 & 태깅
    chunks_with_meta = []
    chunk_index = 0

    for section in pages:
        text_chunks = chunk_text(section["content"])
        for chunk_text_item in text_chunks:
            meta = tag_metadata(
                chunk=chunk_text_item,
                page=section["page"],
                chunk_type=section["type"],
                chunk_index=chunk_index,
                source=source,
            )
            chunks_with_meta.append({"text": chunk_text_item, "metadata": meta})
            chunk_index += 1

    # 짧은 청크 병합
    merged = []
    for item in chunks_with_meta:
        if merged and len(item["text"]) < 30:
            merged[-1]["text"] += " " + item["text"]
            merged[-1]["metadata"]["char_count"] = len(merged[-1]["text"])
        else:
            merged.append(item)

    if not merged:
        print(f"  ⚠️  청크 없음 ({source})")
        return 0

    # STEP 4 & 5: 임베딩 + 저장
    embed_and_store(merged, collection)
    return len(merged)


# ─────────────────────────────────────────────
# 검색 테스트
# ─────────────────────────────────────────────
def search(query: str, n_results: int = 5):
    model  = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(
        [query], normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\n검색어: '{query}'\n{'─'*60}")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"[{i+1}] 유사도: {1-dist:.4f} | 상품: {meta['source']} | 페이지: {meta['page']} | 카테고리: {meta['category']}")
        print(f"     {doc[:150]}...")
        print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("RAG 파이프라인 시작 (다중 PDF)")
    print("=" * 60)

    # PDF 목록 확인
    pdf_dir  = Path(PDF_DIR)
    pdf_list = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_list:
        raise FileNotFoundError(f"PDF 파일이 없습니다: {pdf_dir.resolve()}")

    print(f"\n총 {len(pdf_list)}개 PDF 발견\n")

    # 임베딩 모델 1회만 로드 (전체 공유)
    print(f"[임베딩 모델 로딩] {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embed_and_store._model = model  # 함수에 모델 주입

    # ChromaDB 초기화 (기존 컬렉션 삭제 후 재생성)
    print(f"[ChromaDB 초기화] {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  기존 컬렉션 삭제 완료")
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # CSV 저장용
    csv_path = "./chunks_preview.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "chunk_index", "page", "chunk_type", "category", "char_count", "source", "text"
    ])
    writer.writeheader()

    # PDF 순차 처리
    total_chunks = 0
    failed       = []

    for i, pdf_path in enumerate(pdf_list, 1):
        print(f"\n[{i}/{len(pdf_list)}] {pdf_path.name}")
        count = process_pdf(pdf_path, collection)
        total_chunks += count
        print(f"  → {count}개 청크 저장")

        if count == 0:
            failed.append(pdf_path.name)

    csv_file.close()

    # 결과 요약
    print("\n" + "=" * 60)
    print("  처리 완료 요약") 
    print("=" * 60)
    print(f"  처리된 PDF  : {len(pdf_list) - len(failed)}개")
    print(f"  실패한 PDF  : {len(failed)}개 {failed if failed else ''}")
    print(f"  총 청크 수  : {total_chunks}개")
    print(f"  ChromaDB    : {CHROMA_PATH}/{COLLECTION_NAME}")
    print("=" * 60)

    # 검색 테스트
    print("\n[검색 테스트]")
    search("임플란트 보장 금액")
    search("보험료 납입 면제 조건")


if __name__ == "__main__":
    main()