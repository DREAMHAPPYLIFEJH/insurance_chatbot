# 동양생명 가입설계 챗봇 — 포트폴리오 정리

## 한 줄 소개
> 보험 설계사가 고객 상황을 자연어로 입력하면, AI가 적합한 보험 상품을 추천하고 가입설계까지 자동화하는 내부 업무 지원 챗봇

## 프로젝트 개요
| 항목 | 내용 |
|------|------|
| 기간 | TODO: 직접 기재 |
| 팀 규모 | TODO: 직접 기재 |
| 담당 역할 | TODO: 직접 기재 |
| 유형 | TODO: 직접 기재 |

## 기술 스택

### Backend
- Python / FastAPI
- Streamlit (설계사용 웹 UI)
- httpx (FastAPI ↔ Streamlit 통신)
- JWT (PyJWT) — 설계사 로그인 인증

### AI / ML
- **LLM**: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct (HuggingFace Transformers, 로컬 GPU, bfloat16, Flash Attention 2 우선 시도 후 fallback)
- **임베딩**: BAAI/bge-m3 (SentenceTransformer, normalize_embeddings=True)
- **라우터 분류 모델**: klue/roberta-base 파인튜닝 (4-class, MAX_LEN=64, 클래스 가중치 손실)
- **대화 이력**: langchain-community `ChatMessageHistory` (RunnableWithMessageHistory 미사용 — special_terms 에이전트가 messages 리스트를 직접 구성해 EXAONE chat template에 주입)

### Database
- **ChromaDB** — 보험 상품 PDF 벡터 DB (`insurance_products`), 가입설계 이력 벡터 DB (`design_history`)
- **SQLite** — 설계사(fc), 고객(customer), 가입설계 정보(design_info) 관계형 테이블

### Data Processing
- pdfplumber — 보험 약관 PDF 파싱 (표/텍스트 분리 처리)

### Infra / DevOps
- TODO: 직접 기재 (로컬 실행 기준: uvicorn + streamlit 별도 실행)
- GPU: CUDA (torch bfloat16, device_map="auto")

## 핵심 기능

### 1. 하이브리드 라우터
사용자 입력을 **키워드 매칭(1단계) → klue/roberta-base 파인튜닝 모델(2단계)** 순으로 4가지 intent로 분류.
키워드로 즉시 판별 가능한 경우 LLM 호출 없이 처리하여 응답 속도 최적화.

| Intent | 역할 |
|--------|------|
| `recommendation` | 고객 맞춤 상품 추천 |
| `product_info` | 특정 상품 보장내용 조회 |
| `special_terms` | 특약 구성 멀티턴 상담 |
| `general_qa` | 보험 일반 지식 질의응답 |

### 2. RAG 기반 상품 추천
보험 약관 PDF → pdfplumber 파싱 → 의미단위 청킹 → BAAI/bge-m3 임베딩 → ChromaDB 저장.
추천 요청 시 유사 상품 top15 검색 → source 기준 중복 제거로 고유 상품 3개 확보 → **1등 상품은 전체 chunk(1000자 cut), 2~3등은 매칭 chunk 400자만** 수집 → EXAONE 3.5 프롬프트 → 구조화된 추천 결과 파싱.
응답 시간 단축을 위한 비대칭 컨텍스트 구성.

### 3. 특약 멀티턴 상담
서버 메모리에 `session_id`별 `ChatMessageHistory`를 보관하고, 매 턴 직접 messages 리스트를 만들어 EXAONE `apply_chat_template`에 주입.
응답에서 `추천특약:` 라인을 파싱해 직전 `current_terms`와 diff(added / removed)를 계산.
세션별 현재 특약 목록·current_source는 `api_server._sessions` 딕셔너리에서 관리.

### 4. 설계사 인증 및 가입설계 저장
JWT 기반 설계사 로그인 (8시간 유효), SQLite에 고객 정보 + AI 추천 결과를 가입설계 번호(`DY-YYYYMMDDHHMMSS`)와 함께 저장.

## 시스템 아키텍처

```
[설계사 입력 (Streamlit UI)]
    ↓ HTTP POST /api/chat
[FastAPI 서버]
    ↓
[Router] — 키워드 매칭 → 즉시 반환
         — 키워드 미감지 → klue/roberta-base 분류
    ↓
[Agent 분기]
    ├─ RecommendationAgent  (프롬프트 ~2500자)
    │      ↓ RAG top15 → source 중복 제거 3개
    │      ↓ 1등=전체 chunk(1000자), 2·3등=매칭 chunk 400자
    │      ↓ + 가입 이력 RAG top3 (design_history)
    │      ↓ EXAONE 3.5 추론 → 11개 필드 구조화 파싱
    │
    ├─ ProductInfoAgent     (프롬프트 ~3000자)
    │      ↓ top1으로 source 특정 → 해당 PDF 전체 chunk 3000자 cut
    │      ↓ EXAONE 정리 → 8개 필드 파싱
    │
    ├─ SpecialTermsAgent    (프롬프트 ~2000자)
    │      ↓ source 필터 RAG 5개 + ChatMessageHistory messages
    │      ↓ EXAONE chat_template 직접 호출
    │      ↓ 추천특약 diff(added/removed) 계산
    │
    └─ GeneralQAAgent       (프롬프트 ~1500자)
           ↓ RAG top3 → EXAONE 답변
    ↓
[Streamlit UI 렌더링]
    ↓ (추천 확정 시)
[SQLite 가입설계 저장]
```

## RAG 파이프라인 상세

```
보험 약관 PDF (pdfs/)
    ↓ pdfplumber
표 → 자연어 변환 / 텍스트 → 의미단위 청킹
    ↓ BAAI/bge-m3 임베딩
ChromaDB (insurance_products collection)
    — 메타데이터: source(PDF명), page, chunk_index, category
```

## 알려진 한계 / 후속 과제 (problem.md 기반)

- **응답 시간 30~60s**: recommendation 2500자, product_info 3000자, special_terms 2000자 프롬프트가 EXAONE 추론 병목 → 청크 정책·요약 캐시 검토 필요.
- **특약 정보 영속화 부재**: SQLite `design_info`에 특약 컬럼이 없어 가입설계 저장 시 `current_terms`가 유실됨.
- **PDF 선택특약 자동 추출 미흡**: 현재는 의미단위 청킹 + 카테고리 키워드 태깅만 사용 — 구조 인식형 RAG / 계층형 chunking / section_id 메타데이터 등 검토 중.
- **LangChain 활용 최소화**: `ChatMessageHistory`만 사용하고 Runnable·Tool·SQL agent 등은 미적용.

## 이력서용 한 줄 성과

- FastAPI + Streamlit 기반 보험 설계사용 AI 챗봇 설계 및 구현
- 키워드 매칭 + klue/roberta-base 파인튜닝 하이브리드 라우터로 4가지 intent 자동 분류
- EXAONE 3.5 2.4B 로컬 LLM + ChromaDB RAG 파이프라인으로 보험 상품 추천 시스템 구축
- pdfplumber 기반 보험 약관 PDF 파싱(표→자연어 변환) 및 의미단위 청킹 → 벡터 DB 구축
- 세션별 `ChatMessageHistory` + EXAONE chat template 조합으로 특약 멀티턴 상담 기능 구현
- JWT 인증 + SQLite 가입설계 이력 저장으로 설계사 업무 흐름 전체 자동화
