# 📋 보험 챗봇 프로젝트 지침

## 🤖 코딩 행동 원칙 (Andrej Karpathy 가이드라인)

코딩 전에 생각하고, 단순하게 만들고, 요청한 것만 건드려라.

**1. 코딩 전에 먼저 생각**
- 가정은 명시적으로 밝혀라. 불확실하면 물어봐라.
- 여러 해석이 가능하면 제시하고 선택을 맡겨라.
- 더 단순한 방법이 있으면 먼저 말해라.

**2. 단순함 우선**
- 요청한 것만 만들어라. 추측성 기능 추가 금지.
- 한 번만 쓸 코드에 추상화 금지.
- 200줄로 쓴 걸 50줄로 쓸 수 있으면 다시 써라.

**3. 외과적 변경**
- 요청한 부분만 건드려라. 관련 없는 코드 개선 금지.
- 내 변경으로 생긴 미사용 import/변수/함수만 정리해라.
- 모든 변경 줄은 요청에 직접 연결되어야 한다.

**4. 목표 기반 실행**
- 성공 기준을 먼저 정의해라.
- 여러 단계 작업은 계획을 먼저 제시해라.

---

## 🎯 포트폴리오/이력서 문서화 목적

이 프로젝트의 기술 스택과 핵심 내용을 분석하여
**포트폴리오 및 이력서에 쓸 수 있는 문서**를 생성한다.

---

## 📁 1단계: 프로젝트 구조 파악

아래 순서대로 코드베이스를 탐색해라.

```
1. 루트 디렉토리 구조 확인
2. requirements.txt 확인 → 의존성 전부 파악
3. 메인 진입점 파일 (api_server.py, chatbot_ui.py) 읽기
4. router.py → 라우팅 로직 파악
5. agents/ → 각 에이전트 역할 파악
6. ai_engine.py → LLM/RAG 핵심 로직 파악
7. routes/ → API 엔드포인트 파악
8. scripts/ → 데이터 파이프라인 파악
```

---

## 🔍 2단계: 분석해야 할 항목

### A. 기술 스택 추출
- **언어**: Python 버전
- **프레임워크**: FastAPI + 버전
- **UI**: Streamlit + 버전
- **LLM**: 사용한 로컬 모델명 (EXAONE 등), 추론 방식
- **라우터 모델**: fine-tuned 분류 모델명 (klue/roberta-base 등)
- **RAG**: 임베딩 모델, 벡터 DB (ChromaDB 등)
- **DB**: SQLite (설계사/고객 데이터), ChromaDB (보험 상품 벡터)
- **인프라/배포**: 로컬 GPU 서버, Docker 여부 등
- **기타**: JWT 인증, PDF 파싱 라이브러리 등

### B. 핵심 기능 파악
- 사용자 입력 → 라우팅 파이프라인이 어떻게 구성되어 있는지 (키워드 매칭 + 모델 분류)
- RAG 검색 결과가 LLM 프롬프트에 어떻게 연결되는지
- API 엔드포인트 목록 및 역할
- 에이전트별 역할 분담 (recommendation / product_info / special_terms / general_qa)
- 팀원 간 역할 분담 흔적 (디렉토리 구조, 주석 등으로 유추)

### C. 아키텍처 파악
- 전체 서비스 흐름 (설계사 입력 → 라우터 → 에이전트 → LLM → 응답)
- 레이어 구조 (UI → API → Router → Agent → AI Engine)
- RAG 파이프라인 구조 (PDF → 청킹 → 임베딩 → ChromaDB)
- 세션 관리 구조 (설계사별 대화 맥락 유지)

---

## 📝 3단계: 출력 형식

분석이 끝나면 아래 형식으로 **`PROJECT_SUMMARY.md`** 파일을 생성해라.

---

```markdown
# [프로젝트명] — 포트폴리오 정리

## 한 줄 소개
> (예: 보험 설계사가 고객에게 맞는 상품을 추천·설명할 수 있도록 돕는 AI 가입설계 챗봇)

## 프로젝트 개요
| 항목 | 내용 |
|------|------|
| 기간 | (코드 커밋 날짜 기준으로 추정 or 직접 기재) |
| 팀 규모 | (직접 기재) |
| 담당 역할 | (코드에서 유추 or TODO: 직접 기재) |
| 유형 | 팀 프로젝트 |

## 기술 스택
### Backend
- Python x.x / FastAPI x.x
- Streamlit x.x (설계사 UI)
- SQLite (설계사·고객·가입설계 이력 관리)
- JWT 인증

### AI / ML
- LLM: (모델명 — 로컬 추론, bfloat16)
- 라우터 모델: (fine-tuned 분류 모델명)
- 임베딩: (모델명)
- 벡터 DB: ChromaDB

### RAG Pipeline
- PDF 파싱: pdfplumber (표 → 자연어 변환 포함)
- 청킹: Q&A / 항목기호 / 문장 분리 전략
- 검색: 코사인 유사도 기반 시맨틱 검색

### Infra / DevOps
- (배포 환경, Docker 여부 등)

## 핵심 기능
1. **하이브리드 라우팅**: 키워드 매칭(1차) + fine-tuned klue/roberta-base(2차)로 의도 분류
2. **상품 추천**: RAG로 유사 상품 검색 후 LLM이 고객 상황에 맞는 상품 추천
3. **상품 정보 조회**: PDF 전체 chunk 수집 후 LLM이 보장내용 요약
4. **특약 상담**: 다중 턴 대화로 특약 추가/제거 누적 관리
5. **일반 질의**: 보험 관련 자유 질문 응답
6. **가입설계 저장**: 추천 결과를 DB에 저장 및 PDF 출력

## 시스템 아키텍처
[설계사 입력 (Streamlit UI)]
    ↓
[FastAPI /api/chat]
    ↓
[Router] → 1단계: 키워드 매칭 / 2단계: klue/roberta-base 분류
    ↓
[Agent 분기]
├── recommendation  → RAG(상품+이력) + EXAONE 추론 → 상품 추천
├── product_info    → PDF 전체 chunk + EXAONE 추론 → 상품 요약
├── special_terms   → RAG(특약) + EXAONE 추론 → 특약 구성
└── general_qa      → RAG + EXAONE 추론 → 자유 질의 응답
    ↓
[세션 컨텍스트 업데이트 (current_source / current_terms)]

## 주요 기술적 도전 & 해결
- **도전 1**: LLM API 비용 없이 로컬 추론 → EXAONE 3.5 2.4B 직접 로드, bfloat16 + Flash Attention 2로 메모리 최적화
- **도전 2**: 다양한 의도의 질문 분류 → 키워드 매칭 + klue/roberta-base 파인튜닝 하이브리드 라우터 구현
- **도전 3**: PDF 표 데이터 RAG 처리 → pdfplumber로 표를 자연어 문장으로 변환 후 임베딩
- **도전 4**: (TODO: 직접 확인 필요 — 추가 도전 있으면 기재)

## 이력서용 한 줄 성과 (bullet point)
- FastAPI 기반 보험 가입설계 챗봇 백엔드 설계 및 구현
- EXAONE 3.5 2.4B 로컬 LLM + ChromaDB RAG 파이프라인 구축으로 외부 API 의존성 제거
- klue/roberta-base 파인튜닝으로 보험 도메인 특화 의도 분류 라우터 구현
- pdfplumber 기반 보험 PDF 파싱 및 벡터 DB 구축 자동화 파이프라인 개발
- (TODO: 수치 성과 있으면 추가 — 예: 라우팅 정확도 xx%, 응답 시간 등)
```

---

## ⚠️ 주의사항

- **모르는 부분은 `TODO: 직접 확인 필요`로 표시**하고 넘어가라. 추측으로 채우지 마라.
- 버전 정보는 requirements.txt에서 **정확히** 가져와라.
- LLM(EXAONE)과 라우터 분류 모델(klue/roberta-base)이 **각각 다른 역할**임을 명확히 구분해서 적어라.
- 팀 프로젝트이므로 **본인 담당 부분**을 특정할 수 있으면 별도로 표시해라 (`[내 담당]` 태그 등).
- 분석 완료 후 요약을 먼저 나에게 보고하고, 확인 후 파일 생성해라.

---

## ✅ 실행 순서 요약

```
1. requirements.txt로 기술 스택 버전 확인
2. api_server.py → 엔드포인트 목록 정리
3. router.py → 라우팅 방식 파악
4. agents/ 각 파일 → 에이전트별 역할 파악
5. ai_engine.py → LLM/RAG 핵심 로직 파악
6. scripts/rag_pipeline.py → 데이터 파이프라인 파악
7. chatbot_ui.py → UI 구조 및 기능 파악
8. PROJECT_SUMMARY.md 초안 작성
9. 나에게 검토 요청
```
