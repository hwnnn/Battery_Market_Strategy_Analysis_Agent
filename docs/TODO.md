# 개발 Todo List — 배터리 시장 전략 분석 Agent

> Updated: 2026-07-06

---

## Phase 0. 프로젝트 세팅

- [x] **P0-1** 디렉터리 구조 생성
  ```
  Battery_Market_Strategy_Analysis_Agent/
  ├── data/                  # PDF 문서 (최대 100p)
  ├── agents/                # Agent 모듈
  ├── prompts/               # 프롬프트 템플릿
  ├── outputs/               # 분석 결과 및 보고서
  ├── app.py                 # 실행 진입점
  ├── requirements.txt
  └── README.md
  ```
- [x] **P0-2** `requirements.txt` 작성
  - langgraph, langchain, langchain-openai, langchain-community
  - faiss-cpu
  - sentence-transformers (BAAI/bge-m3)
  - pymupdf (fitz)
  - tavily-python
  - python-dotenv
- [x] **P0-3** `.env.example` 작성 (OPENAI_API_KEY, TAVILY_API_KEY)
- [x] **P0-4** `pip install -r requirements.txt` 실행 및 환경 검증
- [x] **P0-5** Git 초기화 및 `.gitignore` 설정 (`.env`, `data/*.pdf`, `outputs/` 제외)

---

## Phase 1. 데이터 준비 (RAG 문서)

- [x] **P1-1** PDF 문서 수집 (총 100페이지 이내 준수)
  - LGES 사업보고서 (최신)
  - CATL Annual Report (최신)
  - IEA Global EV Outlook 2024/2025
  - 국내 배터리 산업 보고서 (KEIT, KIET 등)
- [x] **P1-2** 총 페이지 수 확인 및 100p 초과 시 우선순위 기준으로 문서 정제
  - 현재 로컬 `data/` 기준 4개 PDF, 총 83페이지
- [x] **P1-3** `data/` 디렉터리에 PDF 배치

---

## Phase 2. 문서 로더 및 벡터 DB 구축

- [x] **P2-1** `agents/document_loader.py` 구현
  - PyMuPDF로 PDF 파싱
  - 페이지 메타데이터(파일명, 페이지 번호) 포함
- [x] **P2-2** 텍스트 청킹 구현
  - `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)`
- [x] **P2-3** 오픈소스 임베딩 설정
  - `HuggingFaceEmbeddings(model_name="BAAI/bge-m3")`
- [x] **P2-4** FAISS 인덱스 생성 및 로컬 저장 (`vectorstore/`)
- [x] **P2-5** 벡터 DB 로드 유틸리티 함수 작성
- [x] **P2-5a** FAISS 캐시 manifest 검증 추가
  - PDF 파일 목록·크기·수정시각 또는 청킹/임베딩 설정이 바뀌면 인덱스 재생성
- [x] **P2-6** 검색 품질 평가 스크립트 작성
  - `eval/eval_ir.py`, `eval/eval_ragas.py`로 Hit Rate@5, MRR, RAGAS 측정

---

## Phase 3. 공통 유틸리티

- [x] **P3-1** `agents/llm_config.py` — LLM 클라이언트 초기화 (GPT-4o-mini)
- [x] **P3-2** `agents/web_search.py` — Tavily 검색 래퍼 구현
  - **확증 편향 방지**: 쿼리 1개 입력 시 긍정/비판/중립 3개 쿼리 자동 생성
  - 3방향 검색을 병렬 실행하고, 결과와 웹 출처 메타데이터를 관점별로 구분해 반환
- [x] **P3-3** `agents/state.py` — `BatteryAnalysisState` TypedDict 정의
- [x] **P3-4** `prompts/` 디렉터리에 각 Agent별 프롬프트 템플릿 파일 작성
  - `prompts/market_research.txt`
  - `prompts/company_analysis.txt`
  - `prompts/comparison.txt`
  - `prompts/report_generator.txt`

---

## Phase 4. RAG Retrieve Tool 구현

- [x] **P4-1** `agents/rag_tool.py` 기본 구조 작성
- [x] **P4-2** `retrieve` 구현 — FAISS에서 관련 문서 검색 (top-k=5)
- [x] **P4-3** plain RAG 기본 경로 구현 — 단발 검색 후 컨텍스트 반환
- [x] **P4-4** Agentic RAG 옵트인 구현 — `RAG_AGENTIC=true`일 때 관련성 평가 + 재검색 루프 실행
- [x] **P4-5** 쿼리 재작성 드리프트 방지 프롬프트 적용
- [x] **P4-6** 참고 출처(파일명, 페이지) 자동 수집 및 `references` 상태에 저장
- [x] **P4-7** 평가 기반 기본값 결정
  - `docs/adr/0001-rag-default-plain.md`에 plain 기본값 채택 근거 기록

---

## Phase 5. 분석 Agent 구현

- [x] **P5-1** `agents/market_research_agent.py` 구현
  - RAG Tool 호출 (시장 배경 관련 문서 검색)
  - Web Search Tool 호출 (확증 편향 방지 적용)
  - 결과를 `market_background` 상태에 저장
- [x] **P5-2** `agents/lges_analysis_agent.py` 구현
  - RAG + Web Search로 LGES 전략 데이터 수집
  - 파트너십·가동률·고정비 리스크를 프롬프트 변수로 주입
  - 결과를 `lges_strategy` 상태에 저장
- [x] **P5-3** `agents/catl_analysis_agent.py` 구현
  - RAG + Web Search로 CATL 전략 데이터 수집
  - 파트너십·지정학·해외공장 리스크를 프롬프트 변수로 주입
  - 결과를 `catl_strategy` 상태에 저장
- [x] **P5-4** `agents/comparison_agent.py` 구현
  - LGES·CATL 전략 비교 테이블 생성
  - 외부 환경(O/T) 분석 추가하여 SWOT 완성
  - 결과를 `comparison_result` 상태에 저장

---

## Phase 6. Report Generator Agent 구현

- [x] **P6-1** `agents/report_generator.py` 구현
- [x] **P6-2** 보고서 목차 구조(DESIGN.md 4장)에 따라 섹션별 내용 생성
- [x] **P6-3** SUMMARY 섹션 생성 (전체 분석 결과 요약, ½페이지 이내)
- [x] **P6-4** REFERENCE 섹션 자동 생성 (수집된 `references` 상태 기반, 형식 준수)
  - LLM 생성 REFERENCE를 제거하고 코드가 결정적으로 다시 부착
  - 누락된 PDF inline citation은 근거 출처 매핑 섹션으로 보강
- [x] **P6-5** Markdown → HTML → PDF 변환 후 `outputs/report.pdf` 저장
  - `outputs/report.md`도 함께 저장해 평가/리뷰/버전 비교에 사용
  - `markdown` 라이브러리로 HTML 변환, `fitz.Story`로 PDF 렌더링
  - A4 포맷 적용

---

## Phase 7. Distributed Pattern + LangGraph 통합

> ⚠️ Supervisor 패턴 제거 → Distributed Pattern(predetermined edge)으로 변경

- [x] **P7-1** ~~`agents/supervisor.py`~~ 불필요 — Distributed Pattern 채택으로 제거
- [x] **P7-2** `app.py` — LangGraph StateGraph 조립
  - 노드 등록: document_loader → market_research → [lges_analysis ‖ catl_analysis] → comparison → report_generation
  - fan-out 엣지 설정 (market_research → lges/catl 병렬)
  - fan-in 엣지 설정 (lges/catl → comparison)
  - LLM 라우팅 호출 없이 edge만으로 전체 흐름 제어
- [x] **P7-3** 그래프 컴파일 및 import 검증 완료
- [ ] **P7-4** 그래프 시각화 이미지 생성 (`outputs/graph.png`) — README용

---

## Phase 8. 테스트 및 품질 검증

- [x] **P8-1** 전체 파이프라인 End-to-End 실행 테스트
  - `outputs/report.pdf`, `outputs/report.md`, `eval/results/architecture_compare.json` 산출
- [x] **P8-2** RAG 검색 품질 확인
  - `eval/RESULTS.md §1~2`에 RAGAS·IR n=100 결과 기록
- [x] **P8-3** 확증 편향 방지 동작 확인 (3방향 쿼리 결과 비교) → `eval/RESULTS.md §5` (3-arm 측정: risk_recall·entropy·neg_share 우위, 개선 C 적용)
- [x] **P8-4** 보고서 내용 완결성 검토 (목차 모든 섹션 포함 여부)
- [x] **P8-5** REFERENCE 형식 검증 및 개선
  - Web Search 결과를 구조화된 web reference로 누적하도록 개선
  - `eval/eval_references.py`로 본문 PDF 인용과 REFERENCE 섹션 정합성 평가
- [ ] **P8-6** 오류 처리 검토 (API 실패, 빈 검색 결과 등 예외 케이스)

---

## Phase 9. 문서화 및 최종 정리

- [x] **P9-1** `README.md` 작성 (샘플 형식 준수)
  - Overview / Features / Tech Stack / Agents / Architecture / Directory Structure / Contributors
  - 그래프 이미지(`outputs/graph.png`) 삽입
- [x] **P9-2** `DESIGN.md` 내용 최종 검토 및 보완
- [ ] **P9-3** 코드 주석 정리 (핵심 로직 중심)
- [x] **P9-4** 최종 보고서 PDF 파일명 변경 (`agent_{X반}_{이름1+이름2}.pdf`)
  - 로컬 산출물: `outputs/agent_Cloud1반_정재환.pdf`
- [ ] **P9-5** GitHub 리포지토리 생성 및 전체 코드 푸시
- [ ] **P9-6** README에 GitHub 링크 반영

---

## 우선순위 요약

| 순서 | Phase | 핵심 체크포인트 |
|------|-------|----------------|
| 1 | P0 + P1 | 환경 세팅 + PDF 문서 100p 이내 준비 |
| 2 | P2 | 벡터 DB 구축 (오픈소스 임베딩 동작 확인) |
| 3 | P3 + P4 | 공통 유틸 + RAG Tool 동작 |
| 4 | P5 + P3-2 | 분석 Agent + 확증 편향 방지 Web Search |
| 5 | P6 | 보고서 생성 (SUMMARY, REFERENCE 자동화) |
| 6 | P7 | LangGraph 통합 (fan-out/fan-in 병렬 실행 포함) |
| 7 | P8 + P9 | 테스트 및 README·보고서 최종 정리 |
