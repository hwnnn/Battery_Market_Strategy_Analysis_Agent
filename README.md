# 배터리 시장 전략 분석 보고서 자동 생성 Agent

# Subject

배터리 시장 전략 분석 보고서 자동 생성 Agent

## Abstract

전기차 캐즘(EV Chasm) 환경에서 글로벌 배터리 기업 LG에너지솔루션(LGES)과 CATL의 포트폴리오 다각화 전략을 객관적·데이터 기반으로 비교 분석하는 보고서를 LangGraph 기반 Multi-Agent 파이프라인으로 자동 생성합니다. PDF 문서 기반 RAG와 확증 편향 방지 설계의 Web Search를 결합해 정보를 수집하고, Supervisor 없이 고정 edge로 흐름을 제어하는 Distributed Pattern 아키텍처를 적용했습니다.

## Overview

- **Objective**: 전기차 캐즘 환경에서 LG에너지솔루션(LGES)과 CATL의 포트폴리오 다각화 전략을 비교 분석하는 보고서를 Multi-Agent 기반으로 자동 생성
- **Method**: Distributed Pattern LangGraph 파이프라인 — PDF 문서 기반 RAG + 확증 편향 방지 Web Search를 결합한 데이터 수집 후 LLM 분석
- **Tools**: RAG Retrieve Tool (기본 plain RAG, `RAG_AGENTIC=true`일 때 Agentic RAG), Web Search Tool (Tavily, 3방향 쿼리)

## Features

- **PDF 문서 기반 정보 추출**: 사업보고서·산업보고서 등 최대 100페이지 PDF를 청킹·임베딩하여 FAISS 벡터 DB 구축
- **RAG 기본 경로 최적화**: 기본값은 단발 검색(plain RAG)이며, 평가 결과 Agentic RAG의 순이득이 확인되지 않아 `RAG_AGENTIC=true`로 켜는 옵트인 기능으로 분리
  - plain 흐름: `retrieve` → `return`
  - Agentic 흐름: `retrieve` → `grade_documents` → 충분(return) / 부족(rewrite_query) → 반복 (최대 3회)
- **확증 편향 방지 전략 (2단계 적용)**:
  - **(1) 데이터 수집**: Web Search 시 단일 쿼리를 긍정·비판·중립 3방향 쿼리로 자동 분해하여 병렬 검색하고 관점별로 구분 전달
  - **(2) 프롬프트 설계**: 각 분석 섹션마다 성과·기대효과와 한계·리스크를 함께 서술하도록 명시적으로 요구. 섹션 5 "주요 리스크 및 한계"를 독립 항목으로 분리하여 비판 시각 강제화
- **병렬 분석**: LGES·CATL 분석을 LangGraph fan-out/fan-in 구조로 병렬 실행하고 독립 state 필드에 저장
- **SWOT 일괄 작성**: Comparison Agent가 내부(S/W)·외부(O/T)를 한 번에 작성하여 일관성 확보
- **자동 보고서 생성**: SUMMARY(0.5p 이내) + PDF·Web REFERENCE 포함 Markdown/PDF 보고서 자동 저장

## Tech Stack

| Category   | Details                                          |
|------------|--------------------------------------------------|
| Framework  | LangGraph, LangChain, Python 3.11+               |
| LLM        | GPT-4o-mini (OpenAI API)                         |
| Retrieval  | FAISS (로컬, 벡터 검색)                           |
| Embedding  | BAAI/bge-m3 (HuggingFace, 오픈소스, 다국어)          |
| Web Search | Tavily Search API (확증 편향 방지)                 |
| PDF 파싱   | PyMuPDF (fitz)                                    |
| Output     | Markdown/PDF (`outputs/report.md`, `outputs/report.pdf`) |

### Document Processing 파라미터

```python
MAX_PAGES = 100              # PDF 총 페이지 수 제한
CHUNK_SIZE = 500            # 텍스트 청크 크기 (토큰)
CHUNK_OVERLAP = 100         # 청크 간 오버랩
EMBEDDING_MODEL = "BAAI/bge-m3"  # 임베딩 모델
MAX_RAG_ITERATIONS = 3      # 쿼리 재작성 최대 횟수
RAG_TOP_K = 5              # FAISS 검색 상위 K개 문서
RAG_AGENTIC = false        # 기본 plain RAG, true일 때 Agentic RAG
WEB_SEARCH_PERSPECTIVES = true  # 기본 3방향 웹검색
```

## Agents

### 핵심 Agent들

- **Market Research Node** (`agents/market_research_agent.py::market_research_node`): 배터리·전기차 시장 배경 조사
  - RAG Tool + Web Search Tool로 시장 현황 데이터 수집
  - 결과: `state['market_background']`

- **LGES Analysis Node** (`agents/lges_analysis_agent.py::lges_analysis_node`): LG에너지솔루션 포트폴리오 전략 분석
  - RAG Tool + Web Search Tool로 LGES 전략 데이터 수집
  - 결과: `state['lges_strategy']` (독립적 필드)
  - **CATL Analysis와 병렬 실행** (fan-out)

- **CATL Analysis Node** (`agents/catl_analysis_agent.py::catl_analysis_node`): CATL 포트폴리오 전략 분석
  - RAG Tool + Web Search Tool로 CATL 전략 데이터 수집
  - 결과: `state['catl_strategy']` (독립적 필드)
  - **LGES Analysis와 병렬 실행** (fan-out)

- **Comparison Agent** (`agents/comparison_agent.py::comparison_node`): LGES·CATL 비교 분석
  - LGES/CATL 분석 결과를 기반으로 전략 비교 매트릭스 작성
  - SWOT 전체(S/W/O/T) 일괄 작성 → 일관성 보장
  - 결과: `state['comparison_result']`

- **Report Generator Agent** (`agents/report_generator.py::report_generation_node`): 최종 보고서 생성
  - 모든 분석 결과를 목차 구조에 맞게 통합
  - SUMMARY (0.5p 이내) + PDF·Web REFERENCE 자동 생성
  - 결과: `outputs/report.md`, `outputs/report.pdf` 파일로 저장

### 공유 유틸리티

- **Research Base** (`agents/research_base.py`): 에이전트 간 공유하는 RAG + Web Search 기본 로직
  - `run_research()`: RAG 검색 → Web Search → LLM 분석 공통 흐름
  - 각 Analysis Node가 독립적으로 호출

- **RAG Tool** (`agents/rag_tool.py`): plain RAG 기본 + Agentic RAG 옵트인 구현
  - 기본값은 plain RAG(단발 검색)
  - `RAG_AGENTIC=true`일 때 관련성 평가 + 쿼리 재작성 재검색 루프 실행
  - 최대 3회 재시도, 반복 중단 시 최선의 결과 반환

- **Web Search Tool** (`agents/web_search.py`): 확증 편향 방지 웹 검색
  - 단일 쿼리 → 긍정/비판/중립 3방향 쿼리 자동 생성
  - 비판 쿼리에 리스크 키워드 포함을 강제하고, 3방향 검색을 병렬 실행
  - 검색 컨텍스트와 웹 출처 메타데이터를 함께 반환하여 최종 REFERENCE에 반영

## Architecture

```
[START]
   │
   ▼
┌──────────────────────────────────────────┐
│             Document Loader              │
│         PDF 파싱 / FAISS 구축              │
└─────────────────────┬────────────────────┘
                      │ fixed edge
                      ▼
┌──────────────────────────────────────────┐
│      Research & Analysis Agent           │
│         (1) Market Background            │
│      Tools: [RAG Retrieve] [Web Search]  │
└─────────────────────┬────────────────────┘
                      │ fan-out (병렬)
           ┌──────────┴──────────┐
           ▼                     ▼
┌─────────────────┐   ┌──────────────────┐
│  Research &     │   │   Research &     │
│  Analysis Agent │   │  Analysis Agent  │
│  LGES Strategy  │   │  CATL Strategy   │
└────────┬────────┘   └────────┬─────────┘
         └──────────┬──────────┘
                    │ fan-in
                    ▼
┌──────────────────────────────────────────┐
│            Comparison Agent              │
│        전략 비교 매트릭스 + SWOT             │
└─────────────────────┬────────────────────┘
                      │ fixed edge
                      ▼
┌──────────────────────────────────────────┐
│         Report Generator Agent           │
│        최종 보고서 + REFERENCE 생성          │
└─────────────────────┬────────────────────┘
                      │
                    [END]
```

## Directory Structure

```
Battery_Market_Strategy_Analysis_Agent/
├── data/                  # PDF 문서 (최대 100p) — .gitignore 제외
├── agents/                # Agent·Tool 모듈
│   ├── state.py           # BatteryAnalysisState TypedDict
│   ├── llm_config.py      # GPT-4o-mini 초기화
│   ├── document_loader.py # PDF 파싱 / FAISS 구축 노드
│   ├── rag_tool.py        # RAG Tool (plain 기본, Agentic RAG 옵트인)
│   ├── web_search.py      # 확증 편향 방지 Web Search Tool (3방향 쿼리)
│   ├── research_base.py   # 에이전트 간 공유: RAG + Web Search 기본 로직
│   ├── market_research_agent.py    # 시장 배경 분석 노드
│   ├── lges_analysis_agent.py      # LGES 분석 노드
│   ├── catl_analysis_agent.py      # CATL 분석 노드
│   ├── research_analysis_agent.py  # 이전 통합형 분석 노드 파일 (현재 app.py 미사용)
│   ├── comparison_agent.py         # 비교 매트릭스 + SWOT 노드
│   └── report_generator.py         # 보고서 생성 노드
├── prompts/               # 프롬프트 템플릿
│   ├── market_research.txt        # 시장 배경 분석 프롬프트
│   ├── company_analysis.txt       # LGES/CATL 회사 분석 프롬프트
│   ├── comparison.txt             # 비교 매트릭스 + SWOT 프롬프트
│   └── report_generator.txt       # 최종 보고서 생성 프롬프트
├── vectorstore/           # FAISS 인덱스 + manifest (자동 생성) — .gitignore 제외
├── outputs/               # 분석 결과 및 보고서 (자동 생성) — .gitignore 제외
├── app.py                 # 실행 진입점 (LangGraph StateGraph 조립)
├── requirements.txt       # Python 의존성
├── .env.example           # 환경 변수 템플릿
├── DESIGN.md              # 워크플로우 설계 및 평가 기준
├── TODO.md                # 개발 파이프라인 체크리스트
├── DATA_COLLECTION_PLAN.md  # 데이터 수집 계획 (4개 코퍼스)
└── README.md              # 프로젝트 개요 (이 문서)
```

## 실행 방법

```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. .env.example 템플릿 확인 및 .env 파일 생성
cat .env.example                 # 필요한 환경 변수 확인
cp .env.example .env             # 템플릿 복사
# .env 파일에 다음 값 입력:
#   OPENAI_API_KEY=<your-openai-api-key>
#   TAVILY_API_KEY=<your-tavily-api-key>
# 선택:
#   RAG_AGENTIC=true              # Agentic RAG 재검색 루프 활성화
#   WEB_SEARCH_PERSPECTIVES=off   # 3방향 웹검색 ablation

# 4. PDF 문서 준비 (총 100페이지 이내)
# data/ 폴더에 PDF 파일 배치
# 권장: DATA_COLLECTION_PLAN.md 참조하여 4개 코퍼스 준비

# 5. 실행
python app.py
```

보고서는 `outputs/report.md`와 `outputs/report.pdf`에 저장됩니다.

## Evaluation Summary

- RAGAS·IR n=100 기준, 이 코퍼스에서는 Agentic RAG가 plain RAG 대비 검색/답변 품질의 순이득을 주지 못해 plain RAG를 기본값으로 채택했습니다.
- Supervisor 구조는 고정 파이프라인에서 라우팅 LLM 호출 7회, 833토큰, 8.47초의 순수 오버헤드가 측정되어 Distributed Pattern을 유지합니다.
- Web Search 3방향 쿼리는 검색 단계에서 부정 관점 비중·관점 균형·리스크 회수율을 개선했지만, 최종 보고서 균형은 프롬프트의 리스크 섹션 강제 효과가 더 크게 작용했습니다.
- Report Generator는 LLM이 임의 작성한 REFERENCE를 제거한 뒤 수집된 출처 기반으로 결정적 REFERENCE를 다시 붙이고, 누락된 PDF 인용은 근거 출처 매핑으로 보강합니다.
- `eval/eval_references.py`로 본문 PDF 인용과 REFERENCE 섹션의 정합성을 정적으로 점검할 수 있습니다. 본문 inline PDF 인용이 0개이면 실패로 판정합니다.

## Contributors

- **정재환** : RAG 설계·평가, 확증 편향 방지 Web Search Tool, LangGraph StateGraph 조립, Report Generator Agent, 프롬프트 엔지니어링
