# 배터리 시장 전략 분석 보고서 자동 생성 Agent

# Subject

배터리 시장 전략 분석 보고서 자동 생성 Agent

## Abstract

전기차 캐즘(EV Chasm) 환경에서 글로벌 배터리 기업 LG에너지솔루션(LGES)과 CATL의 포트폴리오 다각화 전략을 객관적·데이터 기반으로 비교 분석하는 보고서를 LangGraph 기반 Multi-Agent 파이프라인으로 자동 생성합니다. PDF 문서 기반 Agentic RAG와 확증 편향 방지 설계의 Web Search를 결합하여 신뢰도 높은 정보를 수집하고, Distributed Pattern 아키텍처의 병렬 처리로 효율성을 확보합니다.

## Overview

- **Objective**: 전기차 캐즘 환경에서 LG에너지솔루션(LGES)과 CATL의 포트폴리오 다각화 전략을 비교 분석하는 보고서를 Multi-Agent 기반으로 자동 생성
- **Method**: Distributed Pattern LangGraph 파이프라인 — PDF 문서 기반 Agentic RAG + 확증 편향 방지 Web Search를 결합한 데이터 수집 후 LLM 분석
- **Tools**: RAG Retrieve Tool (Agentic RAG, 최대 3회 재시도), Web Search Tool (Tavily, 3방향 쿼리)

## Features

- **PDF 문서 기반 정보 추출**: 사업보고서·산업보고서 등 최대 100페이지 PDF를 청킹·임베딩하여 FAISS 벡터 DB 구축
- **Agentic RAG**: 검색 결과 관련성을 LLM으로 평가 → 부족 시 쿼리 자동 재작성하여 최대 3회 재검색
  - 흐름: `retrieve` → `grade_documents` → 충분(return) / 부족(rewrite_query) → 반복 (최대 3회)
- **확증 편향 방지 전략 (2단계 적용)**:
  - **(1) 데이터 수집**: Web Search 시 단일 쿼리를 긍정·비판·중립 3방향 쿼리로 자동 분해하여 병렬 검색 후 관점별로 구분 전달
  - **(2) 프롬프트 설계**: 각 분석 섹션마다 성과·기대효과와 한계·리스크를 함께 서술하도록 명시적으로 요구. 섹션 5 "주요 리스크 및 한계"를 독립 항목으로 분리하여 비판 시각 강제화
- **병렬 분석**: LGES·CATL 분석을 LangGraph fan-out/fan-in으로 동시 실행하여 처리 시간 단축
- **SWOT 일괄 작성**: Comparison Agent가 내부(S/W)·외부(O/T)를 한 번에 작성하여 일관성 확보
- **자동 보고서 생성**: SUMMARY(0.5p 이내) + REFERENCE(형식 준수) 포함 Markdown 보고서 자동 저장

## Tech Stack

| Category   | Details                                          |
|------------|--------------------------------------------------|
| Framework  | LangGraph, LangChain, Python 3.11+               |
| LLM        | GPT-4o-mini (OpenAI API)                         |
| Retrieval  | FAISS (로컬, 벡터 검색)                           |
| Embedding  | BAAI/bge-m3 (HuggingFace, 오픈소스, 다국어)          |
| Web Search | Tavily Search API (확증 편향 방지)                 |
| PDF 파싱   | PyMuPDF (fitz)                                    |
| Output     | Markdown (outputs/report.md)                     |

### Document Processing 파라미터

```python
MAX_PAGES = 100              # PDF 총 페이지 수 제한
CHUNK_SIZE = 500            # 텍스트 청크 크기 (토큰)
CHUNK_OVERLAP = 100         # 청크 간 오버랩
EMBEDDING_MODEL = "BAAI/bge-m3"  # 임베딩 모델
MAX_RAG_ITERATIONS = 3      # 쿼리 재작성 최대 횟수
RAG_TOP_K = 5              # FAISS 검색 상위 K개 문서
```

## Agents

### 핵심 Agent들

- **Market Research Node** (`agents/research_analysis_agent.py::market_research_node`): 배터리·전기차 시장 배경 조사
  - RAG Tool + Web Search Tool로 시장 현황 데이터 수집
  - 결과: `state['market_background']`

- **LGES Analysis Node** (`agents/research_analysis_agent.py::lges_analysis_node`): LG에너지솔루션 포트폴리오 전략 분석
  - RAG Tool + Web Search Tool로 LGES 전략 데이터 수집
  - 결과: `state['lges_strategy']` (독립적 필드)
  - **CATL Analysis와 병렬 실행** (fan-out)

- **CATL Analysis Node** (`agents/research_analysis_agent.py::catl_analysis_node`): CATL 포트폴리오 전략 분석
  - RAG Tool + Web Search Tool로 CATL 전략 데이터 수집
  - 결과: `state['catl_strategy']` (독립적 필드)
  - **LGES Analysis와 병렬 실행** (fan-out)

- **Comparison Agent** (`agents/comparison_agent.py::comparison_node`): LGES·CATL 비교 분석
  - LGES/CATL 분석 결과를 기반으로 전략 비교 매트릭스 작성
  - SWOT 전체(S/W/O/T) 일괄 작성 → 일관성 보장
  - 결과: `state['comparison_result']`

- **Report Generator Agent** (`agents/report_generator.py::report_generation_node`): 최종 보고서 생성
  - 모든 분석 결과를 목차 구조에 맞게 통합
  - SUMMARY (0.5p 이내) + REFERENCE 자동 생성
  - 결과: `outputs/report.md` 파일로 저장

### 공유 유틸리티

- **Research Base** (`agents/research_base.py`): 에이전트 간 공유하는 RAG + Web Search 기본 로직
  - `run_research()`: RAG 검색 → Web Search → LLM 분석 공통 흐름
  - 각 Analysis Node가 독립적으로 호출

- **RAG Tool** (`agents/rag_tool.py`): Agentic RAG 구현
  - 자동 쿼리 재작성으로 검색 결과 부족 시 재검색
  - 최대 3회 재시도, 반복 중단 시 최선의 결과 반환

- **Web Search Tool** (`agents/web_search.py`): 확증 편향 방지 웹 검색
  - 단일 쿼리 → 긍정/비판/중립 3방향 쿼리 자동 생성
  - 3방향 검색 병렬 실행, 결과를 관점별로 구분하여 반환

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
│   ├── rag_tool.py        # Agentic RAG Tool (자동 쿼리 재작성)
│   ├── web_search.py      # 확증 편향 방지 Web Search Tool (3방향 쿼리)
│   ├── research_base.py   # 에이전트 간 공유: RAG + Web Search 기본 로직
│   ├── research_analysis_agent.py  # Market / LGES / CATL 분석 노드 (3개 함수)
│   ├── lges_analysis_agent.py      # (deprecated: research_analysis_agent에 통합)
│   ├── catl_analysis_agent.py      # (deprecated: research_analysis_agent에 통합)
│   ├── comparison_agent.py         # 비교 매트릭스 + SWOT 노드
│   └── report_generator.py         # 보고서 생성 노드
├── prompts/               # 프롬프트 템플릿
│   ├── market_research.txt        # 시장 배경 분석 프롬프트
│   ├── company_analysis.txt       # LGES/CATL 회사 분석 프롬프트
│   ├── comparison.txt             # 비교 매트릭스 + SWOT 프롬프트
│   └── report_generator.txt       # 최종 보고서 생성 프롬프트
├── vectorstore/           # FAISS 인덱스 (자동 생성) — .gitignore 제외
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

# 4. PDF 문서 준비 (총 100페이지 이내)
# data/ 폴더에 PDF 파일 배치
# 권장: DATA_COLLECTION_PLAN.md 참조하여 4개 코퍼스 준비

# 5. 실행
python app.py
```

보고서는 `outputs/report.md`에 저장됩니다.

## Contributors

- **정재환** : Agentic RAG 설계·구현, 확증 편향 방지 Web Search Tool, LangGraph StateGraph 조립, Report Generator Agent, 프롬프트 엔지니어링
