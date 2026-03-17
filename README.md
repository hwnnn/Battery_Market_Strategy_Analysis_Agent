# 배터리 시장 전략 분석 보고서 자동 생성 Agent

## Overview

- **Objective**: 전기차 캐즘 환경에서 LG에너지솔루션(LGES)과 CATL의 포트폴리오 다각화 전략을 비교 분석하는 보고서를 Multi-Agent 기반으로 자동 생성
- **Method**: Distributed Pattern LangGraph 파이프라인 — PDF 문서 기반 Agentic RAG + 확증 편향 방지 Web Search를 결합한 데이터 수집 후 LLM 분석
- **Tools**: RAG Retrieve Tool (Agentic RAG, 최대 3회 재시도), Web Search Tool (Tavily, 3방향 쿼리)

## Features

- **PDF 문서 기반 정보 추출**: 사업보고서·산업보고서 등 최대 100페이지 PDF를 청킹·임베딩하여 FAISS 벡터 DB 구축
- **Agentic RAG**: 검색 결과 관련성이 부족할 경우 쿼리를 자동 재작성하여 최대 3회 재검색
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
| Retrieval  | FAISS (로컬)                                     |
| Embedding  | BAAI/bge-m3 (HuggingFace, 오픈소스, 다국어)       |
| Web Search | Tavily Search API                                |
| PDF 파싱   | PyMuPDF (fitz)                                   |
| Output     | Markdown (outputs/report.md)                     |

## Agents

- **Research & Analysis Agent** (`agents/research_analysis_agent.py`): RAG Tool + Web Search Tool로 데이터 수집 후 LLM 분석. 동일 에이전트를 Market Background(순차) → LGES/CATL(병렬) 3회 실행
- **Comparison Agent** (`agents/comparison_agent.py`): LGES·CATL 분석 결과를 바탕으로 전략 비교 매트릭스 및 SWOT 전체(S/W/O/T) 일괄 작성
- **Report Generator Agent** (`agents/report_generator.py`): 분석 결과를 보고서 목차 구조에 맞게 통합, SUMMARY·REFERENCE 자동 생성

## Architecture

```
[START]
   │
   ▼
┌──────────────────────────────────────────┐
│             Document Loader              │
│         PDF 파싱 / FAISS 구축             │
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
│        전략 비교 매트릭스 + SWOT          │
└─────────────────────┬────────────────────┘
                      │ fixed edge
                      ▼
┌──────────────────────────────────────────┐
│         Report Generator Agent           │
│        최종 보고서 + REFERENCE 생성       │
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
│   ├── rag_tool.py        # Agentic RAG Tool
│   ├── web_search.py      # 확증 편향 방지 Web Search Tool
│   ├── research_analysis_agent.py  # Market / LGES / CATL 분석 노드
│   ├── comparison_agent.py         # 비교 매트릭스 + SWOT 노드
│   └── report_generator.py         # 보고서 생성 노드
├── prompts/               # 프롬프트 템플릿
│   ├── market_research.txt
│   ├── company_analysis.txt
│   ├── comparison.txt
│   └── report_generator.txt
├── vectorstore/           # FAISS 인덱스 — .gitignore 제외
├── outputs/               # 분석 결과 및 보고서 — .gitignore 제외
├── app.py                 # 실행 진입점 (LangGraph 조립)
├── requirements.txt
├── .env.example
└── README.md
```

## 실행 방법

```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY, TAVILY_API_KEY 입력

# 4. PDF 문서 준비 (총 100페이지 이내)
# data/ 폴더에 PDF 파일 배치

# 5. 실행
python app.py
```

보고서는 `outputs/report.md`에 저장됩니다.

## Contributors

- 기여자 이름 : Agentic RAG 설계·구현, 확증 편향 방지 Web Search Tool
- 기여자 이름 : LangGraph StateGraph 조립, Report Generator Agent, 프롬프트 엔지니어링
