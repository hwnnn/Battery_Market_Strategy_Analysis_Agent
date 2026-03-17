# 배터리 시장 전략 분석 Agent - 설계 산출물

> Updated: 2026-03-17

---

## 1. Workflow 설계

### 1.1 Goal

> 전기차 캐즘 환경 속에서 글로벌 TOP-Tier 배터리 기업인 **LG에너지솔루션(LGES)** 과 **CATL**의 포트폴리오 다각화 전략을 객관적·데이터 기반으로 비교 분석하고, 전략적 차이점·강약점을 담은 **"배터리 시장 전략 분석 보고서"** 를 자동 생성한다.

---

### 1.2 Criteria (평가 기준)

평가 기준은 수식이 고정되어 있어 직접 계산 가능한 표준 메트릭만 적용한다.

#### A. RAG 검색 품질 — RAGAS (arXiv:2309.15217, EACL 2024)

| 메트릭 | 정의 | 목표 |
|--------|------|------|
| **Faithfulness** | 생성된 내용 중 검색 컨텍스트로 뒷받침되는 주장의 비율. 환각(Hallucination) 탐지 | ≥ 0.8 |
| **Answer Relevancy** | 생성된 답변이 원래 쿼리와 얼마나 관련 있는지. 역질문 생성 후 코사인 유사도로 측정 | ≥ 0.8 |
| **Context Precision** | 검색된 청크 중 관련 있는 청크가 상위 순위에 집중되는 정도 (신호 대 잡음비) | ≥ 0.8 |
| **Context Recall** | 답변 생성에 필요한 정보가 검색된 컨텍스트에 포함된 비율 | ≥ 0.8 |

> 모든 메트릭 점수 범위 0 ~ 1. `ragas` 라이브러리로 직접 산출.

#### B. 검색 순위 품질 — IR 표준 메트릭 (Manning et al., Stanford IR)

| 메트릭 | 정의 | 목표 |
|--------|------|------|
| **Hit Rate@5** | 상위 5개 검색 결과에 관련 청크가 1개 이상 포함된 쿼리 비율 | ≥ 0.9 |
| **MRR** (Mean Reciprocal Rank) | 첫 번째 관련 문서의 역순위 평균 | ≥ 0.8 |

> 모든 메트릭 점수 범위 0 ~ 1. 직접 계산 가능.

---

### 1.3 Task 정의

```
T1. 문서 수집 및 벡터 DB 구축
    - PDF 문서 파싱 (최대 100페이지)
    - 텍스트 청킹 → 오픈소스 임베딩 → FAISS 인덱스 저장

T2. 시장 배경 조사
    - RAG: 배터리 시장 현황, 전기차 캐즘 관련 문서 검색
    - Web Search: 최신 시장 데이터, 통계 수집 (확증 편향 방지 적용)

T3. LGES 전략 분석
    - RAG: LGES 관련 문서 검색
    - Web Search: LGES 포트폴리오 다각화 전략, ESS·로봇 배터리·HEV 관련 뉴스

T4. CATL 전략 분석
    - RAG: CATL 관련 문서 검색
    - Web Search: CATL 나트륨이온 배터리, ESS, 신흥시장 공략 관련 뉴스

T5. 비교 분석 및 SWOT 작성
    - T3·T4 결과를 기반으로 전략 비교 테이블 작성
    - 각 기업 SWOT (S/W: 내부, O/T: 외부) 작성

T6. 보고서 생성
    - 목차 구조에 따라 보고서 초안 작성
    - SUMMARY(½p 이내) + REFERENCE 자동 생성
```

---

### 1.4 Control Strategy

**Distributed Pattern** 채택

- **선택 이유**: 파이프라인의 실행 순서가 고정되어 있으므로, 중앙 조율자(Supervisor) 없이 각 노드가 LangGraph의 **predetermined/conditional edge**를 통해 다음 노드로 직접 전달하는 방식이 적합. Supervisor는 매 단계마다 LLM을 추가 호출하는 오버헤드가 발생하며, 이 파이프라인에서 동적 라우팅이 필요한 구간이 없으므로 불필요
- **병렬 실행**: T3(LGES 분석)와 T4(CATL 분석)는 독립적이므로 LangGraph의 fan-out edge로 병렬 수행
- **재시도 로직**: Agentic RAG의 재검색 분기는 Research & Analysis Agent 내부의 **conditional edge**로 처리. Supervisor 없이 동일한 분기 제어 가능
- **확증 편향 방지**: Web Search Tool이 단일 쿼리 입력 시 **긍정·비판·중립** 3방향 쿼리를 자동 생성·병렬 실행하고 결과를 통합하여 반환

```
확증 편향 방지 전략 상세:
  Query 생성 시 3가지 관점을 강제 적용:
    1. 긍정 쿼리  : "CATL ESS 성장 전략 성과"
    2. 비판 쿼리  : "CATL ESS 전략 리스크 한계 문제점"
    3. 중립 쿼리  : "CATL ESS 전략 현황 분석 2025"
  → 3개 결과를 모두 수집 후 Synthesis 단계에서 균형 있게 통합
```

---

### 1.5 Structure (Distributed Pattern)

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


※ Agentic RAG conditional edge (Research & Analysis Agent 내부):

   retrieve ──▶ grade_documents ──▶ [충분] ──▶ generate
                      │
                   [부족]
                      │
               rewrite_query ──▶ retrieve  (최대 3회)
```

---

## 2. Workflow → Agent 매핑

### 2.0 구조 개선 근거

초기 설계(7개 Agent)에서 발견된 비효율을 제거하여 **3개 Agent + 2개 Tool + 1개 Node**로 축소했다.

| 문제 | 원인 | 개선 |
|------|------|------|
| **Supervisor Agent 불필요** | 파이프라인 순서가 고정되어 있어 LLM 기반 동적 라우팅이 필요 없음. 매 단계마다 LLM 호출 오버헤드만 발생 | **Distributed Pattern**으로 전환. 노드 간 predetermined/conditional edge로 라우팅 처리 |
| **LGES·CATL Agent 중복** | 두 에이전트가 완전히 동일한 로직(RAG + Web Search + 전략 분석)을 수행하며 컨텍스트(회사명)만 다름. 코드·프롬프트·유지보수 전체 중복 | 단일 **Research & Analysis Agent**를 컨텍스트만 바꿔 호출로 통합 |
| **Market Research Agent 불필요** | 역할이 "RAG + Web Search로 정보 수집"으로, LGES/CATL Agent와 구조적으로 동일. 독립 에이전트로 분리할 이유 없음 | Research & Analysis Agent의 첫 번째 호출(Market Background)로 흡수 |
| **RAG·Web Search를 Agent로 분리** | 두 컴포넌트는 정보를 검색하는 수단이지, 자체적으로 판단·결정하지 않음. Agent 추상화가 오버엔지니어링이며 Agent→Agent 호출로 불필요한 오버헤드 발생 | **Tool**로 격하. 분석 Agent가 직접 호출 |
| **SWOT 분산 작성** | S/W는 분석 Agent에서, O/T는 Comparison Agent에서 생성 → 두 단계에 걸쳐 SWOT를 완성하면 일관성 저하 위험 | Comparison Agent가 LGES·CATL 분석 결과를 받아 **SWOT 전체(S/W/O/T)를 일괄 작성** |

---

### 2.1 에이전트 정의

**Agents**

| Agent | 역할 | 담당 Task |
|-------|------|-----------|
| **Research & Analysis Agent** | RAG Tool + Web Search Tool을 호출하여 정보를 수집하고 분석. 동일 에이전트를 3회 실행 (① Market Background 순차, ② LGES Strategy + ③ CATL Strategy 병렬). 라우팅은 LangGraph conditional edge가 담당 | T2, T3, T4 |
| **Comparison Agent** | Research Agent 결과를 State에서 읽어 전략 비교 매트릭스 및 SWOT 전체(S/W/O/T) 일괄 작성 | T5 |
| **Report Generator Agent** | 목차 구조에 따라 최종 보고서 작성, SUMMARY·REFERENCE 생성 | T6 |

**Tools** (Agent 아님 — LLM 판단 없이 실행되는 유틸리티)

| Tool | 역할 |
|------|------|
| **RAG Retrieve Tool** | FAISS 검색 + 관련성 부족 시 쿼리 재작성 루프 (최대 3회). Research & Analysis Agent가 직접 호출 |
| **Web Search Tool** | Tavily 검색. 긍정·비판·중립 3방향 쿼리 병렬 실행 후 통합 반환. Research & Analysis Agent가 직접 호출 |

**Node** (LLM 미사용 — 단순 데이터 처리)

| Node | 역할 |
|------|------|
| **Document Loader** | PDF 파싱 → 텍스트 청킹 → BAAI/bge-m3 임베딩 → FAISS 저장. 파이프라인 시작 시 1회만 실행 |

---

### 2.2 RAG 적용 대상

| 구분 | 내용 |
|------|------|
| 적용 대상 | Research & Analysis Agent가 RAG Retrieve Tool을 직접 호출 |
| 문서 범위 | PDF 문서 최대 **100페이지** |
| 예상 문서 | LGES 사업보고서, CATL Annual Report, IEA Global EV Outlook, 산업연구원 배터리 보고서 등 |
| RAG 방식 | **Agentic RAG**: 검색 결과가 불충분할 경우 쿼리를 재구성하여 재검색 (최대 3회 반복) |
| 청킹 전략 | Recursive Text Splitter (chunk_size=500, overlap=100) |
| 벡터 DB | **FAISS** (로컬, 무료) |

---

### 2.3 Embedding 모델

| 항목 | 선택 |
|------|------|
| **모델명** | `BAAI/bge-m3` |
| **유형** | 오픈소스 (HuggingFace) |
| **선택 이유** | 한국어·영어·중국어 다국어 지원, Dense + Sparse + ColBERT 통합 검색, 무료 |
| **대안** | `intfloat/multilingual-e5-large` (경량, 빠른 추론 필요 시) |
| **실행 환경** | 로컬 CPU/GPU (HuggingFaceEmbeddings via LangChain) |

---

## 3. Agent 상세 설계

### 3.1 State 정의

```python
class BatteryAnalysisState(TypedDict):
    # 입력
    query: str                          # 분석 주제

    # 문서 관련
    documents: List[Document]           # 로드된 PDF 문서
    vectorstore: Any                    # FAISS 벡터스토어

    # 중간 결과
    market_background: str              # 시장 배경 분석 결과
    lges_strategy: str                  # LGES 전략 분석 결과
    catl_strategy: str                  # CATL 전략 분석 결과
    lges_swot: Dict[str, List[str]]     # LGES SWOT {S, W, O, T}
    catl_swot: Dict[str, List[str]]     # CATL SWOT {S, W, O, T}
    comparison_result: str              # 비교 분석 결과

    # 참고 자료 추적
    references: List[Dict]              # 활용된 참고 자료 목록

    # 보고서
    report_draft: str                   # 최종 보고서 전문

    # 제어 (라우팅은 LangGraph edge가 담당 — 별도 Supervisor 없음)
    error_messages: List[str]           # 오류 로그
    rag_iteration: int                  # Agentic RAG 재검색 횟수 (conditional edge 판단용)
```

---

### 3.2 Graph 흐름 (LangGraph)

```python
# Distributed Pattern: LLM 라우팅 없이 edge로 흐름 제어
graph.add_edge(START,              "document_loader")
graph.add_edge("document_loader",  "market_research")
graph.add_edge("market_research",  "lges_analysis")   # fan-out (병렬)
graph.add_edge("market_research",  "catl_analysis")   # fan-out (병렬)
graph.add_edge("lges_analysis",    "comparison")      # fan-in
graph.add_edge("catl_analysis",    "comparison")      # fan-in
graph.add_edge("comparison",       "report_generation")
graph.add_edge("report_generation", END)
```

```
[START]
   │
   ▼  fixed edge
[document_loader]          T1 : PDF 파싱 / FAISS 구축
   │
   ▼  fixed edge
[market_research]          T2 : 시장 배경 (Research & Analysis Agent)
   │
   ├──────── fan-out ────────┐
   ▼                         ▼
[lges_analysis]         [catl_analysis]    T3/T4 : 병렬 실행
   │                         │
   └──────── fan-in  ────────┘
                   │
                   ▼  fixed edge
         [comparison]             T5 : 비교 매트릭스 + SWOT
                   │
                   ▼  fixed edge
        [report_generation]       T6 : 최종 보고서
                   │
                 [END]
```

---

## 4. 보고서 목차 (초안)

```
배터리 시장 전략 분석 보고서
LG에너지솔루션 vs CATL 포트폴리오 다각화 전략 비교

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUMMARY                                          (½페이지 이내)
  - 분석 배경 한 줄 요약
  - 핵심 발견사항 3~5개 불릿
  - 최종 시사점 한 줄 결론

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 시장 배경 : 배터리·전기차 시장 환경 변화
   1.1 글로벌 전기차 캐즘 현황
   1.2 HEV 피벗 트렌드 및 시장 재편
   1.3 배터리 원자재·공급망 이슈
   1.4 정책 환경 변화 (IRA, EU 배터리 규정 등)

2. LG에너지솔루션 포트폴리오 다각화 전략
   2.1 현황 및 사업 구조
   2.2 핵심 다각화 방향 (ESS / 로봇 배터리 / HEV 대응)
   2.3 주요 파트너십 및 생산 전략
   2.4 핵심 경쟁력

3. CATL 포트폴리오 다각화 전략
   3.1 현황 및 사업 구조
   3.2 핵심 다각화 방향 (나트륨이온 배터리 / ESS / 신흥시장)
   3.3 주요 파트너십 및 생산 전략
   3.4 핵심 경쟁력

4. 핵심 전략 비교 및 SWOT 분석
   4.1 전략 비교 매트릭스
   4.2 LG에너지솔루션 SWOT
       - Strengths / Weaknesses (내부)
       - Opportunities / Threats (외부)
   4.3 CATL SWOT
       - Strengths / Weaknesses (내부)
       - Opportunities / Threats (외부)
   4.4 두 기업 SWOT 비교 종합표

5. 종합 시사점
   5.1 전략 차별화 포인트
   5.2 한국 배터리 산업에 대한 시사점
   5.3 향후 경쟁 구도 전망

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REFERENCE
  - 기관 보고서
  - 학술 논문
  - 웹페이지
```

---

## 5. 기술 스택 요약

| Category | Details |
|----------|---------|
| Framework | LangGraph, LangChain, Python 3.11+ |
| LLM | GPT-4o-mini (OpenAI API) |
| Retrieval | FAISS |
| Embedding | BAAI/bge-m3 (HuggingFace, 오픈소스) |
| Web Search | Tavily Search API |
| PDF Parsing | PyMuPDF (fitz) |
| Output | Markdown → PDF 변환 |

---

*본 설계 문서는 개발 진행에 따라 업데이트될 수 있습니다.*
