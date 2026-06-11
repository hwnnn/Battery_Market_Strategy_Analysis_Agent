# 평가 실측 결과 (Measured Results)

> 실행 환경: GPT-4o-mini · BAAI/bge-m3 · FAISS(262 청크) · 평가셋 n=30 (청크 기반 합성 Q&A)
> 재현 절차: `eval/README.md`. 원본 JSON: `eval/results/*.json`.
> 모든 수치는 **비교(델타) 기준**으로 해석한다.

---

## 1. Agentic RAG vs 일반 RAG — RAGAS (n=30)

| 지표 | 일반 RAG(재검색 1회) | Agentic RAG(재검색 3회) | Δ |
|---|---|---|---|
| Faithfulness | 0.875 | 0.839 | −0.036 |
| Answer Relevancy | 0.811 | 0.771 | −0.040 |
| Context Precision | 0.849 | 0.879 | +0.030 |
| Context Recall | 0.867 | 0.900 | +0.033 |

- 재검색(grade→rewrite)은 검색 단계 품질(precision/recall)을 올리는 반면, 넓어진 컨텍스트가
  답변 충실도(faithfulness/relevancy)를 소폭 낮추는 방향으로 나타남.
- 델타 폭(±0.03~0.04)은 n=30 기준 노이즈 범위. 방향성 참고용.

## 2. Agentic RAG vs 일반 RAG — IR (n=30, k=5)

| 지표 | 일반 RAG | Agentic RAG | Δ |
|---|---|---|---|
| Hit Rate@5 | 0.800 | 0.800 | +0.000 |
| MRR | 0.707 | 0.723 | +0.017 |

- 정답 청크 포함률(Hit@5)은 동일, 순위(MRR)만 소폭 개선.
- 합성 평가셋에서 첫 검색이 대체로 충분 → 재검색(rewrite) 발동 빈도가 낮아 효과 제한적.

## 3. 아키텍처 — Distributed vs Supervisor (각 1회 실행, 동일 worker)

| 항목 | Distributed | Supervisor | Δ |
|---|---|---|---|
| 전체 시간 | 129.1s | 150.6s | +21.5s |
| LLM 호출 수 | 15 | 22 | +7 |
| 총 토큰 | 43,095 | 44,074 | +979 |
| 추정 비용(USD) | 0.0094 | 0.0097 | +0.0003 |

**라우팅 전용 오버헤드(supervisor 노드 격리 측정)**: 7회 호출 · 833 토큰 · 8.47초

- 고정 순서 파이프라인에서 Supervisor는 단계마다 라우팅 LLM 호출(worker 6 + FINISH = 7회) 발생.
  Distributed(edge 라우팅)는 라우팅 호출 0회.
- 전체 시간 Δ(+21.5s)에는 단발 실행의 웹검색·LLM 지연 변동이 포함됨. 신뢰 가능한 격리 수치는 8.47초/833토큰.

## 4. 처리시간 프로파일 (n=1, 동기 실행)

| 노드 | 시간(s) | 비중 |
|---|---|---|
| document_loader | 4.9 | 2% |
| market_research | 29.1 | 12% |
| lges_analysis | 27.0 | 11% |
| catl_analysis | 44.7 | 18% |
| comparison | 20.7 | 9% |
| report_generation | 143.8 | 59% |
| 합계(동기) | 243.2 | 100% |

- fan-out async 전환 시 예상 절감(추정): `lges+catl(71.7s)` → `max(44.7s)` = −27초(전체의 11%). 현재 동기 구현이라 추정치.
- 최대 비중 노드는 report_generation(59%).
- report_generation은 실행 간 편차가 큼(본 측정 144s, §3 Distributed 런에서는 그보다 짧음). 단발 시간은 변동성 큼.

---

## 측정상의 한계

- 평가셋(n=30)이 코퍼스 청크에서 생성된 합성 Q&A → 첫 검색 난도가 낮아 절대 점수가 높게,
  재검색 효과는 작게 나오는 경향. 재검색이 필요한 어려운 질의(패러프레이즈·멀티홉)는 미포함.
- n=30은 ±0.03 수준 델타의 유의성을 판단하기에 표본이 작음.
- 처리시간·아키텍처 비교는 각 1회 실행 → 라우팅 격리 수치를 제외하면 변동성 큼.
- async 절감(−27s)은 미구현 상태의 추정치(측정값 아님).
