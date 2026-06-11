# 평가 실측 결과 (Measured Results)

> 실행 환경: GPT-4o-mini · BAAI/bge-m3 · FAISS(262 청크)
> 평가셋: 청크 기반 합성 Q&A, easy/hard 각 n=30 (생성 기준은 `eval/DATASET_CARD.md`)
> 재현 절차: `eval/README.md`. 원본 JSON: `eval/results/*.json`.
> 모든 수치는 **비교(델타) 기준**으로 해석한다.

---

## 1. Agentic RAG vs 일반 RAG — RAGAS (easy/hard 각 n=30)

| 지표 | easy 일반 | easy Agentic | easy Δ | hard 일반 | hard Agentic | hard Δ |
|---|---|---|---|---|---|---|
| Faithfulness | 0.875 | 0.839 | −0.036 | 0.822 | 0.909 | **+0.087** |
| Answer Relevancy | 0.811 | 0.771 | −0.040 | 0.626 | 0.583 | −0.043 |
| Context Precision | 0.849 | 0.879 | +0.030 | 0.889 | 0.932 | **+0.042** |
| Context Recall | 0.867 | 0.900 | +0.033 | 0.900 | 0.900 | 0.000 |

**핵심 관찰**:
- **재검색 효과는 질의 난이도에 따라 방향이 바뀐다.**
  - easy(질문↔청크 어휘 일치): 재검색이 거의 불필요 → Faithfulness/Relevancy 소폭 하락(노이즈 추가).
  - hard(어휘 격차 의도): 재검색이 컨텍스트를 더 적절하게 보정 → **Faithfulness +8.7pp, Precision +4.2pp**.
- Answer Relevancy는 두 셋 모두 Agentic에서 소폭 하락(재검색이 컨텍스트 범위를 넓혀 답변 초점 분산) → 일관된 트레이드오프.
- hard셋의 plain Answer Relevancy(0.626)가 easy(0.811)보다 크게 낮음 → 질문이 실제로 더 어려워졌음을 확인(난이도 knob 작동).

## 2. Agentic RAG vs 일반 RAG — IR (easy/hard 각 n=30, k=5)

| 지표 | easy 일반 | easy Agentic | hard 일반 | hard Agentic |
|---|---|---|---|---|
| Hit Rate@5 | 0.800 | 0.800 | 0.933 | 0.933 |
| MRR | 0.707 | 0.723 | 0.751 | 0.718 |

**핵심 관찰**:
- 어휘를 비튼 hard셋에서도 Hit@5가 떨어지지 않음(오히려 높음) → **bge-m3 의미 임베딩이 패러프레이즈에 강건**,
  단일 정답 청크의 검색 회수율은 재검색 없이도 이미 포화.
- 즉 재검색의 가치는 **정답 청크 회수(IR)**가 아니라 **컨텍스트 적절성·답변 충실도(§1 RAGAS)**에서 나타남.
  IR 지표만 보면 효과가 없어 보이지만, 측정 렌즈가 다름.
- MRR이 hard에서 Agentic에 −0.033 → 재작성이 단일 청크 순위를 흐트러뜨리기도 함(회수율은 유지).

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
- 전체 시간 Δ(+21.5s)에는 단발 실행의 웹검색·LLM 지연 변동 포함. 신뢰 가능한 격리 수치는 8.47초/833토큰.

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

## 종합 해석

- **Agentic RAG**: 효과가 **질의 난이도에 조건부**다. 어휘가 일치하는 쉬운 질의엔 이득이 거의 없고(노이즈),
  어휘가 어긋난 어려운 질의에서 답변 충실도·컨텍스트 정밀도를 끌어올린다(Faithfulness +8.7pp, Precision +4.2pp).
  단, Answer Relevancy는 일관되게 소폭 하락하는 트레이드오프가 있다.
- **검색 회수(IR)**: bge-m3가 패러프레이즈에 강건해 단일 청크 회수율은 재검색 없이도 포화. 재검색의 가치는 IR이 아니라 RAGAS 렌즈에서 드러난다.
- **아키텍처**: 고정 파이프라인에서 Supervisor 대비 라우팅 7회·833토큰·8.5초가 순수 오버헤드로 측정됨.
- **처리시간**: 병목은 report_generation(59%). fan-out 병렬화의 절감 여지(추정 11%)보다 보고서 생성 비용이 큼.

## 측정상의 한계

- 평가셋(n=30)이 코퍼스 청크 기반 합성 Q&A. 사람 검수 없이 temp=0 grounding 지시에 의존. 외부 도메인 일반화 보장 아님.
- single-hop 질문만 포함(multi-hop 미포함).
- n=30은 ±0.03 수준 델타의 통계적 유의성을 단정하기엔 작음. hard Faithfulness(+0.087)는 상대적으로 큰 폭.
- 처리시간·아키텍처 비교는 각 1회 실행 → 라우팅 격리 수치 외에는 변동성 큼.
- async 절감(−27s)은 미구현 상태의 추정치(측정값 아님).
