# 평가 (Evaluation)

RAG 검색 품질(RAGAS·IR)·처리시간·아키텍처 오버헤드를 실측하는 스크립트 모음입니다.
프로덕션 코드(`agents/`, `app.py`)는 건드리지 않습니다.

**실측 결과 요약은 [RESULTS.md](RESULTS.md)**, 평가셋 작성 기준은 [DATASET_CARD.md](DATASET_CARD.md) 참고.

## 사전 조건
1. FAISS 인덱스가 있어야 함 → 없으면 먼저 `python app.py` 한 번 실행 (`vectorstore/` 생성)
2. `.env`에 `OPENAI_API_KEY` 설정 (Q&A 생성·RAGAS 채점에 GPT-4o-mini 사용)
   - Web Search 평가(⑥⑦)는 `.env`에 `TAVILY_API_KEY`도 필요 (실제 웹 검색 수행)
3. RAGAS만 추가 설치 필요: `pip install -r eval/requirements.txt` (단위테스트는 `pytest`도 필요)

## 실행 순서

```bash
# ① Q&A 라벨셋 생성 (IR·RAGAS 공통 입력) — 난이도 선택 (기준: DATASET_CARD.md)
python -m eval.build_dataset --n 30 --difficulty easy
python -m eval.build_dataset --n 30 --difficulty hard --out results/qa_dataset_hard.json

# ② IR 메트릭 — Hit Rate@5 / MRR, 일반 vs Agentic (ragas 불필요, 빠름)
python -m eval.eval_ir                                      # easy(기본)
python -m eval.eval_ir --dataset results/qa_dataset_hard.json   # hard

# ③ RAGAS 4지표 — Agentic(재검색3회) vs 일반(재검색1회)
python -m eval.eval_ragas                                       # easy, 두 모드
python -m eval.eval_ragas --dataset results/qa_dataset_hard.json   # hard

# ④ 처리시간 — 전체/노드별 + async 절감 추정 (실제 파이프라인 1회 실행)
python -m eval.measure_latency

# ⑤ 아키텍처 — Distributed vs Supervisor 라우팅 오버헤드 (파이프라인 2회 실행)
python -m eval.supervisor_compare

# ⑥ Web Search 확증편향 — 리스크 체크리스트 생성(1회) → 검색 단계 3-arm 평가
python -m eval.build_risk_checklist                # 토픽별 리스크 정답지(커밋된 라벨)
python -m eval.eval_web_search --refresh           # 3-arm 검색 평가(스냅샷+지표)

# ⑦ Web Search 확증편향 — 보고서 단계 ablation (파이프라인 2*n회, 느림)
python -m eval.eval_report_balance --n 3

# ⑧ REFERENCE 정합성 — 본문 PDF 인용 ↔ REFERENCE 섹션 정적 검증
# 다른 평가가 report.md를 다시 생성할 수 있으므로 전체 파이프라인 평가 후 마지막에 실행
python -m eval.eval_references --report outputs/report.md
```

결과는 모두 `eval/results/*.json`에 저장됩니다(hard는 `*_hard.json`). 요약은 [RESULTS.md](RESULTS.md).

## 스크립트별 산출값

| 스크립트 | 산출값 |
|---|---|
| `eval_ragas.py` | Faithfulness, Answer Relevancy, Context Precision/Recall (일반 vs Agentic) |
| `eval_ir.py` | Hit Rate@5, MRR (일반 vs Agentic) |
| `supervisor_compare.py` | Distributed vs Supervisor 라우팅 오버헤드(호출·토큰·시간) |
| `measure_latency.py` | 전체/노드별 처리시간, LGES·CATL fan-out 병렬 효과 |
| `build_dataset.py` | easy/hard 평가셋 (기준: `DATASET_CARD.md`) |
| `build_risk_checklist.py` | 토픽별 리스크 체크리스트(리스크 recall 정답지) |
| `eval_web_search.py` | 3방향 쿼리 검색 평가 — neg_share, balance_entropy, risk_recall, degeneration, dup_rate (3-arm) |
| `eval_report_balance.py` | 보고서 ablation — risk_section_claims, report_balance_score, report_risk_recall (단일 vs 3방향) |
| `eval_references.py` | REFERENCE 정합성 — inline PDF citation coverage, reference used rate, web reference count, pass/fail |

## 주의
- **IR 정답 식별**: 검색 결과 청크의 `page_content`를 정답 청크와 **정확 매칭**합니다.
  Q&A를 같은 벡터스토어 청크로 생성하므로 매칭이 보장됩니다.
- **fan-out 시간 해석**: LGES·CATL은 LangGraph 같은 superstep에서 병렬 실행됩니다.
  `lges + catl`은 순차 실행이었다면 걸렸을 반사실적 시간이고, 실제 벽시계 시간에는 `max(lges, catl)`이 주로 반영됩니다.
- **REFERENCE 평가 순서**: `measure_latency`, `supervisor_compare`, `eval_report_balance`는 파이프라인을 다시 실행해 `outputs/report.md`를 덮어쓸 수 있습니다. REFERENCE 정합성은 가장 마지막 산출물 기준으로 해석해야 합니다.
- **inline 인용 0개 처리**: PDF REFERENCE가 있는데 본문 inline PDF 인용이 0개이면 coverage를 0.0으로 보고 실패로 판정합니다.
- **RAGAS 버전**: 0.2.x API 기준으로 작성, ragas 0.4.x에서 동작 확인. 메이저 버전이 다르면 import 경로 조정이 필요할 수 있습니다.
