# 평가 (Evaluation)

RAG 검색 품질(RAGAS·IR)·처리시간·아키텍처 오버헤드를 실측하는 스크립트 모음입니다.
프로덕션 코드(`agents/`, `app.py`)는 건드리지 않습니다.

**실측 결과 요약은 [RESULTS.md](RESULTS.md)**, 평가셋 작성 기준은 [DATASET_CARD.md](DATASET_CARD.md) 참고.

## 사전 조건
1. FAISS 인덱스가 있어야 함 → 없으면 먼저 `python app.py` 한 번 실행 (`vectorstore/` 생성)
2. `.env`에 `OPENAI_API_KEY` 설정 (Q&A 생성·RAGAS 채점에 GPT-4o-mini 사용)
3. RAGAS만 추가 설치 필요: `pip install -r eval/requirements.txt`

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
```

결과는 모두 `eval/results/*.json`에 저장됩니다(hard는 `*_hard.json`). 요약은 [RESULTS.md](RESULTS.md).

## 스크립트별 산출값

| 스크립트 | 산출값 |
|---|---|
| `eval_ragas.py` | Faithfulness, Answer Relevancy, Context Precision/Recall (일반 vs Agentic) |
| `eval_ir.py` | Hit Rate@5, MRR (일반 vs Agentic) |
| `supervisor_compare.py` | Distributed vs Supervisor 라우팅 오버헤드(호출·토큰·시간) |
| `measure_latency.py` | 전체/노드별 처리시간, fan-out async 절감 추정치 |
| `build_dataset.py` | easy/hard 평가셋 (기준: `DATASET_CARD.md`) |

## 주의
- **IR 정답 식별**: 검색 결과 청크의 `page_content`를 정답 청크와 **정확 매칭**합니다.
  Q&A를 같은 벡터스토어 청크로 생성하므로 매칭이 보장됩니다.
- **async 절감은 "추정치"**: 현재 LGES·CATL은 동기 구현이라 순차 실행됩니다.
  `max(lges, catl)` 기준으로 병렬화 시 예상 절감을 계산할 뿐, 실제 async 실측이 아닙니다.
- **RAGAS 버전**: 0.2.x API 기준으로 작성, ragas 0.4.x에서 동작 확인. 메이저 버전이 다르면 import 경로 조정이 필요할 수 있습니다.
