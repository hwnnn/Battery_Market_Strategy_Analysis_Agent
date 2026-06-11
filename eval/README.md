# 평가 (Evaluation) — 📊 수치 산출

포트폴리오에 표기된 측정값(RAGAS·IR·처리시간·아키텍처 비교)을 실제로 산출하는 스크립트 모음입니다.
프로덕션 코드(`agents/`, `app.py`)는 건드리지 않습니다.

**실측 결과 요약은 [RESULTS.md](RESULTS.md)** 참고 (비교/델타 중심 해석 + 포트폴리오 기재 가이드).

## 사전 조건
1. FAISS 인덱스가 있어야 함 → 없으면 먼저 `python app.py` 한 번 실행 (`vectorstore/` 생성)
2. `.env`에 `OPENAI_API_KEY` 설정 (Q&A 생성·RAGAS 채점에 GPT-4o-mini 사용)
3. RAGAS만 추가 설치 필요: `pip install -r eval/requirements.txt`

## 실행 순서

```bash
# ① Q&A 라벨셋 생성 (IR·RAGAS 공통 입력) — 청크에서 30문항 자동 생성
python -m eval.build_dataset --n 30

# ② IR 메트릭 — Hit Rate@5 / MRR (ragas 불필요, 빠름)
python -m eval.eval_ir

# ③ RAGAS 4지표 — Agentic(재검색3회) vs 일반(재검색1회) 비교
python -m eval.eval_ragas            # 두 모드 모두
python -m eval.eval_ragas --mode agentic   # 한쪽만

# ④ 처리시간 — 전체/노드별 + async 절감 추정 (LLM 호출 포함, 실제 파이프라인 1회 실행)
python -m eval.measure_latency
```

결과는 모두 `eval/results/*.json`에 저장됩니다.

## 산출되는 수치 → 포트폴리오 매핑

| 스크립트 | 산출값 | 포트폴리오 📊 위치 |
|---|---|---|
| `eval_ragas.py` | Faithfulness, Answer Relevancy, Context Precision/Recall (전·후 비교) | "RAGAS … 실측 점수", "재검색 적용 전후 비교" |
| `eval_ir.py` | Hit Rate@5, MRR (일반 vs Agentic 비교) | "Hit Rate@5·MRR 실측값", "재검색 전후 비교" |
| `supervisor_compare.py` | Distributed vs Supervisor 라우팅 오버헤드(호출·토큰·시간) | "Distributed Pattern … 라우팅 오버헤드 제거" |
| `measure_latency.py` | 전체/노드별 시간, async 전환 절감 예상치 | "처리시간 측정 — async 전환 시 병렬 효과" |

## 주의
- **IR 정답 식별**: 검색 결과 청크의 `page_content`를 정답 청크와 **정확 매칭**합니다.
  Q&A를 같은 벡터스토어 청크로 생성하므로 매칭이 보장됩니다.
- **async 절감은 "추정치"**: 현재 LGES·CATL은 동기 구현이라 순차 실행됩니다.
  `max(lges, catl)` 기준으로 병렬화 시 예상 절감을 계산할 뿐, 실제 async 실측이 아닙니다.
  포트폴리오에 기재 시 "이론상 예상 절감"임을 명시하세요.
- **RAGAS 버전**: 0.2.x API 기준으로 작성했습니다. 메이저 버전이 다르면 import 경로 조정이 필요할 수 있습니다.
