# 설계: 3방향 쿼리(확증 편향 방지) 평가 + 개선

> 작성일: 2026-06-20
> 대상 기능: `agents/web_search.py` — 단일 쿼리 → 긍정/비판/중립 3방향 쿼리 자동 생성·병렬 검색·통합
> 목표: 이 기능이 **실제로 확증 편향을 줄이는지**를 기존 eval 프레임워크 철학(델타 비교·노이즈 통제·진단→개선→재측정)에 맞춰 수치로 검증하고, 약점이 드러나면 개선한다.

## 1. 배경 / 문제

`web_search.py`의 3방향 쿼리 기능은 **구현·설계는 되어 있으나 효과가 측정된 적이 없다**. 기존 `eval/`는 RAG 검색 품질(RAGAS·IR)·아키텍처·지연만 측정하며 web search는 평가 대상이 아니다(`eval/RESULTS.md`에 web 관련 수치 전무). `TODO.md`의 `P8-3 확증 편향 방지 동작 확인`도 미완료.

발표/보고서에 "긍정·비판·중립으로 쿼리를 나눴다"는 설명은 있어도 "그 결과 편향이 N만큼 줄었다"는 **수치 근거가 없다.** 이 스펙은 그 수치를 만든다.

## 2. 핵심 가설과 교란변수

**가설(H):** 기업 전략 자료/웹은 긍정 편향이 있어 단일 쿼리는 긍정 결과로 치우친다. 긍정/비판/중립 3방향은
- (H1) 검색 결과의 **관점 분포가 더 균형** 잡히고(부정 비중↑, 엔트로피↑),
- (H2) 단일 쿼리가 놓치는 **비판·리스크 정보를 더 회수**하며(리스크 recall↑),
- (H3) 최종 **보고서의 균형**(성과 vs 한계)이 개선된다.

**통제할 교란변수:** 3방향은 결과가 3배(3쿼리×5=15개)라 "단지 결과가 많아서" 더 다양해 보일 수 있다. → **볼륨 매칭 baseline**으로 격리한다.

**3개 비교군(arms):**

| arm | 쿼리 | 결과 수 | 역할 |
|---|---|---|---|
| `single_base` | 원본 단일 쿼리 | 5 | 기능이 없을 때의 순수 baseline |
| `single_volume` | 원본 단일 쿼리 | 15 | 볼륨 매칭 — "결과가 많아서" 효과 격리 |
| `three_persp` | 긍정/비판/중립 3개 | 5×3=15 | 평가 대상 기능(현행) |

- `three_persp − single_base` = 기능 전체 효과(볼륨+관점 혼재)
- **`three_persp − single_volume` = "관점 분할"의 순수 기여** (이 스펙의 헤드라인 델타)

## 3. 평가셋 (Topic Set)

프로덕션이 web_search에 넣는 쿼리는 3개뿐:
- market: `글로벌 배터리 전기차 시장 2025 캐즘 현황`
- LGES: `LG에너지솔루션 2025 포트폴리오 다각화 전략`
- CATL: `CATL 2025 포트폴리오 전략 나트륨이온 ESS 신흥시장`

n=3은 노이즈를 못 이긴다. **배터리 산업 기업전략 토픽 ~20개로 확장**한다(프로덕션 3개는 반드시 포함). focus_areas·comparison 차원에서 파생: ESS·나트륨이온·LFP·로봇/Physical AI 배터리·HEV·신흥시장(아프리카)·전고체·IRA/관세 정책·과잉공급·LGES vs CATL 원가 등. 모두 "기업 전략 자료가 긍정 편향되기 쉬운" 성격의 토픽이어야 한다(가설 적용 대상).

- 산출: `eval/results/web_topics.json` (커밋, 재현성). 각 항목 `{topic_id, base_query, subject, category}`.
- 작성 기준은 `eval/DATASET_CARD.md`에 섹션 추가.

## 4. 계층 1 — 검색 단계 평가 (`eval/eval_web_search.py`)

### 4.1 재현성 전략 (스냅샷)
웹은 라이브·비결정적 → 검색을 **1회 실행해 raw Tavily 결과를 스냅샷**(`eval/results/web_snapshots/<topic_id>.json`)으로 저장하고, 채점은 스냅샷 위에서 결정적(LLM-judge temp=0)으로 수행한다. 스냅샷이 있으면 재실행 시 검색을 건너뛴다(`--refresh`로 갱신).
- 스냅샷 단위: arm별·관점별 raw 결과(title/url/content/published_date/domain).
- **웹 변동 분산 측정:** 소수 토픽(3개)에 대해서만 K=3회 반복 검색해 지표 분산을 보고(전수 반복은 비용 과다). RESULTS.md에 "웹 변동성" 한계로 명시.

### 4.2 지표 (arm별 측정 → 델타)

1. **Stance 균형 (H1)**
   - LLM-judge(temp=0, 명시 루브릭)가 각 스니펫의 **대상(subject)에 대한 태도**를 `긍정|부정|중립`으로 분류.
   - `neg_share` = 부정 스니펫 / 전체 (핵심: 기업 토픽은 긍정 치우침 → 기능이 이를 올려야 함).
   - `balance_entropy` = H(분포)/log(3), 0~1 (1.0 = 긍정·부정·중립 완전 균형).
   - 예측: `three_persp`의 neg_share·entropy가 두 단일군보다 높다.

2. **소스 다양성 (교란 통제)**
   - `unique_domains` (절대 수), `unique_domains_per_result`(다양성 밀도, 볼륨 정규화).
   - `cross_perspective_dup_rate` = 관점 간 동일 URL 비율 (three_persp 전용) — 3쿼리가 실제로 갈라지는지/같은 출처로 붕괴하는지 진단.

3. **리스크 claim 커버리지 / recall (H2 — 핵심 payoff)**
   - 토픽별 **리스크 체크리스트**를 `eval/build_risk_checklist.py`로 생성 후 라벨 아티팩트(`eval/results/risk_checklist.json`)로 커밋. 항목 예: CATL → {지정학 리스크, 과잉공급/가동률, IRA/미국 시장 배제, IP 분쟁, LFP 마진 압박 …}.
   - arm별로 LLM-judge가 "각 체크리스트 항목이 스니펫 ≥1개에 의해 뒷받침되는가"를 판정 → `risk_recall` = 충족 항목 / 전체 항목.
   - 예측: `three_persp`(특히 비판 쿼리)의 risk_recall이 단일군보다 높다.

4. **쿼리 분기 건전성 (진단 지표)**
   - 현행 코드는 LLM 출력이 `긍정:/비판:/중립:` 형식을 안 지키면 **조용히 base_query로 3개 모두 폴백**한다(`web_search.py:39-47`). 이때 기능이 무력화됨.
   - `query_degeneration_rate` = (생성된 3쿼리 중 base_query와 동일하거나 서로 중복인 비율). 개선 1순위 후보.

### 4.3 출력
`eval/results/web_search_metrics.json` — arm별 지표 + 델타(`three_persp − single_base`, `three_persp − single_volume`) + per-topic 상세. 표본 분산 동반.

## 5. 계층 2 — 보고서 단계 ablation (`eval/eval_report_balance.py`)

### 5.1 비침습 ablation 스위치
`agents/web_search.py`에 환경변수 분기 추가(**프로덕션 기본 동작 불변**):
- `WEB_SEARCH_PERSPECTIVES` 미설정/`1`/`true` → 현행 3방향(기본).
- `0`/`false` → 단일 쿼리(base_query, max_results=15) 단발 검색 = ablation.

노드 코드(`research_analysis_agent.py` 등)는 건드리지 않고, eval이 env를 토글해 전체 파이프라인을 두 모드로 실행한다.

### 5.2 지표 (최종 보고서 대상)
1. `risk_section_claims` = "주요 리스크 및 한계" 섹션의 distinct 리스크 claim 수(LLM 추출·dedup).
2. `report_balance_score` = 보고서 전체 LLM-judge 균형 점수(성과/기대 vs 한계/리스크 균형 루브릭, 0~1).
3. `report_risk_recall` = 계층 1 체크리스트 대비 보고서가 다루는 리스크 비율.

### 5.3 노이즈 통제
report_generation 분산이 크므로(RESULTS.md §4) **arm당 N=3 실행** 후 평균±표준편차. 파이프라인 총 6회 실행(≈25분+API 비용). N은 인자로 조정 가능.

### 5.4 출력
`eval/results/report_balance.json` — arm별 평균±편차 + 델타.

## 6. 개선 루프 (조건부 — 사전 승인됨)

평가가 약점을 드러내면 수정 → 재측정으로 입증(델타). 예상 후보:
- `query_degeneration_rate` 높음 → 파싱 강건화(structured output 또는 retry) + 3쿼리 distinct 보장.
- `cross_perspective_dup_rate` 높음 → URL dedup 추가.
- 비판 쿼리가 `neg_share`를 못 올림 → 비판 쿼리 생성 프롬프트 강화.
- **`three_persp ≈ single_volume`** (순이득 없음)일 경우 → Agentic RAG 사례처럼 **정직하게 "측정된 순이득 없음 / 과설계 가능성" 보고**. 효과 입증이 아니라 효과 측정이 목적임을 견지.

개선은 production 기본 동작을 유지하면서 적용하고, 개선 전/후를 RESULTS.md에 v1→v2 형식으로 기록(기존 §1-3 패턴과 동일).

## 7. 산출물 / 파일 레이아웃

```
신규:
  eval/eval_web_search.py        # 계층1: 3-arm 검색 평가 + 스냅샷
  eval/eval_report_balance.py    # 계층2: 보고서 ablation
  eval/build_risk_checklist.py   # 토픽별 리스크 체크리스트 생성
  eval/results/web_topics.json
  eval/results/web_snapshots/<topic_id>.json
  eval/results/risk_checklist.json
  eval/results/web_search_metrics.json
  eval/results/report_balance.json

수정:
  agents/web_search.py           # WEB_SEARCH_PERSPECTIVES 분기(+ 개선 루프 결과 반영)
  eval/RESULTS.md                # 새 §5 "Web Search 확증 편향 방지" 결과
  eval/README.md                 # 실행 순서 ⑥⑦ 추가
  eval/DATASET_CARD.md           # 토픽셋·리스크 체크리스트 카드
  TODO.md                        # P8-3 체크
```

## 8. 수용 기준 (Acceptance Criteria)

- [ ] 3-arm 설계로 검색 단계 지표가 산출되고, **볼륨 매칭 델타(`three_persp − single_volume`)**가 명시된다.
- [ ] stance 균형·소스 다양성·리스크 recall·degeneration rate 4개 지표가 수치로 보고된다.
- [ ] 스냅샷으로 채점이 재현 가능하고, 웹 변동성이 소표본 K=3 분산으로 정량화된다.
- [ ] 보고서 ablation(env 토글, N=3)으로 최종 보고서 균형 델타가 보고된다.
- [ ] 프로덕션 기본 동작(3방향)은 변하지 않는다(env 미설정 시 현행과 동일).
- [ ] RESULTS.md에 결과·해석·한계가 기존 톤(정직한 델타 해석)으로 기록된다.
- [ ] 효과가 없으면 "없음"을, 개선이 효과를 냈으면 v1→v2 델타를 정직하게 기록한다.

## 9. 비목표 (Non-goals)

- Tavily/검색엔진 자체 품질 벤치마크가 아니다(고정 백엔드 가정).
- 사람 라벨링 기반 골드셋 구축이 아니다(LLM-judge temp=0 + 커밋된 체크리스트로 재현성 확보; 한계로 명시).
- multi-hop/외부 도메인 일반화 보장 아님(배터리 산업 토픽 한정).
- 프로덕션 노드 흐름 리팩터링 아님(env 스위치만 추가, 비침습).
