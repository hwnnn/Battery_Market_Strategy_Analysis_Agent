# Web Search 확증편향 평가 + 개선 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `agents/web_search.py`의 긍정/비판/중립 3방향 쿼리 기능이 확증 편향을 실제로 줄이는지 검색·보고서 두 계층에서 수치로 측정하고, 약점이 드러나면 개선한다.

**Architecture:** 볼륨 교란을 통제한 3-arm 비교(single_base / single_volume / three_persp)로 검색 단계 4지표(stance 균형·소스 다양성·리스크 recall·degeneration rate)를 스냅샷 기반으로 재현 가능하게 측정한다. 보고서 단계는 `WEB_SEARCH_PERSPECTIVES` 환경변수로 비침습 ablation을 걸어 전체 파이프라인을 두 모드로 실행하고 최종 보고서 균형을 비교한다. 순수 지표/파서 함수는 TDD로, API 호출 오케스트레이터는 소표본 스모크 실행으로 검증한다.

**Tech Stack:** Python 3.11, LangChain(OpenAI GPT-4o-mini, `get_llm`), Tavily, LangGraph, pytest(신규, 순수 함수 단위테스트용).

## Global Constraints

- 프로덕션 기본 동작 불변: `WEB_SEARCH_PERSPECTIVES` 미설정 시 현행 3방향 동작과 100% 동일해야 한다.
- LLM 호출은 전부 `agents.llm_config.get_llm(temperature=0.0)` 사용 (모델 gpt-4o-mini, 결정성 위해 temp=0).
- 환경변수 파싱 idiom은 기존과 통일: `os.getenv(NAME, default).strip().lower() in ("1", "true", "yes", "on")` (참조: `agents/rag_tool.py:33`).
- 결과 산출물은 전부 `eval/results/`에 JSON으로 저장 (기존 패턴). 한글 보존 `ensure_ascii=False, indent=2`.
- eval 스크립트는 `python -m eval.<name>` 로 실행 가능해야 하며 `sys.path.insert(0, <repo_root>)` 부트스트랩을 포함한다 (참조: `eval/eval_ir.py:26`).
- 모든 지표는 **델타(arm 간 차이)** 로 해석한다. `three_persp − single_volume` 가 헤드라인.
- 정직성 원칙: 효과가 없으면 "없음"을, 개선 효과는 v1→v2 델타로 기록한다 (기존 `eval/RESULTS.md` 톤 유지).
- 사전 조건: FAISS 인덱스(`vectorstore/`)가 있어야 보고서 단계 실행 가능 → 없으면 `python app.py` 1회. `.env`에 `OPENAI_API_KEY`·`TAVILY_API_KEY` 필요.

---

## File Structure

```
신규:
  eval/web_metrics.py            # 순수 지표 함수 (API 미사용) — TDD
  eval/test_web_metrics.py       # web_metrics + 순수 파서 단위테스트
  eval/judges.py                 # LLM-judge (stance/risk/balance) + 순수 파서
  eval/build_risk_checklist.py   # 토픽별 리스크 체크리스트 생성 스크립트
  eval/eval_web_search.py        # 계층1: 3-arm 검색 평가 + 스냅샷
  eval/eval_report_balance.py    # 계층2: 보고서 ablation
  eval/results/web_topics.json   # 평가 토픽셋 (~20, 수기 작성)
  eval/results/risk_checklist.json     # build_risk_checklist.py 산출
  eval/results/web_snapshots/<topic_id>.json   # 검색 raw 스냅샷
  eval/results/web_search_metrics.json # 계층1 결과
  eval/results/report_balance.json     # 계층2 결과

수정:
  agents/web_search.py           # 헬퍼 노출 + WEB_SEARCH_PERSPECTIVES 토글 (개선 루프 결과 반영)
  eval/requirements.txt          # pytest 추가
  eval/README.md                 # 실행 순서 ⑥⑦ 추가
  eval/DATASET_CARD.md           # 토픽셋·리스크 체크리스트 카드
  eval/RESULTS.md                # §5 Web Search 결과
  TODO.md                        # P8-3 체크
```

---

### Task 1: 순수 지표 모듈 `eval/web_metrics.py` (TDD)

API를 호출하지 않는 결정적 지표 함수. 결과 정확성이 조용히 깨지면 안 되는 핵심이라 TDD로 작성.

**Files:**
- Create: `eval/web_metrics.py`
- Create: `eval/test_web_metrics.py`
- Modify: `eval/requirements.txt` (pytest 추가)

**Interfaces:**
- Consumes: (없음 — 표준 라이브러리만)
- Produces:
  - `STANCES = ("긍정", "부정", "중립")`
  - `domain_of(url: str) -> str`
  - `stance_counts(stances: list[str]) -> dict[str, int]`  # 키는 STANCES 3개 고정
  - `neg_share(stances: list[str]) -> float`
  - `balance_entropy(stances: list[str]) -> float`  # 정규화 0..1, 빈 입력 0.0
  - `unique_domain_count(urls: list[str]) -> int`
  - `cross_perspective_dup_rate(per_persp_urls: dict[str, list[str]]) -> float`  # 0..1
  - `recall(hits: list[bool]) -> float`  # 빈 입력 0.0
  - `is_degenerate(base_query: str, queries: dict[str, str]) -> bool`

- [ ] **Step 1: pytest 의존성 추가**

`eval/requirements.txt` 끝에 추가:
```
pytest>=7.0.0
```

- [ ] **Step 2: 실패하는 테스트 작성**

Create `eval/test_web_metrics.py`:
```python
import math
import pytest

from eval.web_metrics import (
    domain_of, stance_counts, neg_share, balance_entropy,
    unique_domain_count, cross_perspective_dup_rate, recall, is_degenerate,
)


def test_domain_of_strips_scheme_and_path():
    assert domain_of("https://www.reuters.com/business/x?y=1") == "reuters.com"
    assert domain_of("http://catl.com/") == "catl.com"
    assert domain_of("not a url") == "not a url"


def test_stance_counts_has_all_three_keys():
    c = stance_counts(["긍정", "긍정", "부정"])
    assert c == {"긍정": 2, "부정": 1, "중립": 0}


def test_neg_share():
    assert neg_share(["긍정", "부정", "부정", "중립"]) == 0.5
    assert neg_share([]) == 0.0


def test_balance_entropy_uniform_is_one():
    # 완전 균형(각 1/3) → 정규화 엔트로피 1.0
    assert balance_entropy(["긍정", "부정", "중립"]) == pytest.approx(1.0)


def test_balance_entropy_skewed_is_low():
    # 전부 긍정 → 0.0
    assert balance_entropy(["긍정", "긍정", "긍정"]) == pytest.approx(0.0)
    assert balance_entropy([]) == 0.0


def test_unique_domain_count_dedups():
    urls = ["https://a.com/1", "https://a.com/2", "https://b.com/x"]
    assert unique_domain_count(urls) == 2


def test_cross_perspective_dup_rate():
    # 긍정/비판이 같은 URL 1개 공유 → 중복
    per = {
        "긍정": ["https://a.com/1", "https://b.com/2"],
        "비판": ["https://a.com/1", "https://c.com/3"],
        "중립": ["https://d.com/4"],
    }
    # 전체 5개 URL 중 a.com/1 이 2회 등장 → 중복 슬롯 1 / 전체 5 = 0.2
    assert cross_perspective_dup_rate(per) == pytest.approx(0.2)
    assert cross_perspective_dup_rate({"긍정": [], "비판": [], "중립": []}) == 0.0


def test_recall():
    assert recall([True, False, True, True]) == 0.75
    assert recall([]) == 0.0


def test_is_degenerate_detects_fallback():
    base = "CATL 전략"
    # 정상: 3개가 모두 base와 다르고 서로 다름
    ok = {"긍정": "CATL 성과", "비판": "CATL 리스크", "중립": "CATL 현황"}
    assert is_degenerate(base, ok) is False
    # 폴백: 전부 base와 동일
    bad = {"긍정": base, "비판": base, "중립": base}
    assert is_degenerate(base, bad) is True
    # 부분 붕괴: 두 개가 동일
    partial = {"긍정": "CATL 성과", "비판": "CATL 성과", "중립": "CATL 현황"}
    assert is_degenerate(base, partial) is True
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python -m pytest eval/test_web_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.web_metrics'`

- [ ] **Step 4: 최소 구현 작성**

Create `eval/web_metrics.py`:
```python
"""
Web Search 평가용 순수 지표 함수 (API 미사용, 결정적).
스냅샷에서 추출한 stance 라벨/URL 리스트를 입력으로 받아 지표를 계산한다.
"""

import math
from urllib.parse import urlparse

STANCES = ("긍정", "부정", "중립")


def domain_of(url: str) -> str:
    """URL → 등록 도메인(www 제거). 파싱 실패 시 원본 반환."""
    netloc = urlparse(url).netloc
    if not netloc:
        return url
    return netloc[4:] if netloc.startswith("www.") else netloc


def stance_counts(stances: list[str]) -> dict[str, int]:
    """STANCES 3키를 항상 포함하는 카운트 딕셔너리."""
    counts = {s: 0 for s in STANCES}
    for s in stances:
        if s in counts:
            counts[s] += 1
    return counts


def neg_share(stances: list[str]) -> float:
    """부정 스니펫 비율."""
    if not stances:
        return 0.0
    return round(stance_counts(stances)["부정"] / len(stances), 4)


def balance_entropy(stances: list[str]) -> float:
    """긍정/부정/중립 분포의 정규화 섀넌 엔트로피(0..1). 1.0=완전 균형."""
    n = len(stances)
    if n == 0:
        return 0.0
    h = 0.0
    for c in stance_counts(stances).values():
        if c:
            p = c / n
            h -= p * math.log(p)
    return round(h / math.log(len(STANCES)), 4)


def unique_domain_count(urls: list[str]) -> int:
    return len({domain_of(u) for u in urls})


def cross_perspective_dup_rate(per_persp_urls: dict[str, list[str]]) -> float:
    """관점 간 동일 URL 중복 슬롯 비율 = (전체 슬롯 − 고유 URL 수) / 전체 슬롯."""
    all_urls = [u for urls in per_persp_urls.values() for u in urls]
    if not all_urls:
        return 0.0
    return round((len(all_urls) - len(set(all_urls))) / len(all_urls), 4)


def recall(hits: list[bool]) -> float:
    if not hits:
        return 0.0
    return round(sum(1 for h in hits if h) / len(hits), 4)


def is_degenerate(base_query: str, queries: dict[str, str]) -> bool:
    """3방향 쿼리가 base로 폴백했거나 서로 중복이면 True(기능 무력화)."""
    vals = [q.strip() for q in queries.values()]
    if any(v == base_query.strip() for v in vals):
        return True
    return len(set(vals)) < len(vals)
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `python -m pytest eval/test_web_metrics.py -v`
Expected: PASS (10 passed)

- [ ] **Step 6: 커밋**

```bash
git add eval/web_metrics.py eval/test_web_metrics.py eval/requirements.txt
git commit -m "feat(eval): web search 평가용 순수 지표 모듈 + 테스트"
```

---

### Task 2: `web_search.py` 헬퍼 노출 + 비침습 ablation 토글

평가가 raw 결과를 다루고 토글로 ablation 하려면 내부를 헬퍼로 노출해야 한다. **동작 변경 금지** — degeneration을 측정해야 하므로 기존 파싱 로직은 그대로 둔다(개선은 Task 9).

**Files:**
- Modify: `agents/web_search.py`
- Modify: `eval/test_web_metrics.py` (토글/헬퍼 테스트 추가) — 또는 신규 `eval/test_web_search_helpers.py`

**Interfaces:**
- Consumes: `agents.llm_config.get_llm`
- Produces:
  - `build_queries(base_query: str, llm) -> dict[str, str]`  # 키 긍정/비판/중립 (기존 `_build_three_queries`를 공개 이름으로 노출, 동작 동일)
  - `tavily_search(query: str, max_results: int = 5) -> list[dict]`  # 각 원소 {title,url,content,published,domain}
  - `perspectives_enabled() -> bool`  # WEB_SEARCH_PERSPECTIVES, 기본 True
  - `web_search(base_query, llm, max_results=5) -> str`  # 토글 분기 추가, 기본 경로는 기존과 동일 문자열

- [ ] **Step 1: 실패하는 테스트 작성**

Create `eval/test_web_search_helpers.py`:
```python
import os
import pytest

from agents.web_search import perspectives_enabled


def test_perspectives_enabled_default_true(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_PERSPECTIVES", raising=False)
    assert perspectives_enabled() is True  # 프로덕션 기본 = 3방향


@pytest.mark.parametrize("val,expected", [
    ("0", False), ("false", False), ("FALSE", False), ("no", False),
    ("1", True), ("true", True), ("on", True),
])
def test_perspectives_enabled_env(monkeypatch, val, expected):
    monkeypatch.setenv("WEB_SEARCH_PERSPECTIVES", val)
    assert perspectives_enabled() is expected
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest eval/test_web_search_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name 'perspectives_enabled'`

- [ ] **Step 3: `web_search.py` 수정**

`agents/web_search.py`에서 (1) `_build_three_queries`를 `build_queries`로 공개(기존 호출부 호환 위해 별칭 유지), (2) `tavily_search` 헬퍼 추출, (3) `perspectives_enabled()` 추가, (4) `web_search()` 토글 분기. 기존 import 블록 아래에 추가하고 본문을 아래로 교체:

```python
def perspectives_enabled() -> bool:
    """WEB_SEARCH_PERSPECTIVES 환경변수로 3방향 on/off (기본 on)."""
    return os.getenv("WEB_SEARCH_PERSPECTIVES", "true").strip().lower() in (
        "1", "true", "yes", "on"
    )


def build_queries(base_query: str, llm: ChatOpenAI) -> Dict[str, str]:
    """기본 쿼리에서 긍정·비판·중립 3방향 쿼리를 LLM으로 생성한다.
    (기존 _build_three_queries 와 동일 동작 — 공개 이름)"""
    return _build_three_queries(base_query, llm)


def tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """단일 쿼리 Tavily 검색 → 정규화된 결과 리스트.
    각 원소: {title, url, content, published, domain}"""
    from eval.web_metrics import domain_of  # 지연 import (프로덕션 경로 비의존)

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY가 설정되지 않았습니다. .env를 확인하세요.")
    client = TavilyClient(api_key=api_key)
    resp = client.search(query=query, max_results=max_results, search_depth="advanced")
    out = []
    for r in resp.get("results", []):
        url = r.get("url", "")
        out.append({
            "title": r.get("title", "제목 없음"),
            "url": url,
            "content": r.get("content", "내용 없음")[:500],
            "published": r.get("published_date", "날짜 미상"),
            "domain": domain_of(url),
        })
    return out
```

그리고 기존 `web_search()` 본문을 토글 분기로 교체 (3방향 경로의 출력 포맷은 기존과 동일 유지):

```python
def web_search(base_query: str, llm: ChatOpenAI, max_results: int = 5) -> str:
    """확증 편향 방지 웹 검색 Tool.
    WEB_SEARCH_PERSPECTIVES=off 면 단일 쿼리(ablation), 기본은 3방향."""
    if not perspectives_enabled():
        # ablation: 볼륨을 3방향과 맞추기 위해 max_results*3
        results = tavily_search(base_query, max_results=max_results * 3)
        section = f"\n=== [단일 쿼리(ablation)] 검색 쿼리: {base_query} ===\n"
        for i, r in enumerate(results, 1):
            section += (f"\n[{i}] {r['title']}\n출처: {r['url']}\n"
                        f"날짜: {r['published']}\n내용: {r['content']}\n")
        return section

    queries = build_queries(base_query, llm)
    all_results: List[str] = []
    for perspective, query in queries.items():
        try:
            results = tavily_search(query, max_results=max_results)
            section = f"\n=== [{perspective} 관점] 검색 쿼리: {query} ===\n"
            for i, r in enumerate(results, 1):
                section += (f"\n[{i}] {r['title']}\n출처: {r['url']}\n"
                            f"날짜: {r['published']}\n내용: {r['content']}\n")
            all_results.append(section)
        except Exception as e:
            all_results.append(f"\n=== [{perspective} 관점] 검색 실패: {str(e)} ===\n")
    return "\n".join(all_results)
```

주의: 기존 `_build_three_queries` 함수 정의는 그대로 둔다(`build_queries`가 위임). 파일 상단 docstring에 토글 설명 한 줄 추가.

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest eval/test_web_search_helpers.py -v`
Expected: PASS

- [ ] **Step 5: 회귀 확인 — 기본 경로 import/동작 무결**

Run: `python -c "import agents.web_search as w; print(w.perspectives_enabled()); print(hasattr(w,'build_queries'), hasattr(w,'tavily_search'))"`
Expected: `True` / `True True` (env 미설정 시 기본 3방향)

- [ ] **Step 6: 커밋**

```bash
git add agents/web_search.py eval/test_web_search_helpers.py
git commit -m "feat(web_search): 평가용 헬퍼 노출 + WEB_SEARCH_PERSPECTIVES ablation 토글(기본 동작 불변)"
```

---

### Task 3: 평가 토픽셋 `web_topics.json` + 데이터 카드

프로덕션 3쿼리는 노이즈를 못 이김 → 배터리 기업전략 토픽 ~20개로 확장. 수기 데이터 파일.

**Files:**
- Create: `eval/results/web_topics.json`
- Modify: `eval/DATASET_CARD.md`

**Interfaces:**
- Produces: `web_topics.json` = `list[{topic_id, base_query, subject, category}]`. `eval_web_search.py`·`build_risk_checklist.py`가 소비.

- [ ] **Step 1: 토픽 파일 작성**

Create `eval/results/web_topics.json` — 프로덕션 3개 + 파생 토픽 합쳐 20개. 각 항목 `subject`는 stance/risk 판정의 대상(기업·기술·시장). 예시 스키마와 시작 항목(나머지는 동일 형식으로 총 20개 채움):
```json
[
  {"topic_id": "prod_market", "base_query": "글로벌 배터리 전기차 시장 2025 캐즘 현황", "subject": "글로벌 배터리·전기차 시장", "category": "market"},
  {"topic_id": "prod_lges", "base_query": "LG에너지솔루션 2025 포트폴리오 다각화 전략", "subject": "LG에너지솔루션(LGES)", "category": "company"},
  {"topic_id": "prod_catl", "base_query": "CATL 2025 포트폴리오 전략 나트륨이온 ESS 신흥시장", "subject": "CATL", "category": "company"},
  {"topic_id": "lges_ess", "base_query": "LG에너지솔루션 ESS 사업 전략 전망", "subject": "LG에너지솔루션 ESS 사업", "category": "company_segment"},
  {"topic_id": "lges_robot", "base_query": "LG에너지솔루션 로봇 Physical AI 배터리 전략", "subject": "LG에너지솔루션 로봇용 배터리", "category": "company_segment"},
  {"topic_id": "lges_hev", "base_query": "LG에너지솔루션 HEV 하이브리드 배터리 대응", "subject": "LG에너지솔루션 HEV 배터리", "category": "company_segment"},
  {"topic_id": "catl_sodium", "base_query": "CATL 나트륨이온 배터리 상업화 전략", "subject": "CATL 나트륨이온 배터리", "category": "company_segment"},
  {"topic_id": "catl_africa", "base_query": "CATL ESS 아프리카 신흥시장 진출 전략", "subject": "CATL 신흥시장 ESS", "category": "company_segment"},
  {"topic_id": "catl_lfp", "base_query": "CATL LFP 배터리 원가 경쟁력 전략", "subject": "CATL LFP 원가 경쟁력", "category": "company_segment"},
  {"topic_id": "solid_state", "base_query": "전고체 배터리 상업화 전망 2025", "subject": "전고체 배터리 상업화", "category": "technology"},
  {"topic_id": "sodium_market", "base_query": "나트륨이온 배터리 시장 전망", "subject": "나트륨이온 배터리 시장", "category": "technology"},
  {"topic_id": "ess_market", "base_query": "글로벌 ESS 에너지저장장치 시장 성장 전망", "subject": "글로벌 ESS 시장", "category": "market"},
  {"topic_id": "ira_policy", "base_query": "미국 IRA 인플레이션감축법 배터리 영향", "subject": "미국 IRA 배터리 정책", "category": "policy"},
  {"topic_id": "eu_tariff", "base_query": "유럽 중국 배터리 관세 규제 영향", "subject": "유럽 배터리 관세 규제", "category": "policy"},
  {"topic_id": "overcapacity", "base_query": "배터리 산업 과잉공급 가동률 문제", "subject": "배터리 과잉공급", "category": "market"},
  {"topic_id": "lges_vs_catl", "base_query": "LG에너지솔루션 CATL 경쟁력 비교", "subject": "LGES vs CATL 경쟁", "category": "comparison"},
  {"topic_id": "ev_chasm", "base_query": "전기차 캐즘 수요 둔화 2025 전망", "subject": "전기차 캐즘", "category": "market"},
  {"topic_id": "kr_battery_policy", "base_query": "한국 배터리 산업 지원 정책 전략", "subject": "한국 배터리 정책", "category": "policy"},
  {"topic_id": "battery_recycling", "base_query": "배터리 재활용 사업 시장 전망", "subject": "배터리 재활용 시장", "category": "technology"},
  {"topic_id": "samsung_sdi", "base_query": "삼성SDI 배터리 포트폴리오 전략 2025", "subject": "삼성SDI", "category": "company"}
]
```

- [ ] **Step 2: JSON 유효성·개수 검증**

Run: `python -c "import json; d=json.load(open('eval/results/web_topics.json', encoding='utf-8')); assert len(d)==20; assert all({'topic_id','base_query','subject','category'} <= set(x) for x in d); print('topics OK', len(d))"`
Expected: `topics OK 20`

- [ ] **Step 3: 데이터 카드 갱신**

`eval/DATASET_CARD.md` 끝에 섹션 추가 (실제 파일 톤에 맞춰):
```markdown

## Web Search 평가 토픽셋 (web_topics.json)

- 목적: 3방향 쿼리(긍정/비판/중립)의 확증편향 저감 효과 측정용 base 쿼리 모음.
- 구성: 총 20개. 프로덕션 실제 쿼리 3개(`prod_*`) + 배터리 기업전략 파생 토픽 17개.
- 카테고리: market / company / company_segment / technology / policy / comparison.
- 선정 기준: "기업 전략 자료가 긍정 편향되기 쉬운" 성격의 토픽(가설 적용 대상)으로 한정.
- 각 항목 `subject`는 stance/risk 판정의 대상 엔티티(기업·기술·시장).
- 한계: 배터리 산업 한정, 외부 도메인 일반화 보장 아님.
```

- [ ] **Step 4: 커밋**

```bash
git add eval/results/web_topics.json eval/DATASET_CARD.md
git commit -m "feat(eval): web search 평가 토픽셋 20개 + 데이터 카드"
```

---

### Task 4: 리스크 체크리스트 생성기 `build_risk_checklist.py`

토픽별 "알려진 리스크 항목" 라벨셋을 LLM으로 생성해 커밋(리스크 recall의 정답지).

**Files:**
- Create: `eval/build_risk_checklist.py`
- Create (산출): `eval/results/risk_checklist.json`

**Interfaces:**
- Consumes: `web_topics.json`, `agents.llm_config.get_llm`
- Produces: `risk_checklist.json` = `dict[topic_id, list[str]]` (토픽당 4~7개 리스크 항목)

- [ ] **Step 1: 스크립트 작성**

Create `eval/build_risk_checklist.py`:
```python
"""
토픽별 리스크/비판 체크리스트 생성 (리스크 recall 평가의 정답지).
web_topics.json 의 각 subject에 대해 '잘 알려진 리스크·한계·논쟁점'을
GPT-4o-mini로 4~7개 추출해 risk_checklist.json 으로 저장한다.

생성물은 라벨 아티팩트로 '커밋'하여 재현성을 확보한다(매번 재생성 금지).

실행:
    python -m eval.build_risk_checklist
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from agents.llm_config import get_llm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TOPICS = os.path.join(RESULTS_DIR, "web_topics.json")
OUT = os.path.join(RESULTS_DIR, "risk_checklist.json")

SYS = (
    "당신은 배터리 산업 애널리스트입니다. 주어진 대상에 대해 '실제로 보도·지적된' "
    "리스크·한계·논쟁점을 4~7개 한국어로 나열하세요. 홍보성 장점이 아니라 부정적 측면만. "
    "각 항목은 8~20자의 짧은 명사구로, 서로 중복되지 않게.\n"
    '반드시 JSON 배열로만 출력: ["항목1", "항목2", ...]'
)


def _gen(llm, subject: str) -> list[str]:
    raw = llm.invoke([
        SystemMessage(content=SYS),
        HumanMessage(content=f"대상: {subject}"),
    ]).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
    try:
        arr = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        return [s.strip() for s in arr if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, ValueError):
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="앞 N개 토픽만(스모크)")
    args = parser.parse_args()

    topics = json.load(open(TOPICS, encoding="utf-8"))
    if args.limit:
        topics = topics[:args.limit]
    llm = get_llm(temperature=0.0)

    checklist = {}
    for i, t in enumerate(topics, 1):
        items = _gen(llm, t["subject"])
        checklist[t["topic_id"]] = items
        print(f"  [{i}/{len(topics)}] {t['topic_id']}: {len(items)}개 — {items}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # --limit 스모크 시 기존 항목 보존 병합
    if args.limit and os.path.exists(OUT):
        existing = json.load(open(OUT, encoding="utf-8"))
        existing.update(checklist)
        checklist = existing
    json.dump(checklist, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장 → {OUT} ({len(checklist)} 토픽)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 스모크 실행(2 토픽)**

Run: `python -m eval.build_risk_checklist --limit 2`
Expected: 2개 토픽에 대해 각 4~7개 리스크 항목 출력, `risk_checklist.json` 생성. 항목이 부정적 측면인지 눈으로 확인.

- [ ] **Step 3: 전체 생성**

Run: `python -m eval.build_risk_checklist`
Expected: 20개 토픽 전부 항목 생성, 빈 리스트 없음.

- [ ] **Step 4: 커밋**

```bash
git add eval/build_risk_checklist.py eval/results/risk_checklist.json
git commit -m "feat(eval): 토픽별 리스크 체크리스트 생성기 + 라벨 아티팩트"
```

---

### Task 5: LLM-judge 모듈 `eval/judges.py` (순수 파서는 TDD)

stance 분류·리스크 충족 판정·보고서 리스크 추출·균형 점수. LLM 호출부는 스모크, 출력 파서는 단위테스트.

**Files:**
- Create: `eval/judges.py`
- Modify: `eval/test_web_metrics.py` 또는 신규 `eval/test_judges.py` (파서 테스트)

**Interfaces:**
- Consumes: `agents.llm_config.get_llm`, `eval.web_metrics.STANCES`
- Produces:
  - `parse_stance_label(raw: str) -> str`  # 순수, STANCES 중 하나(미상이면 "중립")
  - `parse_score(raw: str) -> float`  # 순수, 0..1 (실패 시 0.0)
  - `classify_stance(snippet: str, subject: str, llm) -> str`
  - `risk_covered(risk_item: str, snippets: list[str], llm) -> bool`
  - `extract_risk_claims(report: str, llm) -> list[str]`
  - `report_balance_score(report: str, llm) -> float`

- [ ] **Step 1: 파서 실패 테스트 작성**

Create `eval/test_judges.py`:
```python
import pytest
from eval.judges import parse_stance_label, parse_score


@pytest.mark.parametrize("raw,expected", [
    ("긍정", "긍정"),
    ("이 글의 태도는 부정입니다", "부정"),
    ("중립적", "중립"),
    ("판단 불가", "중립"),   # 미상 → 중립 폴백
    ("", "중립"),
])
def test_parse_stance_label(raw, expected):
    assert parse_stance_label(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("0.8", 0.8),
    ("점수: 0.65", 0.65),
    ("1", 1.0),
    ("균형 점수는 0.4 정도", 0.4),
    ("없음", 0.0),
])
def test_parse_score(raw, expected):
    assert parse_score(raw) == pytest.approx(expected)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest eval/test_judges.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.judges'`

- [ ] **Step 3: `judges.py` 구현**

Create `eval/judges.py`:
```python
"""
LLM-judge 모듈 — web search 평가용 (temp=0).
순수 파서(parse_*)는 단위테스트, LLM 호출부는 스모크 검증.
"""
import re
from langchain_core.messages import HumanMessage, SystemMessage
from eval.web_metrics import STANCES

_STANCE_SYS = (
    "당신은 텍스트 태도 분류기입니다. 주어진 검색 스니펫이 '대상'에 대해 어떤 태도를 "
    "보이는지 한 단어로만 답하세요: 긍정 / 부정 / 중립.\n"
    "- 긍정: 성과·기대·강점 강조\n- 부정: 리스크·한계·문제점·비판\n- 중립: 사실·현황 위주\n"
    "반드시 '긍정', '부정', '중립' 중 한 단어만 출력."
)
_RISK_SYS = (
    "주어진 '리스크 항목'이 아래 검색 스니펫들에 의해 실제로 언급/뒷받침되는지 판정하세요. "
    "뒷받침되면 'YES', 아니면 'NO' 한 단어만 출력."
)
_EXTRACT_SYS = (
    "다음 보고서에서 언급된 '리스크·한계·부정적 요인'을 중복 없이 짧은 명사구로 나열하세요. "
    "성과·장점은 제외.\n반드시 JSON 배열로만 출력: [\"항목1\", ...]"
)
_BALANCE_SYS = (
    "다음 보고서가 '성과·기대'와 '한계·리스크'를 얼마나 균형 있게 다루는지 0.0~1.0 점수로 "
    "평가하세요. 1.0=완전 균형, 0.0=한쪽으로 완전 치우침. 숫자만 출력."
)


def parse_stance_label(raw: str) -> str:
    for s in STANCES:
        if s in raw:
            return s
    return "중립"


def parse_score(raw: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", raw)
    if not m:
        return 0.0
    return round(min(1.0, max(0.0, float(m.group()))), 4)


def classify_stance(snippet: str, subject: str, llm) -> str:
    raw = llm.invoke([
        SystemMessage(content=_STANCE_SYS),
        HumanMessage(content=f"대상: {subject}\n\n스니펫:\n{snippet[:800]}"),
    ]).content
    return parse_stance_label(raw)


def risk_covered(risk_item: str, snippets: list[str], llm) -> bool:
    joined = "\n---\n".join(s[:400] for s in snippets)
    raw = llm.invoke([
        SystemMessage(content=_RISK_SYS),
        HumanMessage(content=f"리스크 항목: {risk_item}\n\n검색 스니펫들:\n{joined}"),
    ]).content
    return "YES" in raw.upper()


def extract_risk_claims(report: str, llm) -> list[str]:
    import json
    raw = llm.invoke([
        SystemMessage(content=_EXTRACT_SYS),
        HumanMessage(content=report[:12000]),
    ]).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
    try:
        arr = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        return [s.strip() for s in arr if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, ValueError):
        return []


def report_balance_score(report: str, llm) -> float:
    raw = llm.invoke([
        SystemMessage(content=_BALANCE_SYS),
        HumanMessage(content=report[:12000]),
    ]).content
    return parse_score(raw)
```

- [ ] **Step 4: 파서 테스트 통과 확인**

Run: `python -m pytest eval/test_judges.py -v`
Expected: PASS

- [ ] **Step 5: LLM-judge 스모크**

Run:
```bash
python -c "
from agents.llm_config import get_llm
from eval.judges import classify_stance, report_balance_score
llm = get_llm(0.0)
print(classify_stance('CATL은 과잉공급과 가동률 하락 위기에 직면했다', 'CATL', llm))
print(classify_stance('CATL이 사상 최대 실적과 점유율 1위를 기록했다', 'CATL', llm))
"
```
Expected: 첫 줄 `부정`, 둘째 줄 `긍정` (대략 — 분류기 동작 확인).

- [ ] **Step 6: 커밋**

```bash
git add eval/judges.py eval/test_judges.py
git commit -m "feat(eval): LLM-judge 모듈(stance/risk/balance) + 파서 테스트"
```

---

### Task 6: 계층1 검색 평가 `eval/eval_web_search.py`

3-arm 검색 → 스냅샷 → 지표(stance 균형·소스 다양성·리스크 recall·degeneration) → JSON.

**Files:**
- Create: `eval/eval_web_search.py`
- Create (산출): `eval/results/web_snapshots/<topic_id>.json`, `eval/results/web_search_metrics.json`

**Interfaces:**
- Consumes: `agents.web_search.{build_queries, tavily_search}`, `eval.web_metrics.*`, `eval.judges.{classify_stance, risk_covered}`, `web_topics.json`, `risk_checklist.json`, `get_llm`
- Produces: `web_search_metrics.json` = arm별 평균 지표 + 델타 + per-topic 상세

스냅샷 스키마(`web_snapshots/<topic_id>.json`):
```json
{"topic_id": "...", "base_query": "...", "subject": "...",
 "queries": {"긍정": "...", "비판": "...", "중립": "..."},
 "arms": {"single_base": [<result>...],
          "single_volume": [<result>...],
          "three_persp": {"긍정": [<result>...], "비판": [...], "중립": [...]}}}
```
`<result>` = `{title,url,content,published,domain}` (tavily_search 출력).

- [ ] **Step 1: 스냅샷 수집부 작성**

Create `eval/eval_web_search.py` (1차: 스냅샷 빌드):
```python
"""
계층1 — Web Search 3방향 쿼리 평가 (검색 단계).
3-arm 비교로 볼륨 교란을 통제하고 stance 균형·소스 다양성·리스크 recall·
degeneration rate 를 측정한다. 검색 결과는 스냅샷으로 저장해 채점을 재현 가능하게 한다.

  arms: single_base(원쿼리×5) / single_volume(원쿼리×15) / three_persp(3쿼리×5)
  헤드라인 델타: three_persp − single_volume (관점 분할의 순수 기여)

실행:
    python -m eval.eval_web_search                 # 스냅샷 있으면 재사용
    python -m eval.eval_web_search --refresh       # 검색 다시 수행
    python -m eval.eval_web_search --limit 2       # 앞 2토픽만(스모크)
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_config import get_llm
from agents.web_search import build_queries, tavily_search
from eval import web_metrics as M
from eval.judges import classify_stance, risk_covered

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SNAP_DIR = os.path.join(RESULTS_DIR, "web_snapshots")
TOPICS = os.path.join(RESULTS_DIR, "web_topics.json")
CHECKLIST = os.path.join(RESULTS_DIR, "risk_checklist.json")
OUT = os.path.join(RESULTS_DIR, "web_search_metrics.json")
MAX_RESULTS = 5


def _snap_path(topic_id):
    return os.path.join(SNAP_DIR, f"{topic_id}.json")


def build_snapshot(topic, llm):
    base = topic["base_query"]
    queries = build_queries(base, llm)
    snap = {
        "topic_id": topic["topic_id"], "base_query": base,
        "subject": topic["subject"], "queries": queries,
        "arms": {
            "single_base": tavily_search(base, MAX_RESULTS),
            "single_volume": tavily_search(base, MAX_RESULTS * 3),
            "three_persp": {p: tavily_search(q, MAX_RESULTS) for p, q in queries.items()},
        },
    }
    os.makedirs(SNAP_DIR, exist_ok=True)
    json.dump(snap, open(_snap_path(topic["topic_id"]), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return snap


def load_or_build(topic, llm, refresh):
    p = _snap_path(topic["topic_id"])
    if os.path.exists(p) and not refresh:
        return json.load(open(p, encoding="utf-8"))
    return build_snapshot(topic, llm)
```

- [ ] **Step 2: 스냅샷 스모크**

위 1차 파일에 임시 `if __name__ == "__main__":` 로 2토픽 스냅샷만 만들어 확인하거나, Step 3 완료 후 한 번에 검증. (권장: Step 3까지 작성 후 `--limit 2`로 검증)

- [ ] **Step 3: 채점·집계부 추가**

같은 파일에 이어서 작성:
```python
def _arm_snippets(arm_data):
    """arm 데이터 → (snippets:list[str], urls:list[str]) 평탄화."""
    if isinstance(arm_data, dict):  # three_persp
        results = [r for lst in arm_data.values() for r in lst]
    else:
        results = arm_data
    snippets = [f"{r['title']} {r['content']}" for r in results]
    urls = [r["url"] for r in results]
    return snippets, urls


def score_arm(arm_name, arm_data, subject, risk_items, llm):
    snippets, urls = _arm_snippets(arm_data)
    stances = [classify_stance(s, subject, llm) for s in snippets]
    hits = [risk_covered(item, snippets, llm) for item in risk_items]
    metrics = {
        "n_snippets": len(snippets),
        "neg_share": M.neg_share(stances),
        "balance_entropy": M.balance_entropy(stances),
        "unique_domains": M.unique_domain_count(urls),
        "risk_recall": M.recall(hits),
        "stance_counts": M.stance_counts(stances),
    }
    if arm_name == "three_persp" and isinstance(arm_data, dict):
        per = {p: [r["url"] for r in lst] for p, lst in arm_data.items()}
        metrics["cross_perspective_dup_rate"] = M.cross_perspective_dup_rate(per)
    return metrics


def _mean(dicts, key):
    vals = [d[key] for d in dicts if key in d]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    topics = json.load(open(TOPICS, encoding="utf-8"))
    checklist = json.load(open(CHECKLIST, encoding="utf-8"))
    if args.limit:
        topics = topics[:args.limit]
    llm = get_llm(temperature=0.0)

    per_topic = []
    degen_flags = []
    for i, t in enumerate(topics, 1):
        snap = load_or_build(t, llm, args.refresh)
        degen = M.is_degenerate(t["base_query"], snap["queries"])
        degen_flags.append(degen)
        risk_items = checklist.get(t["topic_id"], [])
        arms_scores = {
            arm: score_arm(arm, snap["arms"][arm], t["subject"], risk_items, llm)
            for arm in ("single_base", "single_volume", "three_persp")
        }
        per_topic.append({"topic_id": t["topic_id"], "degenerate": degen,
                          "arms": arms_scores})
        print(f"  [{i}/{len(topics)}] {t['topic_id']} "
              f"neg_share 3p={arms_scores['three_persp']['neg_share']} "
              f"vol={arms_scores['single_volume']['neg_share']} degen={degen}")

    # arm별 평균
    agg = {}
    for arm in ("single_base", "single_volume", "three_persp"):
        arm_dicts = [pt["arms"][arm] for pt in per_topic]
        agg[arm] = {k: _mean(arm_dicts, k) for k in
                    ("neg_share", "balance_entropy", "unique_domains", "risk_recall")}
    agg["three_persp"]["cross_perspective_dup_rate"] = _mean(
        [pt["arms"]["three_persp"] for pt in per_topic], "cross_perspective_dup_rate")

    def delta(base_arm):
        return {k: round(agg["three_persp"][k] - agg[base_arm][k], 4)
                for k in ("neg_share", "balance_entropy", "unique_domains", "risk_recall")}

    report = {
        "n_topics": len(topics),
        "query_degeneration_rate": round(sum(degen_flags) / len(degen_flags), 4),
        "arms": agg,
        "delta_three_persp_minus_single_base": delta("single_base"),
        "delta_three_persp_minus_single_volume": delta("single_volume"),
        "per_topic": per_topic,
    }
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"계층1 Web Search 평가 (n={len(topics)})")
    print("=" * 60)
    for arm in ("single_base", "single_volume", "three_persp"):
        a = agg[arm]
        print(f"  {arm:14s} neg={a['neg_share']:.3f} ent={a['balance_entropy']:.3f} "
              f"dom={a['unique_domains']:.1f} risk={a['risk_recall']:.3f}")
    print(f"  degeneration_rate = {report['query_degeneration_rate']:.3f}")
    print(f"  Δ(3p−volume) neg={report['delta_three_persp_minus_single_volume']['neg_share']:+.3f} "
          f"risk={report['delta_three_persp_minus_single_volume']['risk_recall']:+.3f}")
    print(f"\n상세 → {OUT}")


if __name__ == "__main__":
    main()
```
주의: `delta()`의 두 번째 인자가 비교 기준 arm. 첫 인자 `"a"`는 미사용(가독용) — 실제로는 `b`만 쓴다.

- [ ] **Step 4: 스모크 실행(2토픽)**

Run: `python -m eval.eval_web_search --limit 2`
Expected: 2토픽 스냅샷 생성(`web_snapshots/*.json`), arm별 지표·degeneration 출력, `web_search_metrics.json` 생성. neg_share/risk_recall 값이 0~1 범위, three_persp n_snippets≈15.

- [ ] **Step 5: 커밋(코드만)**

```bash
git add eval/eval_web_search.py
git commit -m "feat(eval): 계층1 web search 3-arm 검색 평가(스냅샷+지표)"
```

---

### Task 7: 계층2 보고서 ablation `eval/eval_report_balance.py`

`WEB_SEARCH_PERSPECTIVES` 토글로 전체 파이프라인을 두 모드로 N회 실행 → 최종 보고서 균형 비교.

**Files:**
- Create: `eval/eval_report_balance.py`
- Create (산출): `eval/results/report_balance.json`

**Interfaces:**
- Consumes: `app.build_graph`, `agents.state.BatteryAnalysisState`, `eval.judges.{extract_risk_claims, report_balance_score, risk_covered}`, `risk_checklist.json`, `get_llm`
- Produces: `report_balance.json` = mode별(perspectives_on/off) 평균±std + 델타

- [ ] **Step 1: 스크립트 작성**

Create `eval/eval_report_balance.py`:
```python
"""
계층2 — 보고서 단계 ablation.
WEB_SEARCH_PERSPECTIVES 토글로 전체 파이프라인을 두 모드로 N회 실행하고
최종 보고서(report_draft)의 균형을 비교한다.

  off : 단일 쿼리 web search (ablation)
  on  : 3방향 web search (현행)

지표: risk_section_claims(추출 리스크 수), report_balance_score(0~1),
      report_risk_recall(prod 토픽 체크리스트 대비)

실행(비용 큼 — 파이프라인 2*N회):
    python -m eval.eval_report_balance --n 3
    python -m eval.eval_report_balance --n 1   # 스모크
"""
import os
import sys
import json
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from agents.state import BatteryAnalysisState
from agents.llm_config import get_llm
from eval.judges import extract_risk_claims, report_balance_score, risk_covered

load_dotenv()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHECKLIST = os.path.join(RESULTS_DIR, "risk_checklist.json")
OUT = os.path.join(RESULTS_DIR, "report_balance.json")
QUERY = "전기차 캐즘 환경에서 LGES vs CATL 포트폴리오 다각화 전략 비교 분석"
# 보고서는 LGES/CATL/시장 중심 → prod 체크리스트로 recall 측정
RECALL_TOPICS = ["prod_lges", "prod_catl", "prod_market"]


def _run_pipeline():
    from app import build_graph  # 지연 import (env 토글 후 그래프 생성)
    app = build_graph()
    init: BatteryAnalysisState = {
        "query": QUERY, "vectorstore": None, "market_background": "",
        "lges_strategy": "", "catl_strategy": "", "comparison_result": "",
        "references": [], "report_draft": "", "error_messages": [], "rag_iteration": 0,
    }
    return app.invoke(init)["report_draft"]


def _score_report(report, checklist, llm):
    claims = extract_risk_claims(report, llm)
    balance = report_balance_score(report, llm)
    items = [it for tid in RECALL_TOPICS for it in checklist.get(tid, [])]
    hits = [risk_covered(it, [report], llm) for it in items]
    from eval.web_metrics import recall
    return {"risk_section_claims": len(claims), "report_balance_score": balance,
            "report_risk_recall": recall(hits)}


def _run_mode(enabled: bool, n: int, checklist, llm):
    prev = os.environ.get("WEB_SEARCH_PERSPECTIVES")
    os.environ["WEB_SEARCH_PERSPECTIVES"] = "true" if enabled else "false"
    runs = []
    try:
        for i in range(n):
            print(f"  [{'on' if enabled else 'off'} {i+1}/{n}] 파이프라인 실행...")
            report = _run_pipeline()
            runs.append(_score_report(report, checklist, llm))
    finally:
        if prev is None:
            os.environ.pop("WEB_SEARCH_PERSPECTIVES", None)
        else:
            os.environ["WEB_SEARCH_PERSPECTIVES"] = prev
    return runs


def _agg(runs):
    keys = ("risk_section_claims", "report_balance_score", "report_risk_recall")
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k] = {"mean": round(statistics.mean(vals), 4),
                  "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="모드당 실행 횟수")
    args = parser.parse_args()

    checklist = json.load(open(CHECKLIST, encoding="utf-8"))
    llm = get_llm(temperature=0.0)

    off = _agg(_run_mode(False, args.n, checklist, llm))
    on = _agg(_run_mode(True, args.n, checklist, llm))
    delta = {k: round(on[k]["mean"] - off[k]["mean"], 4) for k in off}

    report = {"n_runs_per_mode": args.n,
              "perspectives_off": off, "perspectives_on": on,
              "delta_on_minus_off": delta}
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"계층2 보고서 ablation (n={args.n}/mode)")
    print("=" * 60)
    for k in delta:
        print(f"  {k:24s} off={off[k]['mean']:.3f} on={on[k]['mean']:.3f} "
              f"Δ={delta[k]:+.3f}")
    print(f"\n상세 → {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 스모크 실행(n=1)**

전제: FAISS 인덱스 존재(`vectorstore/`). 없으면 `python app.py` 먼저.
Run: `python -m eval.eval_report_balance --n 1`
Expected: off/on 각 1회 파이프라인 실행 완료, 3지표·델타 출력, `report_balance.json` 생성. (시간 ≈ 8~10분 — 정상)

- [ ] **Step 3: 커밋(코드만)**

```bash
git add eval/eval_report_balance.py
git commit -m "feat(eval): 계층2 보고서 ablation 평가(env 토글)"
```

---

### Task 8: 전체 측정 실행 (실측 데이터 수집)

코드가 아니라 "실행"이 deliverable. 실제 수치를 만든다.

**Files:** (산출 갱신) `web_search_metrics.json`, `report_balance.json`, `web_snapshots/*.json`

- [ ] **Step 1: 계층1 전체 실행**

Run: `python -m eval.eval_web_search --refresh`
Expected: 20토픽 스냅샷 + `web_search_metrics.json`. 콘솔에서 arm별 평균·degeneration·델타 확인.

- [ ] **Step 2: 계층2 전체 실행(n=3)**

Run: `python -m eval.eval_report_balance --n 3`
Expected: off/on 각 3회 → `report_balance.json`. (≈25분)

- [ ] **Step 3: 결과 스냅샷 커밋**

```bash
git add eval/results/web_search_metrics.json eval/results/report_balance.json eval/results/web_snapshots
git commit -m "test(eval): web search 확증편향 평가 1차 실측 결과(v1)"
```

---

### Task 9: 개선 루프 (조건부 — 측정값 기반)

Task 8 수치를 보고 결정. 아래 규칙으로 분기하며, 개선 후 Task 8 재실행해 v1→v2 델타로 입증.

**Files:** `agents/web_search.py` (개선 대상), 재측정 결과 JSON

**결정 규칙:**
- `query_degeneration_rate > 0.1` → **개선 A(파싱 강건화)** 적용.
- `three_persp.cross_perspective_dup_rate > 0.3` → **개선 B(URL dedup)** 적용.
- `delta_three_persp_minus_single_volume.neg_share <= 0.05` 이고 degeneration 낮음 → **개선 C(비판 쿼리 프롬프트 강화)** 적용.
- 위 어디에도 해당 없고 델타가 미미 → 개선하지 않고 "측정된 순이득 없음"으로 결론(Task 10에 정직 기록).

- [ ] **Step 1: 개선 A — 파싱 강건화 (degeneration 높을 때)**

`agents/web_search.py`의 `_build_three_queries` 파싱부를 강건화. 형식 미준수 시 폴백 대신, 라벨 누락 줄을 순서대로 매핑하고, 여전히 실패하면 base에 관점 키워드를 접미해 강제 분기:
```python
def _build_three_queries(base_query: str, llm: ChatOpenAI) -> Dict[str, str]:
    # ... system_prompt 동일 ...
    response = llm.invoke([SystemMessage(content=system_prompt),
                           HumanMessage(content=f"주제: {base_query}")])
    lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
    queries = {}
    for line in lines:
        for key in ("긍정", "비판", "중립"):
            if line.startswith(f"{key}:"):
                queries[key] = line.split(":", 1)[1].strip()
    # 누락 관점은 base에 관점 키워드를 붙여 강제 분기(폴백해도 degenerate 방지)
    suffix = {"긍정": "성과 강점", "비판": "리스크 한계 문제점", "중립": "현황 분석"}
    for key in ("긍정", "비판", "중립"):
        if not queries.get(key):
            queries[key] = f"{base_query} {suffix[key]}"
    return queries
```

- [ ] **Step 2: 개선 B — URL dedup (중복 높을 때)**

`web_search.py`의 3방향 경로에서 이미 본 URL을 건너뛰도록 `tavily_search` 결과를 누적 dedup. `web_search()` 3방향 루프에 `seen=set()` 추가하고 `r['url'] in seen`이면 skip.

- [ ] **Step 3: 개선 C — 비판 쿼리 강화 (neg 효과 약할 때)**

`_build_three_queries`의 system_prompt에서 비판 쿼리 지시를 구체화:
```
비판: <부정·리스크 측면 강조 — "리스크", "한계", "실패", "논란", "규제" 중 1개 이상 포함>
```

- [ ] **Step 4: 개선분 단위테스트 갱신·통과**

개선 A 적용 시 `eval/test_web_search_helpers.py`에 누락-관점 강제분기 테스트 추가(예: build_queries가 항상 3개 distinct 보장 — LLM 모킹 또는 결과 키 존재 확인). 적용한 개선에 해당하는 테스트만.
Run: `python -m pytest eval/ -v`
Expected: PASS

- [ ] **Step 5: 재측정 + v2 결과 커밋**

```bash
python -m eval.eval_web_search --refresh
python -m eval.eval_report_balance --n 3
git add agents/web_search.py eval/test_web_search_helpers.py eval/results/
git commit -m "fix(web_search): <적용한 개선 A/B/C> — degeneration/dup/비판쿼리 개선 + v2 재측정"
```

---

### Task 10: 결과 문서화 (RESULTS.md §5, README, TODO)

수치·해석·한계를 기존 톤으로 기록. 효과 없으면 "없음"을, 개선했으면 v1→v2를 정직하게.

**Files:**
- Modify: `eval/RESULTS.md` (§5 신규)
- Modify: `eval/README.md` (실행 순서 ⑥⑦)
- Modify: `TODO.md` (P8-3 체크)

- [ ] **Step 1: RESULTS.md §5 작성**

`eval/RESULTS.md` 끝에 §5 추가. 실측 표를 `web_search_metrics.json`·`report_balance.json`에서 채운다. 골격:
```markdown
## 5. Web Search 확증 편향 방지 — 3방향 쿼리 (n=20 토픽)

> 실행: `eval/eval_web_search.py`(검색)·`eval/eval_report_balance.py`(보고서). 원본 JSON: `eval/results/web_*.json`.
> 3-arm으로 볼륨 교란 통제. 헤드라인 델타 = three_persp − single_volume.

### 5-1. 검색 단계 (arm별 평균)
| 지표 | single_base | single_volume | three_persp | Δ(3p−volume) |
|---|---|---|---|---|
| neg_share | … | … | … | … |
| balance_entropy | … | … | … | … |
| unique_domains | … | … | … | … |
| risk_recall | … | … | … | … |

- query_degeneration_rate: …
- cross_perspective_dup_rate(three_persp): …

**핵심 관찰**: (volume 매칭 대비 관점 분할의 순수 기여를 1~3문장으로. 효과 유/무를 단정 말고 델타 크기로.)

### 5-2. 보고서 단계 ablation (n=3/mode)
| 지표 | off(단일) | on(3방향) | Δ(on−off) |
|---|---|---|---|
| risk_section_claims | … | … | … |
| report_balance_score | … | … | … |
| report_risk_recall | … | … | … |

### 5-3. 개선 (있었다면 v1→v2)
(degeneration/dup/비판쿼리 개선 적용 시 전후 델타. 없으면 "개선 불요 — degeneration·dup 낮음" 명시.)

### 측정상의 한계
- 웹 라이브 변동: 스냅샷 1회 기준(소표본 K=3 분산은 별도). stance/risk 판정은 LLM-judge temp=0(사람 검수 아님).
- 보고서 단계는 mode당 n=3, report_generation 분산 큼 → "소폭" 델타는 단정 안 함.
- 배터리 산업 토픽 한정, 외부 일반화 보장 아님.
```

- [ ] **Step 2: 종합 해석 + README/TODO 갱신**

- `RESULTS.md` "종합 해석" 섹션에 web search 한 줄 추가(델타 기준 결론).
- `eval/README.md` "실행 순서"에 ⑥ `python -m eval.build_risk_checklist` → `python -m eval.eval_web_search`, ⑦ `python -m eval.eval_report_balance --n 3` 추가. "스크립트별 산출값" 표에 3개 신규 스크립트 행 추가.
- `TODO.md`의 `P8-3 확증 편향 방지 동작 확인 (3방향 쿼리 결과 비교)`를 `[x]`로 변경하고 결과 위치(`eval/RESULTS.md §5`) 주석.

- [ ] **Step 3: 최종 커밋**

```bash
git add eval/RESULTS.md eval/README.md TODO.md
git commit -m "docs(eval): web search 확증편향 평가 결과(§5)·README·TODO 갱신"
```

---

## Self-Review

**1. Spec coverage:**
- 3-arm(single_base/single_volume/three_persp) → Task 2(토글), Task 6(arm 구성) ✓
- 4지표(stance 균형/소스 다양성/리스크 recall/degeneration) → Task 1(함수)+Task 6(집계) ✓
- 스냅샷 재현성 → Task 6 ✓ / 웹 변동 K=3 분산 → RESULTS.md 한계로 명시(Task 10). **주의: 소표본 K=3 반복 측정 스크립트는 본 플랜에 미구현(한계로만 기록).** 필요 시 별도 태스크.
- 보고서 ablation(env, N=3) → Task 7, Task 8 ✓
- 프로덕션 기본 동작 불변 → Task 2 Step 5 회귀 확인 ✓
- 리스크 체크리스트 → Task 4 ✓
- 개선 루프(조건부, v1→v2) → Task 9 ✓
- 정직한 기록 → Task 10 ✓

**갭 발견:** 스펙 4.1의 "소수 토픽 K=3 반복 분산"이 실행 태스크로 빠짐. 결정: 헤드라인이 아니므로 v1에서는 RESULTS.md 한계 문구로 대체하고, 원하면 Task 6에 `--repeat` 옵션을 추가하는 후속 작업으로 둔다(YAGNI). Task 10 Step1 한계 문구에 이미 반영.

**2. Placeholder scan:** RESULTS.md 표의 `…`는 실측값 채움 자리(실행 후 입력)로 의도된 것 — 코드 placeholder 아님. 그 외 TBD/TODO 없음. ✓

**3. Type consistency:** `tavily_search`→`{title,url,content,published,domain}` (Task2)와 `_arm_snippets`/`score_arm` 소비(Task6) 일치 ✓. `STANCES`(Task1)와 `parse_stance_label`/`classify_stance`(Task5) 일치 ✓. `recall`(Task1)을 Task6·Task7에서 동일 시그니처로 사용 ✓. `web_metrics`의 `is_degenerate`(Task1)와 `eval_web_search` 사용(Task6) 일치 ✓.
```
