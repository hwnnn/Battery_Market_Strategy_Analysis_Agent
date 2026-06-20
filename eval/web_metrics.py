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
