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
