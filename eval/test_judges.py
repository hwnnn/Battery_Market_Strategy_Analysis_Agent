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
    # regression: year must NOT win when an in-range value is present
    ("2025년 기준 균형 점수 0.7", 0.7),
    # regression: no in-range number → no number at all → 0.0
    ("점수 없음", 0.0),
])
def test_parse_score(raw, expected):
    assert parse_score(raw) == pytest.approx(expected)
