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
