import os
import pytest

from agents import web_search as web_search_module
from agents.web_search import build_queries, perspectives_enabled


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, _messages):
        return _FakeResponse(self.content)


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


def test_build_queries_enforces_risk_keyword():
    llm = _FakeLLM("긍정: CATL 성과\n비판: CATL 문제\n중립: CATL 현황")

    queries = build_queries("CATL 전략", llm)

    assert "리스크" in queries["비판"]


def test_web_search_with_refs_single_mode(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("WEB_SEARCH_PERSPECTIVES", "off")

    calls = []

    def fake_search(query, max_results):
        calls.append((query, max_results))
        return [{
            "title": "Example title",
            "url": "https://www.example.com/news",
            "content": "Example content",
            "published": "2026-01-01",
            "domain": "example.com",
        }]

    monkeypatch.setattr(web_search_module, "tavily_search", fake_search)

    context, refs = web_search_module.web_search_with_refs(
        "battery market", _FakeLLM(""), max_results=5,
    )

    assert calls == [("battery market", 15)]
    assert "단일 쿼리(ablation)" in context
    assert refs == [{
        "type": "web",
        "source": "example.com",
        "title": "Example title",
        "url": "https://www.example.com/news",
        "date": "2026-01-01",
        "perspective": "단일",
        "query": "battery market",
    }]


def test_web_search_with_refs_three_perspectives(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("WEB_SEARCH_PERSPECTIVES", "true")
    monkeypatch.setattr(web_search_module, "build_queries", lambda _base, _llm: {
        "긍정": "positive query",
        "비판": "critical query 리스크",
        "중립": "neutral query",
    })

    def fake_search(query, max_results):
        slug = query.split()[0]
        return [{
            "title": f"{query} title",
            "url": f"https://{slug}.example.com/news",
            "content": "content",
            "published": "2026-01-01",
            "domain": f"{slug}.example.com",
        }]

    monkeypatch.setattr(web_search_module, "tavily_search", fake_search)

    context, refs = web_search_module.web_search_with_refs(
        "battery market", _FakeLLM(""), max_results=5,
    )

    assert context.find("[긍정 관점]") < context.find("[비판 관점]") < context.find("[중립 관점]")
    assert [r["perspective"] for r in refs] == ["긍정", "비판", "중립"]
    assert {r["query"] for r in refs} == {
        "positive query", "critical query 리스크", "neutral query",
    }
