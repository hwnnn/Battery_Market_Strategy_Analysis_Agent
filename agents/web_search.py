"""
Web Search Tool — 확증 편향 방지 설계
단일 쿼리 입력 → 긍정/비판/중립 3방향 쿼리 자동 생성 → 병렬 검색 → 통합 반환
WEB_SEARCH_PERSPECTIVES=off 로 단일 쿼리 ablation 전환 가능 (기본: 3방향 on)
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from urllib.parse import urlparse
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

PERSPECTIVES = ("긍정", "비판", "중립")
RISK_TERMS = ("리스크", "한계", "실패", "논란", "규제")


def perspectives_enabled() -> bool:
    """WEB_SEARCH_PERSPECTIVES 환경변수로 3방향 on/off (기본 on)."""
    return os.getenv("WEB_SEARCH_PERSPECTIVES", "true").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _build_three_queries(base_query: str, llm: ChatOpenAI) -> Dict[str, str]:
    """
    기본 쿼리에서 긍정·비판·중립 3방향 쿼리를 LLM으로 생성한다.
    확증 편향 방지: 검색 결과가 특정 관점에 치우치지 않도록 강제
    """
    system_prompt = """당신은 검색 쿼리 전문가입니다.
주어진 주제에 대해 확증 편향 없이 균형 잡힌 정보를 수집하기 위해
아래 3가지 관점의 검색 쿼리를 각각 생성하세요.

형식 (정확히 지켜주세요):
긍정: <긍정적 측면 강조 쿼리>
비판: <부정·리스크 측면 강조 — "리스크", "한계", "실패", "논란", "규제" 중 1개 이상 포함>
중립: <현황·사실 중심 쿼리>

각 쿼리는 한 줄씩, 15~30자 이내로 작성하세요."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"주제: {base_query}")
    ])

    lines = response.content.strip().split("\n")
    queries = {"긍정": base_query, "비판": base_query, "중립": base_query}

    for line in lines:
        line = line.strip()
        if line.startswith("긍정:"):
            queries["긍정"] = line.replace("긍정:", "").strip()
        elif line.startswith("비판:"):
            queries["비판"] = line.replace("비판:", "").strip()
        elif line.startswith("중립:"):
            queries["중립"] = line.replace("중립:", "").strip()

    if not any(term in queries["비판"] for term in RISK_TERMS):
        queries["비판"] = f"{queries['비판']} 리스크"

    return queries


def build_queries(base_query: str, llm: ChatOpenAI) -> Dict[str, str]:
    """기본 쿼리에서 긍정·비판·중립 3방향 쿼리를 LLM으로 생성한다.
    (기존 _build_three_queries 와 동일 동작 — 공개 이름)"""
    return _build_three_queries(base_query, llm)


def _require_tavily_key() -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    return api_key


def domain_of(url: str) -> str:
    """URL에서 www.를 제거한 도메인을 추출한다. 파싱 실패 시 원본을 반환."""
    netloc = urlparse(url).netloc
    if not netloc:
        return url
    return netloc[4:] if netloc.startswith("www.") else netloc


def tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """단일 쿼리 Tavily 검색 → 정규화된 결과 리스트.
    각 원소: {title, url, content, published, domain}"""
    api_key = _require_tavily_key()
    client = TavilyClient(api_key=api_key)
    resp = client.search(query=query, max_results=max_results, search_depth="advanced")
    out = []
    for r in resp.get("results", []):
        url = r.get("url", "")
        out.append({
            "title": r.get("title", "제목 없음")[:200],
            "url": url,
            "content": r.get("content", "내용 없음")[:500],
            "published": r.get("published_date", "날짜 미상"),
            "domain": domain_of(url),
        })
    return out


def _format_section(label: str, query: str, results: List[dict]) -> str:
    section = f"\n=== [{label}] 검색 쿼리: {query} ===\n"
    for i, r in enumerate(results, 1):
        section += (f"\n[{i}] {r['title']}\n출처: {r['url']}\n"
                    f"날짜: {r['published']}\n내용: {r['content']}\n")
    return section


def _refs_from_results(results: List[dict], perspective: str, query: str) -> List[dict]:
    refs = []
    for r in results:
        refs.append({
            "type": "web",
            "source": r.get("domain") or domain_of(r.get("url", "")),
            "title": r.get("title", "제목 없음"),
            "url": r.get("url", "URL 없음"),
            "date": r.get("published", "날짜 미상"),
            "perspective": perspective,
            "query": query,
        })
    return refs


def web_search_with_refs(
    base_query: str,
    llm: ChatOpenAI,
    max_results: int = 5,
) -> Tuple[str, List[dict]]:
    """확증 편향 방지 웹 검색 Tool.
    WEB_SEARCH_PERSPECTIVES=off 면 단일 쿼리(ablation), 기본은 3방향.
    Returns:
        context: 프롬프트에 넣을 검색 결과 문자열
        references: REFERENCE 섹션에 넘길 웹 출처 메타데이터
    """
    _require_tavily_key()
    if not perspectives_enabled():
        # ablation: 볼륨을 3방향과 맞추기 위해 max_results*3
        results = tavily_search(base_query, max_results=max_results * 3)
        return (
            _format_section("단일 쿼리(ablation)", base_query, results),
            _refs_from_results(results, "단일", base_query),
        )

    queries = build_queries(base_query, llm)
    sections_by_perspective: Dict[str, str] = {}
    refs_by_perspective: Dict[str, List[dict]] = {}

    def run_one(perspective: str, query: str) -> tuple[str, str, List[dict]]:
        results = tavily_search(query, max_results=max_results)
        section = _format_section(f"{perspective} 관점", query, results)
        return perspective, section, _refs_from_results(results, perspective, query)

    with ThreadPoolExecutor(max_workers=len(PERSPECTIVES)) as executor:
        futures = {
            executor.submit(run_one, perspective, queries[perspective]): perspective
            for perspective in PERSPECTIVES
        }
        for future in as_completed(futures):
            perspective = futures[future]
            try:
                p, section, perspective_refs = future.result()
                sections_by_perspective[p] = section
                refs_by_perspective[p] = perspective_refs
            except Exception as e:
                sections_by_perspective[perspective] = (
                    f"\n=== [{perspective} 관점] 검색 실패: {str(e)} ===\n"
                )
                refs_by_perspective[perspective] = []

    ordered_sections = [sections_by_perspective[p] for p in PERSPECTIVES]
    refs = [ref for p in PERSPECTIVES for ref in refs_by_perspective[p]]
    return "\n".join(ordered_sections), refs


def web_search(base_query: str, llm: ChatOpenAI, max_results: int = 5) -> str:
    """기존 호출부 호환용: 검색 컨텍스트 문자열만 반환."""
    context, _refs = web_search_with_refs(base_query, llm, max_results=max_results)
    return context
