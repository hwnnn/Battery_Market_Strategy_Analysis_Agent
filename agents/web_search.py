"""
Web Search Tool — 확증 편향 방지 설계
단일 쿼리 입력 → 긍정/비판/중립 3방향 쿼리 자동 생성 → 병렬 검색 → 통합 반환
"""

import os
import asyncio
from typing import List, Dict
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


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
비판: <부정적·리스크 측면 강조 쿼리>
중립: <현황·사실 중심 쿼리>

각 쿼리는 한 줄씩, 15~30자 이내로 작성하세요."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"주제: {base_query}")
    ])

    lines = response.content.strip().split("\n")
    queries = {"긍정": base_query, "비판": base_query, "중립": base_query}

    for line in lines:
        if line.startswith("긍정:"):
            queries["긍정"] = line.replace("긍정:", "").strip()
        elif line.startswith("비판:"):
            queries["비판"] = line.replace("비판:", "").strip()
        elif line.startswith("중립:"):
            queries["중립"] = line.replace("중립:", "").strip()

    return queries


def web_search(base_query: str, llm: ChatOpenAI, max_results: int = 5) -> str:
    """
    확증 편향 방지 웹 검색 Tool
    - 3방향 쿼리 생성 후 각각 Tavily 검색
    - 결과를 관점별로 구분하여 통합 반환
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    client = TavilyClient(api_key=api_key)

    # 3방향 쿼리 생성
    queries = _build_three_queries(base_query, llm)

    all_results: List[str] = []

    for perspective, query in queries.items():
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
            )
            results = response.get("results", [])

            section = f"\n=== [{perspective} 관점] 검색 쿼리: {query} ===\n"
            for i, r in enumerate(results, 1):
                title = r.get("title", "제목 없음")
                url = r.get("url", "")
                content = r.get("content", "내용 없음")[:500]  # 500자 제한
                published = r.get("published_date", "날짜 미상")
                section += f"\n[{i}] {title}\n출처: {url}\n날짜: {published}\n내용: {content}\n"

            all_results.append(section)

        except Exception as e:
            all_results.append(f"\n=== [{perspective} 관점] 검색 실패: {str(e)} ===\n")

    return "\n".join(all_results)
