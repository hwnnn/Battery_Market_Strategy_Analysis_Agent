"""
Research Base — 에이전트 간 공유 유틸리티
각 Research Agent는 이 모듈을 통해 RAG + Web Search를 호출한다.
직접 실행되지 않으며, Agent가 자신의 state 필드에 독립적으로 결과를 저장한다.
"""

import os
from typing import Tuple, List
from langchain_openai import ChatOpenAI

from agents.rag_tool import rag_retrieve
from agents.web_search import web_search


def load_prompt(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run_research(
    query_rag: str,
    query_web: str,
    prompt_template: str,
    prompt_kwargs: dict,
    vectorstore,
    llm: ChatOpenAI,
) -> Tuple[str, List[dict]]:
    """
    RAG 검색 → Web Search → LLM 분석 공통 흐름
    각 에이전트가 독립적으로 호출하며, 결과를 자신의 state 필드에만 저장

    Returns:
        result (str): LLM 분석 결과
        references (List[dict]): 참고 문서 목록
    """
    from langchain_core.messages import HumanMessage

    # 1. Agentic RAG (최대 3회 재시도)
    rag_context, references = rag_retrieve(query_rag, vectorstore, llm)

    # 2. Web Search (확증 편향 방지: 긍정/비판/중립 3방향)
    web_context = web_search(query_web, llm)

    # 3. 프롬프트 조립
    prompt = prompt_template.format(
        rag_context=rag_context,
        web_context=web_context,
        **prompt_kwargs,
    )

    # 4. LLM 분석
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, references
