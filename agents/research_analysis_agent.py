"""
Research & Analysis Agent
동일한 에이전트를 3가지 컨텍스트로 호출:
  - market_research_node   : 시장 배경 (T2)
  - lges_analysis_node     : LGES 전략 (T3)
  - catl_analysis_node     : CATL 전략 (T4) — lges와 병렬 실행

각 노드는 RAG Tool + Web Search Tool을 호출하여
정보를 수집하고 LLM으로 분석 결과를 생성한다.
"""

import os
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import BatteryAnalysisState
from agents.llm_config import get_llm
from agents.rag_tool import rag_retrieve
from agents.web_search import web_search_with_refs

# 프롬프트 로더
def _load_prompt(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────
# 내부 공통 함수
# ─────────────────────────────────────────────────────────────

def _run_research(
    query_rag: str,
    query_web: str,
    prompt_template: str,
    prompt_kwargs: dict,
    vectorstore,
    llm,
) -> tuple[str, list]:
    """
    RAG + Web Search → LLM 분석 공통 로직
    Returns: (분석_결과_str, references_list)
    """
    # 1. RAG 검색 (기본 plain, RAG_AGENTIC=true면 재검색 루프)
    rag_context, references = rag_retrieve(query_rag, vectorstore, llm)

    # 2. Web Search (확증 편향 방지: 3방향 쿼리)
    web_context, web_references = web_search_with_refs(query_web, llm)

    # 3. 프롬프트 조립
    prompt = prompt_template.format(
        rag_context=rag_context,
        web_context=web_context,
        **prompt_kwargs,
    )

    # 4. LLM 분석
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, references + web_references


# ─────────────────────────────────────────────────────────────
# LangGraph 노드 함수
# ─────────────────────────────────────────────────────────────

def market_research_node(state: BatteryAnalysisState) -> dict:
    """T2: 배터리·전기차 시장 배경 조사"""
    print("[MarketResearch] 시장 배경 분석 시작...")
    llm = get_llm()
    vectorstore = state["vectorstore"]
    prompt_template = _load_prompt("market_research.txt")

    result, refs = _run_research(
        query_rag="배터리 시장 현황 전기차 캐즘 HEV 공급망 정책",
        query_web="글로벌 배터리 전기차 시장 2025 캐즘 현황",
        prompt_template=prompt_template,
        prompt_kwargs={},
        vectorstore=vectorstore,
        llm=llm,
    )
    print("[MarketResearch] 완료")
    return {
        "market_background": result,
        "references": refs,
        "error_messages": [],
    }


def lges_analysis_node(state: BatteryAnalysisState) -> dict:
    """T3: LG에너지솔루션 전략 분석"""
    print("[LGES] 전략 분석 시작...")
    llm = get_llm()
    vectorstore = state["vectorstore"]
    prompt_template = _load_prompt("company_analysis.txt")

    result, refs = _run_research(
        query_rag="LG에너지솔루션 LGES 포트폴리오 전략 ESS 로봇 배터리 HEV",
        query_web="LG에너지솔루션 2025 포트폴리오 다각화 전략",
        prompt_template=prompt_template,
        prompt_kwargs={
            "company_name": "LG에너지솔루션(LGES)",
            "focus_areas": "ESS(에너지저장장치), 로봇·Physical AI용 배터리, HEV 배터리 대응",
        },
        vectorstore=vectorstore,
        llm=llm,
    )
    print("[LGES] 완료")
    return {
        "lges_strategy": result,
        "references": refs,
        "error_messages": [],
    }


def catl_analysis_node(state: BatteryAnalysisState) -> dict:
    """T4: CATL 전략 분석"""
    print("[CATL] 전략 분석 시작...")
    llm = get_llm()
    vectorstore = state["vectorstore"]
    prompt_template = _load_prompt("company_analysis.txt")

    result, refs = _run_research(
        query_rag="CATL 나트륨이온 배터리 ESS 신흥시장 아프리카 LFP",
        query_web="CATL 2025 포트폴리오 전략 나트륨이온 ESS 신흥시장",
        prompt_template=prompt_template,
        prompt_kwargs={
            "company_name": "CATL(Contemporary Amperex Technology)",
            "focus_areas": "나트륨이온 배터리 상업화, ESS 신흥시장 공략(아프리카 등), LFP 원가 경쟁력",
        },
        vectorstore=vectorstore,
        llm=llm,
    )
    print("[CATL] 완료")
    return {
        "catl_strategy": result,
        "references": refs,
        "error_messages": [],
    }
