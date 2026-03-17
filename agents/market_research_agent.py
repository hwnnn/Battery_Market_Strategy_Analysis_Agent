"""
Market Research Agent — T2
배터리·전기차 시장 배경 조사

담당 State 필드: market_background (다른 에이전트 필드에 절대 쓰지 않음)
"""

from agents.state import BatteryAnalysisState
from agents.llm_config import get_llm
from agents.research_base import run_research, load_prompt


def market_research_node(state: BatteryAnalysisState) -> dict:
    """
    T2: 글로벌 배터리·전기차 시장 환경 분석
    결과 저장: state['market_background'] 전용
    """
    print("[MarketResearch] 시장 배경 분석 시작...")
    llm = get_llm()
    vectorstore = state["vectorstore"]

    result, refs = run_research(
        query_rag="배터리 시장 현황 전기차 캐즘 HEV 공급망 정책",
        query_web="글로벌 배터리 전기차 시장 2025 캐즘 현황",
        prompt_template=load_prompt("market_research.txt"),
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
