"""
Comparison Agent — T5
LGES·CATL 전략 분석 결과를 State에서 읽어
전략 비교 매트릭스 + SWOT 전체(S/W/O/T) 일괄 작성
"""

import os
from langchain_core.messages import HumanMessage

from agents.state import BatteryAnalysisState
from agents.llm_config import get_llm


def _load_prompt(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def comparison_node(state: BatteryAnalysisState) -> dict:
    """
    T5: 전략 비교 매트릭스 + SWOT 전체(S/W/O/T) 일괄 작성
    SWOT를 분산하지 않고 여기서 한 번에 완성 → 일관성 보장
    """
    print("[Comparison] 비교 분석 및 SWOT 작성 시작...")
    llm = get_llm()
    prompt_template = _load_prompt("comparison.txt")

    lges_strategy = state.get("lges_strategy", "LGES 분석 결과 없음")
    catl_strategy = state.get("catl_strategy", "CATL 분석 결과 없음")
    market_background = state.get("market_background", "시장 배경 정보 없음")

    prompt = prompt_template.format(
        lges_strategy=lges_strategy,
        catl_strategy=catl_strategy,
        market_background=market_background,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    print("[Comparison] 완료")

    return {
        "comparison_result": response.content,
        "error_messages": [],
    }
