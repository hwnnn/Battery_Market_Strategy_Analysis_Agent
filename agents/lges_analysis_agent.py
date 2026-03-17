"""
LGES Analysis Agent — T3
LG에너지솔루션 포트폴리오 다각화 전략 분석

담당 State 필드: lges_strategy (다른 에이전트 필드에 절대 쓰지 않음)
CATL Analysis Agent와 병렬 실행되며 state 필드가 완전히 분리되어 있음
"""

from agents.state import BatteryAnalysisState
from agents.llm_config import get_llm
from agents.research_base import run_research, load_prompt


def lges_analysis_node(state: BatteryAnalysisState) -> dict:
    """
    T3: LG에너지솔루션 전략 분석
    결과 저장: state['lges_strategy'] 전용 — catl_strategy에 절대 쓰지 않음
    """
    print("[LGES Analysis] 전략 분석 시작...")
    llm = get_llm()
    vectorstore = state["vectorstore"]

    result, refs = run_research(
        query_rag=(
            "LG에너지솔루션 LGES 포트폴리오 전략 ESS 로봇 배터리 HEV "
            "JV 합작공장 가동률 고정비 파트너십 리스크 수익성"
        ),
        query_web=(
            "LG에너지솔루션 2025 포트폴리오 전략 파트너십 "
            "북미 합작공장 가동률 고정비 JV 리스크 수익성 문제"
        ),
        prompt_template=load_prompt("company_analysis.txt"),
        prompt_kwargs={
            "company_name": "LG에너지솔루션(LGES)",
            "focus_areas": "ESS(에너지저장장치), 로봇·Physical AI용 배터리, HEV 배터리 대응",
            "partnership_focus": (
                "GM, 스텔란티스 등 북미 JV 파트너사별 현황·계약 규모와 함께 "
                "가동률 저하·고정비 부담·합작 리스크를 파트너별로 구체적으로 서술"
            ),
        },
        vectorstore=vectorstore,
        llm=llm,
    )

    print("[LGES Analysis] 완료")
    return {
        "lges_strategy": result,
        "references": refs,
        "error_messages": [],
    }
