"""
CATL Analysis Agent — T4
CATL 포트폴리오 다각화 전략 분석

담당 State 필드: catl_strategy (다른 에이전트 필드에 절대 쓰지 않음)
LGES Analysis Agent와 병렬 실행되며 state 필드가 완전히 분리되어 있음
"""

from agents.state import BatteryAnalysisState
from agents.llm_config import get_llm
from agents.research_base import run_research, load_prompt


def catl_analysis_node(state: BatteryAnalysisState) -> dict:
    """
    T4: CATL 전략 분석
    결과 저장: state['catl_strategy'] 전용 — lges_strategy에 절대 쓰지 않음
    """
    print("[CATL Analysis] 전략 분석 시작...")
    llm = get_llm()
    vectorstore = state["vectorstore"]

    result, refs = run_research(
        query_rag=(
            "CATL 나트륨이온 배터리 ESS 신흥시장 아프리카 LFP "
            "BMW 포드 테슬라 파트너십 생산거점 수익성 리스크"
        ),
        query_web=(
            "CATL 2025 파트너십 협력사 생산 전략 "
            "BMW 포드 테슬라 계약 현황 지정학 리스크 서방 규제"
        ),
        prompt_template=load_prompt("company_analysis.txt"),
        prompt_kwargs={
            "company_name": "CATL(Contemporary Amperex Technology)",
            "focus_areas": "나트륨이온 배터리 상업화, ESS 신흥시장 공략(아프리카 등), LFP 원가 경쟁력",
            "partnership_focus": (
                "BMW·포드·테슬라 등 주요 파트너사별 계약 규모·공급 현황과 함께 "
                "서방 국가의 중국 배터리 규제 리스크, 헝가리·브라질 등 해외 공장의 "
                "진출 배경과 현지 리스크를 구체적으로 서술"
            ),
        },
        vectorstore=vectorstore,
        llm=llm,
    )

    print("[CATL Analysis] 완료")
    return {
        "catl_strategy": result,
        "references": refs,
        "error_messages": [],
    }
