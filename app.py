"""
Battery Market Strategy Analysis Agent — 메인 실행 진입점

Distributed Pattern (LangGraph StateGraph)
- Supervisor 없음
- 노드 간 predetermined/conditional edge로 라우팅
- LGES·CATL 분석은 fan-out/fan-in으로 병렬 실행

Graph Flow:
  document_loader
      → market_research
      → [lges_analysis ‖ catl_analysis]  (병렬)
      → comparison
      → report_generation
      → END
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from agents.state import BatteryAnalysisState
from agents.document_loader import document_loader_node
from agents.market_research_agent import market_research_node
from agents.lges_analysis_agent import lges_analysis_node
from agents.catl_analysis_agent import catl_analysis_node
from agents.comparison_agent import comparison_node
from agents.report_generator import report_generation_node

load_dotenv()


def build_graph() -> StateGraph:
    """LangGraph StateGraph 조립 — Distributed Pattern"""
    graph = StateGraph(BatteryAnalysisState)

    # ── 노드 등록 ────────────────────────────────────────────
    graph.add_node("document_loader",    document_loader_node)
    graph.add_node("market_research",    market_research_node)
    graph.add_node("lges_analysis",      lges_analysis_node)
    graph.add_node("catl_analysis",      catl_analysis_node)
    graph.add_node("comparison",         comparison_node)
    graph.add_node("report_generation",  report_generation_node)

    # ── 엣지 설정 (Distributed: edge가 라우팅 담당) ──────────
    graph.add_edge(START,               "document_loader")
    graph.add_edge("document_loader",   "market_research")

    # fan-out: market_research → [lges_analysis, catl_analysis] 병렬
    graph.add_edge("market_research",   "lges_analysis")
    graph.add_edge("market_research",   "catl_analysis")

    # fan-in: 둘 다 끝나야 comparison 시작
    graph.add_edge("lges_analysis",     "comparison")
    graph.add_edge("catl_analysis",     "comparison")

    graph.add_edge("comparison",        "report_generation")
    graph.add_edge("report_generation", END)

    return graph.compile()


def run_analysis(query: str = "전기차 캐즘 환경에서 LGES vs CATL 포트폴리오 다각화 전략 비교 분석"):
    """파이프라인 실행"""
    print("=" * 60)
    print("배터리 시장 전략 분석 Agent 시작")
    print(f"분석 주제: {query}")
    print("=" * 60)

    # API 키 검증
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
    if not os.getenv("TAVILY_API_KEY"):
        raise EnvironmentError("TAVILY_API_KEY가 없습니다. .env 파일을 확인하세요.")

    app = build_graph()

    # 초기 State
    initial_state: BatteryAnalysisState = {
        "query": query,
        "vectorstore": None,
        "market_background": "",
        "lges_strategy": "",
        "catl_strategy": "",
        "comparison_result": "",
        "references": [],
        "report_draft": "",
        "error_messages": [],
        "rag_iteration": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("분석 완료!")
    print(f"보고서 저장 위치: outputs/report.md")
    if final_state.get("error_messages"):
        print(f"[경고] 오류 로그: {final_state['error_messages']}")
    print("=" * 60)

    return final_state


def save_graph_image():
    """그래프 시각화 이미지 저장 (README용)"""
    try:
        app = build_graph()
        graph_image = app.get_graph().draw_mermaid_png()
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/graph.png", "wb") as f:
            f.write(graph_image)
        print("그래프 이미지 저장: outputs/graph.png")
    except Exception as e:
        print(f"그래프 이미지 저장 실패 (선택 사항): {e}")


if __name__ == "__main__":
    run_analysis()
