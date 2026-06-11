"""
아키텍처 비교 — Distributed Pattern vs Supervisor Pattern
=========================================================
완전히 동일한 worker 노드를 사용하되, 라우팅 방식만 다른 두 그래프를
각각 1회 실행해 비교한다.

  - Distributed : 고정 edge 라우팅 (라우팅용 LLM 호출 0회) — 현 프로덕션 구조
  - Supervisor  : 매 단계 supervisor 노드가 LLM으로 다음 worker 결정 (라우팅 호출 N회)

측정(공정성을 위해 동일 worker 사용 → 차이는 라우팅 오버헤드로 귀속):
  - 총 LLM 호출 수 / 총 토큰 / 추정 비용  (langchain get_openai_callback)
  - 전체 실행 시간
  - 라우팅 전용 호출 수 / 토큰 / 시간       (supervisor 노드에서 격리 집계)

주의: 실제 파이프라인을 2회 돌리므로 Tavily 웹검색 + GPT-4o-mini 호출 비용이 발생하고
수 분이 소요된다.

실행:
    python -m eval.supervisor_compare
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import BatteryAnalysisState
from agents.document_loader import document_loader_node
from agents.market_research_agent import market_research_node
from agents.lges_analysis_agent import lges_analysis_node
from agents.catl_analysis_agent import catl_analysis_node
from agents.comparison_agent import comparison_node
from agents.report_generator import report_generation_node
from agents.llm_config import get_llm

load_dotenv()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
QUERY = "전기차 캐즘 환경에서 LGES vs CATL 포트폴리오 다각화 전략 비교 분석"

# supervisor가 순차 라우팅할 worker 순서 (fan-out도 supervisor는 하나씩 처리)
WORKER_ORDER = [
    "document_loader", "market_research", "lges_analysis",
    "catl_analysis", "comparison", "report_generation",
]

# 라우팅 오버헤드 격리 집계
routing_stats = {"calls": 0, "tokens": 0, "cost": 0.0, "time": 0.0}


def _initial_state() -> dict:
    return {
        "query": QUERY, "vectorstore": None,
        "market_background": "", "lges_strategy": "", "catl_strategy": "",
        "comparison_result": "", "references": [], "report_draft": "",
        "error_messages": [], "rag_iteration": 0,
    }


# ── Distributed (현 프로덕션 구조) ────────────────────────────────
def build_distributed():
    g = StateGraph(BatteryAnalysisState)
    g.add_node("document_loader", document_loader_node)
    g.add_node("market_research", market_research_node)
    g.add_node("lges_analysis", lges_analysis_node)
    g.add_node("catl_analysis", catl_analysis_node)
    g.add_node("comparison", comparison_node)
    g.add_node("report_generation", report_generation_node)
    g.add_edge(START, "document_loader")
    g.add_edge("document_loader", "market_research")
    g.add_edge("market_research", "lges_analysis")
    g.add_edge("market_research", "catl_analysis")
    g.add_edge("lges_analysis", "comparison")
    g.add_edge("catl_analysis", "comparison")
    g.add_edge("comparison", "report_generation")
    g.add_edge("report_generation", END)
    return g.compile()


# ── Supervisor 변형 ───────────────────────────────────────────────
class SupervisorState(BatteryAnalysisState):
    completed: list
    next: str


SUPERVISOR_SYS = (
    "당신은 멀티 에이전트 파이프라인의 supervisor입니다.\n"
    "아래 worker 중 '아직 실행되지 않은' 다음 worker 하나의 이름만 정확히 출력하세요.\n"
    "모든 worker가 끝났으면 'FINISH'만 출력하세요.\n"
    f"worker 목록(권장 순서): {', '.join(WORKER_ORDER)}"
)


def supervisor_node(state: SupervisorState) -> dict:
    """매 단계 LLM을 호출해 다음 worker를 결정 (= Supervisor 패턴의 라우팅 오버헤드)"""
    completed = state.get("completed", [])
    llm = get_llm(temperature=0.0)

    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        resp = llm.invoke([
            SystemMessage(content=SUPERVISOR_SYS),
            HumanMessage(content=f"이미 완료된 worker: {completed or '없음'}\n다음 worker는?"),
        ])
    routing_stats["calls"] += 1
    routing_stats["tokens"] += cb.total_tokens
    routing_stats["cost"] += cb.total_cost
    routing_stats["time"] += time.perf_counter() - t0

    decision = resp.content.strip()
    # LLM 출력 검증 → 신뢰 가능한 진행 보장(호출 자체는 이미 발생 = 오버헤드 측정됨)
    remaining = [w for w in WORKER_ORDER if w not in completed]
    if not remaining:
        nxt = "FINISH"
    elif decision in remaining:
        nxt = decision
    else:
        nxt = remaining[0]  # fallback: 결정론적 진행
    return {"next": nxt}


def _make_worker(name, fn):
    def wrapper(state):
        out = fn(state)
        out["completed"] = state.get("completed", []) + [name]
        return out
    return wrapper


def build_supervisor():
    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor_node)
    workers = {
        "document_loader": document_loader_node,
        "market_research": market_research_node,
        "lges_analysis": lges_analysis_node,
        "catl_analysis": catl_analysis_node,
        "comparison": comparison_node,
        "report_generation": report_generation_node,
    }
    for name, fn in workers.items():
        g.add_node(name, _make_worker(name, fn))
        g.add_edge(name, "supervisor")  # worker 끝나면 supervisor로 복귀

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {**{w: w for w in WORKER_ORDER}, "FINISH": END},
    )
    return g.compile()


def _run(label, app, initial):
    print(f"\n[{label}] 실행 시작...")
    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        app.invoke(initial, {"recursion_limit": 50})
    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "total_sec": round(elapsed, 2),
        "llm_calls": cb.successful_requests,
        "total_tokens": cb.total_tokens,
        "total_cost_usd": round(cb.total_cost, 6),
    }


def main():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        raise SystemExit("OPENAI_API_KEY / TAVILY_API_KEY 가 필요합니다 (.env).")

    dist = _run("Distributed", build_distributed(), _initial_state())

    routing_stats.update({"calls": 0, "tokens": 0, "cost": 0.0, "time": 0.0})
    sup = _run("Supervisor", build_supervisor(),
               {**_initial_state(), "completed": [], "next": ""})
    routing = {
        "routing_llm_calls": routing_stats["calls"],
        "routing_tokens": routing_stats["tokens"],
        "routing_cost_usd": round(routing_stats["cost"], 6),
        "routing_time_sec": round(routing_stats["time"], 2),
    }

    report = {
        "distributed": dist,
        "supervisor": sup,
        "supervisor_routing_overhead": routing,
        "delta": {
            "extra_llm_calls": sup["llm_calls"] - dist["llm_calls"],
            "extra_tokens": sup["total_tokens"] - dist["total_tokens"],
            "extra_cost_usd": round(sup["total_cost_usd"] - dist["total_cost_usd"], 6),
            "extra_sec": round(sup["total_sec"] - dist["total_sec"], 2),
        },
    }

    out = os.path.join(RESULTS_DIR, "architecture_compare.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("아키텍처 비교: Distributed vs Supervisor")
    print("=" * 60)
    print(f"{'':22s}{'Distributed':>14s}{'Supervisor':>14s}")
    print(f"{'전체 시간(s)':20s}{dist['total_sec']:>14}{sup['total_sec']:>14}")
    print(f"{'총 LLM 호출':20s}{dist['llm_calls']:>14}{sup['llm_calls']:>14}")
    print(f"{'총 토큰':20s}{dist['total_tokens']:>14}{sup['total_tokens']:>14}")
    print(f"{'추정 비용($)':20s}{dist['total_cost_usd']:>14}{sup['total_cost_usd']:>14}")
    print("-" * 60)
    print(f"라우팅 전용 오버헤드: {routing['routing_llm_calls']}회 호출, "
          f"{routing['routing_tokens']} 토큰, {routing['routing_time_sec']}s")
    print(f"Supervisor 추가분: +{report['delta']['extra_llm_calls']}회, "
          f"+{report['delta']['extra_tokens']} 토큰, +{report['delta']['extra_sec']}s")
    print(f"\n상세 → {out}")


if __name__ == "__main__":
    main()
