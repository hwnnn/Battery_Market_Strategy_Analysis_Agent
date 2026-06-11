"""
처리시간 계측 — 전체 / 노드별 + async 절감 추정
=================================================
각 노드를 perf_counter로 감싸 실행시간을 측정한다.
LGES·CATL은 현재 동기 구현이라 순차 실행되므로,
  - 현 동기 합계      = lges_time + catl_time
  - async 전환 시 예상 = max(lges_time, catl_time)
의 차이로 "병렬화 시 절감 예상치"를 추정한다 (실제 async 구현 없이 산출).

실행:
    python -m eval.measure_latency
"""

import os
import sys
import json
import time
from functools import wraps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from agents.state import BatteryAnalysisState
from agents import (
    document_loader, market_research_agent, lges_analysis_agent,
    catl_analysis_agent, comparison_agent, report_generator,
)

load_dotenv()

# 노드명 → (모듈, 함수명)
NODES = {
    "document_loader":   (document_loader, "document_loader_node"),
    "market_research":   (market_research_agent, "market_research_node"),
    "lges_analysis":     (lges_analysis_agent, "lges_analysis_node"),
    "catl_analysis":     (catl_analysis_agent, "catl_analysis_node"),
    "comparison":        (comparison_agent, "comparison_node"),
    "report_generation": (report_generator, "report_generation_node"),
}

timings: dict[str, float] = {}


def _timed(name, fn):
    @wraps(fn)
    def wrapper(state):
        start = time.perf_counter()
        result = fn(state)
        timings[name] = time.perf_counter() - start
        print(f"[timing] {name:20s} {timings[name]:7.2f}s")
        return result
    return wrapper


def build_timed_graph():
    graph = StateGraph(BatteryAnalysisState)
    for name, (mod, fname) in NODES.items():
        graph.add_node(name, _timed(name, getattr(mod, fname)))

    graph.add_edge(START, "document_loader")
    graph.add_edge("document_loader", "market_research")
    graph.add_edge("market_research", "lges_analysis")
    graph.add_edge("market_research", "catl_analysis")
    graph.add_edge("lges_analysis", "comparison")
    graph.add_edge("catl_analysis", "comparison")
    graph.add_edge("comparison", "report_generation")
    graph.add_edge("report_generation", END)
    return graph.compile()


def main():
    query = "전기차 캐즘 환경에서 LGES vs CATL 포트폴리오 다각화 전략 비교 분석"
    initial: BatteryAnalysisState = {
        "query": query, "vectorstore": None,
        "market_background": "", "lges_strategy": "", "catl_strategy": "",
        "comparison_result": "", "references": [], "report_draft": "",
        "error_messages": [], "rag_iteration": 0,
    }

    app = build_timed_graph()
    t0 = time.perf_counter()
    app.invoke(initial)
    total = time.perf_counter() - t0

    lges = timings.get("lges_analysis", 0.0)
    catl = timings.get("catl_analysis", 0.0)
    sync_fanout = lges + catl
    async_fanout = max(lges, catl)
    projected_saving = sync_fanout - async_fanout
    projected_total = total - projected_saving

    report = {
        "total_sync_sec": round(total, 2),
        "per_node_sec": {k: round(v, 2) for k, v in timings.items()},
        "fanout": {
            "lges_sec": round(lges, 2),
            "catl_sec": round(catl, 2),
            "sync_sum_sec": round(sync_fanout, 2),
            "async_max_sec": round(async_fanout, 2),
            "projected_saving_sec": round(projected_saving, 2),
        },
        "projected_total_async_sec": round(projected_total, 2),
    }

    out = os.path.join(os.path.dirname(__file__), "results", "latency.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 55)
    print("처리시간 계측 결과")
    print("=" * 55)
    print(f"  전체(동기)            : {total:6.2f}s")
    print(f"  └ LGES 분석           : {lges:6.2f}s")
    print(f"  └ CATL 분석           : {catl:6.2f}s")
    print(f"  fan-out 동기 합        : {sync_fanout:6.2f}s")
    print(f"  fan-out async 예상(max): {async_fanout:6.2f}s")
    print(f"  → async 전환 시 절감 예상: {projected_saving:6.2f}s "
          f"(전체 {total:.1f}s → {projected_total:.1f}s)")
    print(f"\n상세 → {out}")


if __name__ == "__main__":
    main()
