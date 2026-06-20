"""
계층2 — 보고서 단계 ablation.
WEB_SEARCH_PERSPECTIVES 토글로 전체 파이프라인을 두 모드로 N회 실행하고
최종 보고서(report_draft)의 균형을 비교한다.

  off : 단일 쿼리 web search (ablation)
  on  : 3방향 web search (현행)

지표: risk_section_claims(추출 리스크 수), report_balance_score(0~1),
      report_risk_recall(prod 토픽 체크리스트 대비)

실행(비용 큼 — 파이프라인 2*N회):
    python -m eval.eval_report_balance --n 3
    python -m eval.eval_report_balance --n 1   # 스모크
"""
import os
import sys
import json
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from agents.state import BatteryAnalysisState
from agents.llm_config import get_llm
from eval.judges import extract_risk_claims, report_balance_score, risk_covered
from eval.web_metrics import recall  # Fix 4: moved from inside _score_report

load_dotenv()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHECKLIST = os.path.join(RESULTS_DIR, "risk_checklist.json")
OUT = os.path.join(RESULTS_DIR, "report_balance.json")
QUERY = "전기차 캐즘 환경에서 LGES vs CATL 포트폴리오 다각화 전략 비교 분석"
# 보고서는 LGES/CATL/시장 중심 → prod 체크리스트로 recall 측정
RECALL_TOPICS = ["prod_lges", "prod_catl", "prod_market"]


def _run_pipeline():
    from app import build_graph  # 지연 import (env 토글 후 그래프 생성)
    app = build_graph()
    init: BatteryAnalysisState = {
        "query": QUERY, "vectorstore": None, "market_background": "",
        "lges_strategy": "", "catl_strategy": "", "comparison_result": "",
        "references": [], "report_draft": "", "error_messages": [], "rag_iteration": 0,
    }
    return app.invoke(init)["report_draft"]


def _score_report(report, checklist, llm):
    claims = extract_risk_claims(report, llm)
    balance = report_balance_score(report, llm)
    items = [it for tid in RECALL_TOPICS for it in checklist.get(tid, [])]
    hits = [risk_covered(it, [report], llm) for it in items]
    return {"risk_section_claims": len(claims), "report_balance_score": balance,
            "report_risk_recall": recall(hits)}


def _run_mode(enabled: bool, n: int, checklist, llm):
    prev = os.environ.get("WEB_SEARCH_PERSPECTIVES")
    os.environ["WEB_SEARCH_PERSPECTIVES"] = "true" if enabled else "false"
    runs = []
    try:
        for i in range(n):
            print(f"  [{'on' if enabled else 'off'} {i+1}/{n}] 파이프라인 실행...")
            try:
                report = _run_pipeline()
                runs.append(_score_report(report, checklist, llm))
            except Exception as e:
                print(f"    run 실패 — 건너뜀: {e}")
    finally:
        if prev is None:
            os.environ.pop("WEB_SEARCH_PERSPECTIVES", None)
        else:
            os.environ["WEB_SEARCH_PERSPECTIVES"] = prev
    return runs


def _agg(runs):
    keys = ("risk_section_claims", "report_balance_score", "report_risk_recall")
    # Fix 2: guard against empty runs list
    if not runs:
        return {k: {"mean": None, "std": None, "n": 0} for k in keys}
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k] = {"mean": round(statistics.mean(vals), 4),
                  "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                  "n": len(runs)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="모드당 실행 횟수")
    args = parser.parse_args()

    # Fix 3: use context managers for file I/O
    with open(CHECKLIST, encoding="utf-8") as f:
        checklist = json.load(f)
    llm = get_llm(temperature=0.0)

    off = _agg(_run_mode(False, args.n, checklist, llm))
    on = _agg(_run_mode(True, args.n, checklist, llm))
    # Fix 2: skip delta for metrics where either mode has None mean
    delta = {}
    for k in off:
        off_mean = off[k]["mean"]
        on_mean = on[k]["mean"]
        if off_mean is None or on_mean is None:
            delta[k] = None
        else:
            delta[k] = round(on_mean - off_mean, 4)

    report = {"n_runs_per_mode": args.n,
              "perspectives_off": off, "perspectives_on": on,
              "delta_on_minus_off": delta}
    # Fix 3: use context manager for json.dump write
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"계층2 보고서 ablation (n={args.n}/mode)")
    print("=" * 60)
    for k in delta:
        off_mean = off[k]["mean"]
        on_mean = on[k]["mean"]
        d = delta[k]
        if off_mean is None or on_mean is None:
            print(f"  {k:24s} off=N/A  on=N/A  Δ=N/A  (no successful runs)")
        else:
            print(f"  {k:24s} off={off_mean:.3f} on={on_mean:.3f} "
                  f"Δ={d:+.3f}")
    print(f"\n상세 → {OUT}")


if __name__ == "__main__":
    main()
