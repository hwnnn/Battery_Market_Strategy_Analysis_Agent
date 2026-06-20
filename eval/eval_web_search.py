"""
계층1 — Web Search 3방향 쿼리 평가 (검색 단계).
3-arm 비교로 볼륨 교란을 통제하고 stance 균형·소스 다양성·리스크 recall·
degeneration rate 를 측정한다. 검색 결과는 스냅샷으로 저장해 채점을 재현 가능하게 한다.

  arms: single_base(원쿼리×5) / single_volume(원쿼리×15) / three_persp(3쿼리×5)
  헤드라인 델타: three_persp − single_volume (관점 분할의 순수 기여)

실행:
    python -m eval.eval_web_search                 # 스냅샷 있으면 재사용
    python -m eval.eval_web_search --refresh       # 검색 다시 수행
    python -m eval.eval_web_search --limit 2       # 앞 2토픽만(스모크)
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_config import get_llm
from agents.web_search import build_queries, tavily_search
from eval import web_metrics as M
from eval.judges import classify_stance, risk_covered

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SNAP_DIR = os.path.join(RESULTS_DIR, "web_snapshots")
TOPICS = os.path.join(RESULTS_DIR, "web_topics.json")
CHECKLIST = os.path.join(RESULTS_DIR, "risk_checklist.json")
OUT = os.path.join(RESULTS_DIR, "web_search_metrics.json")
MAX_RESULTS = 5


def _snap_path(topic_id):
    return os.path.join(SNAP_DIR, f"{topic_id}.json")


def build_snapshot(topic, llm):
    base = topic["base_query"]
    queries = build_queries(base, llm)
    snap = {
        "topic_id": topic["topic_id"], "base_query": base,
        "subject": topic["subject"], "queries": queries,
        "arms": {
            "single_base": tavily_search(base, MAX_RESULTS),
            "single_volume": tavily_search(base, MAX_RESULTS * 3),
            "three_persp": {p: tavily_search(q, MAX_RESULTS) for p, q in queries.items()},
        },
    }
    os.makedirs(SNAP_DIR, exist_ok=True)
    json.dump(snap, open(_snap_path(topic["topic_id"]), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return snap


def load_or_build(topic, llm, refresh):
    p = _snap_path(topic["topic_id"])
    if os.path.exists(p) and not refresh:
        return json.load(open(p, encoding="utf-8"))
    return build_snapshot(topic, llm)


def _arm_snippets(arm_data):
    """arm 데이터 → (snippets:list[str], urls:list[str]) 평탄화."""
    if isinstance(arm_data, dict):  # three_persp
        results = [r for lst in arm_data.values() for r in lst]
    else:
        results = arm_data
    snippets = [f"{r['title']} {r['content']}" for r in results]
    urls = [r["url"] for r in results]
    return snippets, urls


def score_arm(arm_name, arm_data, subject, risk_items, llm):
    snippets, urls = _arm_snippets(arm_data)
    stances = [classify_stance(s, subject, llm) for s in snippets]
    hits = [risk_covered(item, snippets, llm) for item in risk_items]
    metrics = {
        "n_snippets": len(snippets),
        "neg_share": M.neg_share(stances),
        "balance_entropy": M.balance_entropy(stances),
        "unique_domains": M.unique_domain_count(urls),
        "risk_recall": M.recall(hits),
        "stance_counts": M.stance_counts(stances),
    }
    if arm_name == "three_persp" and isinstance(arm_data, dict):
        per = {p: [r["url"] for r in lst] for p, lst in arm_data.items()}
        metrics["cross_perspective_dup_rate"] = M.cross_perspective_dup_rate(per)
    return metrics


def _mean(dicts, key):
    vals = [d[key] for d in dicts if key in d]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    topics = json.load(open(TOPICS, encoding="utf-8"))
    checklist = json.load(open(CHECKLIST, encoding="utf-8"))
    if args.limit:
        topics = topics[:args.limit]
    llm = get_llm(temperature=0.0)

    per_topic = []
    degen_flags = []
    for i, t in enumerate(topics, 1):
        snap = load_or_build(t, llm, args.refresh)
        degen = M.is_degenerate(t["base_query"], snap["queries"])
        degen_flags.append(degen)
        risk_items = checklist.get(t["topic_id"], [])
        arms_scores = {
            arm: score_arm(arm, snap["arms"][arm], t["subject"], risk_items, llm)
            for arm in ("single_base", "single_volume", "three_persp")
        }
        per_topic.append({"topic_id": t["topic_id"], "degenerate": degen,
                          "arms": arms_scores})
        print(f"  [{i}/{len(topics)}] {t['topic_id']} "
              f"neg_share 3p={arms_scores['three_persp']['neg_share']} "
              f"vol={arms_scores['single_volume']['neg_share']} degen={degen}")

    # arm별 평균
    agg = {}
    for arm in ("single_base", "single_volume", "three_persp"):
        arm_dicts = [pt["arms"][arm] for pt in per_topic]
        agg[arm] = {k: _mean(arm_dicts, k) for k in
                    ("neg_share", "balance_entropy", "unique_domains", "risk_recall")}
    agg["three_persp"]["cross_perspective_dup_rate"] = _mean(
        [pt["arms"]["three_persp"] for pt in per_topic], "cross_perspective_dup_rate")

    def delta(base_arm):
        return {k: round(agg["three_persp"][k] - agg[base_arm][k], 4)
                for k in ("neg_share", "balance_entropy", "unique_domains", "risk_recall")}

    report = {
        "n_topics": len(topics),
        "query_degeneration_rate": round(sum(degen_flags) / len(degen_flags), 4),
        "arms": agg,
        "delta_three_persp_minus_single_base": delta("single_base"),
        "delta_three_persp_minus_single_volume": delta("single_volume"),
        "per_topic": per_topic,
    }
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"계층1 Web Search 평가 (n={len(topics)})")
    print("=" * 60)
    for arm in ("single_base", "single_volume", "three_persp"):
        a = agg[arm]
        print(f"  {arm:14s} neg={a['neg_share']:.3f} ent={a['balance_entropy']:.3f} "
              f"dom={a['unique_domains']:.1f} risk={a['risk_recall']:.3f}")
    print(f"  degeneration_rate = {report['query_degeneration_rate']:.3f}")
    print(f"  Δ(3p−volume) neg={report['delta_three_persp_minus_single_volume']['neg_share']:+.3f} "
          f"risk={report['delta_three_persp_minus_single_volume']['risk_recall']:+.3f}")
    print(f"\n상세 → {OUT}")


if __name__ == "__main__":
    main()
