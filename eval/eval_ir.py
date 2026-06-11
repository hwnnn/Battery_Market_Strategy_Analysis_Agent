"""
IR(정보 검색) 메트릭 평가 — Hit Rate@5 / MRR (일반 vs Agentic 비교)
====================================================================
qa_dataset.json의 각 질문으로 검색을 수행하고 "정답 청크(gold_chunk)"의
순위를 측정한다. 두 모드를 비교해 재검색(rewrite) 효과를 정량화한다.

  - plain   : 원본 쿼리로 단발 검색 (일반 RAG)
  - agentic : grade_documents 판정 → rewrite_query 재검색 (최대 3회)

지표:
  - Hit Rate@5 : 정답 청크가 상위 5개 안에 포함된 질문 비율
  - MRR        : 정답 청크 순위의 역수 평균 (1위=1.0, 미포함=0)

실제 검색 품질 비교가 목적이므로 rag_tool의 helper(_retrieve/_grade/_rewrite)를
그대로 재사용한다.

실행:
    python -m eval.eval_ir
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.document_loader import load_vectorstore_if_exists
from agents.rag_tool import (
    TOP_K, MAX_ITERATIONS, _retrieve, _grade_documents, _rewrite_query,
)
from agents.llm_config import get_llm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _load_dataset(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} 가 없습니다. 먼저 `python -m eval.build_dataset` 실행.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _out_path(dataset_path):
    """qa_dataset_hard.json → ir_metrics_hard.json"""
    base = os.path.basename(dataset_path).replace("qa_dataset", "ir_metrics")
    return os.path.join(RESULTS_DIR, base)


def _rank_of_gold(docs, gold_chunk: str) -> int | None:
    """검색 결과에서 정답 청크의 순위(1-based). 없으면 None."""
    gold = gold_chunk.strip()
    for rank, doc in enumerate(docs, 1):
        if doc.page_content.strip() == gold:
            return rank
    return None


def _retrieve_plain(vectorstore, question, llm):
    """일반 RAG: 원본 쿼리 단발 검색"""
    return _retrieve(vectorstore, question)


def _retrieve_agentic(vectorstore, question, llm):
    """Agentic RAG: grade→rewrite 재검색 (rag_tool 로직 그대로 복제)"""
    current_query = question
    best_docs = []
    for iteration in range(1, MAX_ITERATIONS + 1):
        docs = _retrieve(vectorstore, current_query)
        best_docs = docs
        if _grade_documents(docs, question, llm):
            break
        if iteration < MAX_ITERATIONS:
            current_query = _rewrite_query(current_query, llm, iteration,
                                           original_query=question)
    return best_docs


def _score(dataset, vectorstore, retrieve_fn, llm, k):
    hits = 0
    rr_sum = 0.0
    per_q = []
    for item in dataset:
        docs = retrieve_fn(vectorstore, item["question"], llm)
        rank = _rank_of_gold(docs, item["gold_chunk"])
        hit = rank is not None and rank <= k
        rr = (1.0 / rank) if rank else 0.0
        hits += int(hit)
        rr_sum += rr
        per_q.append({
            "question": item["question"][:60],
            "gold": f"{item['source']} p.{item['page']}",
            "rank": rank,
        })
    n = len(dataset)
    return {
        f"hit_rate@{k}": round(hits / n, 4) if n else 0.0,
        "mrr": round(rr_sum / n, 4) if n else 0.0,
        "per_question": per_q,
    }


def evaluate_ir(dataset_path, k: int = TOP_K, out_path=None):
    dataset = _load_dataset(dataset_path)
    out_path = out_path or _out_path(dataset_path)
    vectorstore = load_vectorstore_if_exists()
    if vectorstore is None:
        raise SystemExit("FAISS 벡터스토어가 없습니다. 먼저 `python app.py` 실행.")
    llm = get_llm(temperature=0.0)

    print(f"[IR] plain 모드 평가 (n={len(dataset)})...")
    plain = _score(dataset, vectorstore, _retrieve_plain, llm, k)
    print(f"[IR] agentic 모드 평가 (재검색 최대 {MAX_ITERATIONS}회)...")
    agentic = _score(dataset, vectorstore, _retrieve_agentic, llm, k)

    def _delta(a, b):
        return round(b - a, 4)

    result = {
        "n_questions": len(dataset),
        "top_k": k,
        "plain": {f"hit_rate@{k}": plain[f"hit_rate@{k}"], "mrr": plain["mrr"]},
        "agentic": {f"hit_rate@{k}": agentic[f"hit_rate@{k}"], "mrr": agentic["mrr"]},
        "delta": {
            f"hit_rate@{k}": _delta(plain[f"hit_rate@{k}"], agentic[f"hit_rate@{k}"]),
            "mrr": _delta(plain["mrr"], agentic["mrr"]),
        },
        "per_question_agentic": agentic["per_question"],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 55)
    print(f"IR 평가 (n={len(dataset)}, k={k})   일반 RAG → Agentic RAG")
    print("=" * 55)
    print(f"  Hit Rate@{k} : {plain[f'hit_rate@{k}']:.3f} → {agentic[f'hit_rate@{k}']:.3f}"
          f"  (Δ {result['delta'][f'hit_rate@{k}']:+.3f})")
    print(f"  MRR         : {plain['mrr']:.3f} → {agentic['mrr']:.3f}"
          f"  (Δ {result['delta']['mrr']:+.3f})")
    print(f"\n상세 → {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(RESULTS_DIR, "qa_dataset.json"))
    parser.add_argument("--out", default=None, help="결과 출력 경로 override")
    args = parser.parse_args()
    ds = args.dataset if os.path.isabs(args.dataset) else os.path.join(os.path.dirname(__file__), args.dataset)
    out = args.out
    if out and not os.path.isabs(out):
        out = os.path.join(os.path.dirname(__file__), out)
    evaluate_ir(ds, out_path=out)
