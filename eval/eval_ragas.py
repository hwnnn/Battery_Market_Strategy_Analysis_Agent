"""
RAGAS 평가 — Faithfulness · Answer Relevancy · Context Precision · Context Recall
================================================================================
qa_dataset.json의 각 질문에 대해:
  1) rag_retrieve로 컨텍스트 검색  2) 그 컨텍스트로 GPT-4o-mini가 답변 생성
  3) RAGAS 4지표 채점

Agentic RAG(재검색 최대 3회) vs 일반 RAG(재검색 1회) 두 모드를 비교 측정해
"재검색 적용 전후" 점수를 한 번에 산출한다.

사전 설치:
    pip install ragas datasets

실행:
    python -m eval.eval_ragas              # Agentic + 일반 둘 다
    python -m eval.eval_ragas --mode agentic
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage

import agents.rag_tool as rag_tool
from agents.rag_tool import rag_retrieve
from agents.document_loader import load_vectorstore_if_exists, EMBEDDING_MODEL
from agents.llm_config import get_llm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

ANSWER_PROMPT = (
    "아래 컨텍스트에만 근거해 질문에 한국어로 답하세요. "
    "컨텍스트에 없으면 '자료에 없음'이라고 답하세요.\n\n"
    "컨텍스트:\n{context}\n\n질문: {question}\n\n답변:"
)


def _load_dataset(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} 가 없습니다. 먼저 `python -m eval.build_dataset` 실행.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _out_path(dataset_path):
    """qa_dataset_hard.json → ragas_metrics_hard.json"""
    base = os.path.basename(dataset_path).replace("qa_dataset", "ragas_metrics")
    return os.path.join(RESULTS_DIR, base)


def _build_samples(dataset, vectorstore, llm):
    """각 질문에 대해 RAGAS 입력 샘플 생성 (검색 + 답변)"""
    samples = []
    for i, item in enumerate(dataset, 1):
        q = item["question"]
        context, _refs = rag_retrieve(q, vectorstore, llm)
        answer = llm.invoke([
            HumanMessage(content=ANSWER_PROMPT.format(context=context, question=q))
        ]).content
        # RAGAS는 contexts를 리스트로 받음 — 출처 구분자로 분할
        retrieved = [c for c in context.split("\n\n---\n\n") if c.strip()]
        samples.append({
            "user_input": q,
            "retrieved_contexts": retrieved or [context],
            "response": answer,
            "reference": item["ground_truth"],
        })
        print(f"  [{i}/{len(dataset)}] 샘플 생성 완료")
    return samples


def _run_ragas(samples):
    """RAGAS 0.2.x API로 채점"""
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import (
            Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as e:
        raise SystemExit(
            f"RAGAS 미설치: {e}\n  pip install ragas datasets  로 설치하세요."
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    eval_llm = LangchainLLMWrapper(get_llm(temperature=0.0))
    eval_emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))

    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_emb),
        ContextPrecision(llm=eval_llm),
        ContextRecall(llm=eval_llm),
    ]

    ds = EvaluationDataset.from_list(samples)
    result = evaluate(dataset=ds, metrics=metrics)

    # 버전 안정적 점수 추출: to_pandas()의 메트릭 컬럼 평균
    df = result.to_pandas()
    metric_cols = [c for c in df.columns
                   if c not in ("user_input", "retrieved_contexts",
                                "response", "reference")
                   and df[c].dtype.kind in "fi"]
    return {c: round(float(df[c].mean()), 4) for c in metric_cols}


def evaluate_mode(mode: str, dataset, vectorstore):
    """mode: 'agentic'(재검색 루프) | 'plain'(단발) — 프로덕션 토글(RAG_AGENTIC) 제어"""
    original = os.environ.get("RAG_AGENTIC")
    os.environ["RAG_AGENTIC"] = "true" if mode == "agentic" else "false"
    print(f"\n[{mode}] RAG_AGENTIC={os.environ['RAG_AGENTIC']} 로 평가 시작")
    try:
        llm = get_llm(temperature=0.0)
        samples = _build_samples(dataset, vectorstore, llm)
        scores = _run_ragas(samples)
    finally:
        if original is None:
            os.environ.pop("RAG_AGENTIC", None)
        else:
            os.environ["RAG_AGENTIC"] = original

    return scores


def main(mode: str, dataset_path: str, out=None):
    dataset = _load_dataset(dataset_path)
    vectorstore = load_vectorstore_if_exists()
    if vectorstore is None:
        raise SystemExit("FAISS 벡터스토어가 없습니다. 먼저 `python app.py` 실행.")

    modes = ["plain", "agentic"] if mode == "both" else [mode]
    report = {}
    for m in modes:
        report[m] = evaluate_mode(m, dataset, vectorstore)

    out = out or _out_path(dataset_path)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 55)
    print("RAGAS 평가 결과")
    print("=" * 55)
    for m, scores in report.items():
        label = "Agentic RAG(재검색 3회)" if m == "agentic" else "일반 RAG(재검색 1회)"
        print(f"\n[{label}]")
        for k, v in scores.items():
            print(f"  {k:24s}: {v}")
    print(f"\n상세 → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agentic", "plain", "both"], default="both")
    parser.add_argument("--dataset", default=os.path.join(RESULTS_DIR, "qa_dataset.json"))
    parser.add_argument("--out", default=None, help="결과 출력 경로 override")
    args = parser.parse_args()
    ds = args.dataset if os.path.isabs(args.dataset) else os.path.join(os.path.dirname(__file__), args.dataset)
    out = args.out
    if out and not os.path.isabs(out):
        out = os.path.join(os.path.dirname(__file__), out)
    main(args.mode, ds, out=out)
