"""
평가용 Q&A 라벨셋 자동 생성
================================
FAISS 벡터스토어의 청크를 샘플링 → 각 청크로만 답할 수 있는 질문 1개를
GPT-4o-mini로 생성 → (질문, 정답, 정답 청크) 쌍을 JSON으로 저장.

이 한 개의 데이터셋이 IR 평가(eval_ir.py)와 RAGAS 평가(eval_ragas.py)에
공통으로 쓰인다.

- IR:    질문 → 검색 결과 안에 "정답 청크"가 몇 번째로 들어오는지 (Hit@5 / MRR)
- RAGAS: 질문 → ground_truth(정답)로 Context Precision/Recall 채점

실행:
    python -m eval.build_dataset --n 30
"""

import os
import sys
import json
import argparse

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage

from agents.document_loader import load_vectorstore_if_exists
from agents.llm_config import get_llm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATASET_PATH = os.path.join(RESULTS_DIR, "qa_dataset.json")

# 너무 짧은 청크(목차/페이지 번호 등)는 질문 생성에 부적합
MIN_CHUNK_LEN = 200

QGEN_SYSTEM = (
    "당신은 배터리 산업 전략 문서를 평가하는 출제자입니다.\n"
    "주어진 문서 청크 '하나만으로' 명확히 답할 수 있는 한국어 질문 1개와 "
    "그 정답을 만드세요.\n"
    "- 질문은 구체적이고 사실 기반이어야 합니다(수치/전략/기업명 등).\n"
    "- 청크에 없는 내용으로 질문을 만들지 마세요.\n"
    "- '이 문서', '이 청크', '위 자료' 처럼 문서 자체를 가리키는 메타 질문은 금지합니다. "
    "독자가 출처를 몰라도 이해되는 자립적 질문이어야 합니다.\n"
    "- 정답은 청크 내용에 근거해 1~3문장으로 작성하세요.\n"
    "반드시 아래 JSON 형식으로만 출력하세요:\n"
    '{"question": "...", "ground_truth": "..."}'
)


def _iter_chunks(vectorstore):
    """FAISS docstore에서 (page_content, metadata) 청크를 순회"""
    docstore = vectorstore.docstore._dict  # {id: Document}
    for doc in docstore.values():
        yield doc


def _gen_qa(llm, chunk_text: str) -> dict | None:
    resp = llm.invoke([
        SystemMessage(content=QGEN_SYSTEM),
        HumanMessage(content=f"문서 청크:\n{chunk_text}"),
    ])
    raw = resp.content.strip()
    # 코드펜스 제거
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    try:
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        if obj.get("question") and obj.get("ground_truth"):
            return obj
    except (json.JSONDecodeError, ValueError):
        return None
    return None


def build_dataset(n: int = 30, seed_stride: int = 1):
    vectorstore = load_vectorstore_if_exists()
    if vectorstore is None:
        raise SystemExit(
            "FAISS 벡터스토어가 없습니다. 먼저 `python app.py`로 인덱스를 구축하세요."
        )

    # 길이 필터 후, 결정적으로 균등 샘플링 (Math.random 불필요)
    candidates = [d for d in _iter_chunks(vectorstore)
                  if len(d.page_content) >= MIN_CHUNK_LEN]
    if not candidates:
        raise SystemExit("적합한 청크가 없습니다. MIN_CHUNK_LEN을 낮춰보세요.")

    stride = max(1, len(candidates) // n)
    sampled = candidates[::stride][:n]
    print(f"[build_dataset] 후보 청크 {len(candidates)}개 → {len(sampled)}개 샘플링")

    llm = get_llm(temperature=0.0)
    dataset = []
    for i, doc in enumerate(sampled, 1):
        qa = _gen_qa(llm, doc.page_content)
        if qa is None:
            print(f"  [{i}/{len(sampled)}] 생성 실패 — 스킵")
            continue
        dataset.append({
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            # IR 정답 식별용: 검색 결과의 page_content와 정확 매칭
            "gold_chunk": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
        })
        print(f"  [{i}/{len(sampled)}] OK  Q: {qa['question'][:50]}...")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n[build_dataset] {len(dataset)}개 Q&A 저장 → {DATASET_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="생성할 Q&A 개수")
    args = parser.parse_args()
    build_dataset(n=args.n)
