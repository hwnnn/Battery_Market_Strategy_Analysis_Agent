"""
평가용 Q&A 라벨셋 자동 생성 (난이도 선택)
==========================================
FAISS 벡터스토어의 청크를 샘플링 → 각 청크로만 답할 수 있는 질문 1개를
GPT-4o-mini로 생성 → (질문, 정답, 정답 청크) 쌍을 JSON으로 저장.

이 데이터셋이 IR 평가(eval_ir.py)와 RAGAS 평가(eval_ragas.py)에 공통으로 쓰인다.
- IR:    질문 → 검색 결과 안에 "정답 청크"가 몇 번째로 들어오는지 (Hit@5 / MRR)
- RAGAS: 질문 → ground_truth(정답)로 Context Precision/Recall 채점

난이도(--difficulty):
- easy : 청크의 핵심 키워드를 그대로 쓴 직접 질문. 첫 검색이 쉽게 성공.
- hard : 청크의 표면 어휘를 동의어/풀어쓰기/추론으로 치환해 어휘 격차를 만든 질문.
         첫 검색이 실패할 여지를 키워 재검색(Agentic RAG)의 효과를 시험.

설계 기준의 상세 근거는 eval/DATASET_CARD.md 참고.

실행:
    python -m eval.build_dataset --n 30 --difficulty easy
    python -m eval.build_dataset --n 30 --difficulty hard --out results/qa_dataset_hard.json
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
DEFAULT_OUT = os.path.join(RESULTS_DIR, "qa_dataset.json")

# 너무 짧은 청크(목차/페이지 번호 등)는 질문 생성에 부적합
MIN_CHUNK_LEN = 200

# ── 공통 규칙 ─────────────────────────────────────────────────────
_COMMON_RULES = (
    "- 질문은 사실 기반이어야 하며, 주어진 청크 '하나만으로' 명확히 답할 수 있어야 합니다.\n"
    "- 청크에 없는 내용으로 질문하거나 정답을 지어내지 마세요(정답은 반드시 청크에 근거).\n"
    "- '이 문서', '이 청크', '위 자료'처럼 출처를 가리키는 메타 질문은 금지합니다. "
    "독자가 출처를 몰라도 이해되는 자립적 질문이어야 합니다.\n"
    "- 정답(ground_truth)은 청크 내용에 근거해 1~3문장으로 작성하세요.\n"
)

QGEN_EASY = (
    "당신은 배터리 산업 전략 문서를 평가하는 출제자입니다.\n"
    "주어진 청크의 핵심 사실을 묻는 '직접적인' 한국어 질문 1개와 정답을 만드세요.\n"
    "- 청크에 나온 핵심 용어/수치/기업명을 질문에 그대로 사용해도 됩니다.\n"
    + _COMMON_RULES +
    '반드시 JSON으로만 출력: {"question": "...", "ground_truth": "..."}'
)

QGEN_HARD = (
    "당신은 검색 시스템의 강건성을 시험하는 까다로운 출제자입니다.\n"
    "주어진 청크로 답할 수 있되, '첫 검색에서 잘 안 잡히도록' 표면 어휘를 비튼 "
    "한국어 질문 1개와 정답을 만드세요. 아래 난이도 기법 중 1~2개를 적용하세요:\n"
    "  (1) 동의어/상위어 치환: 청크의 고유 용어를 일반어·동의어로 바꿔 표현 "
    "(예: '연결기준 매출'→'한 해 벌어들인 총수익').\n"
    "  (2) 풀어쓰기(paraphrase): 청크 문장 구조와 다른 어순·표현으로 재서술.\n"
    "  (3) 간접/추론: 키워드 직접 매칭 대신 한 단계 추론이 필요한 질문.\n"
    "- 단, 정답은 여전히 '이 청크 하나'에 근거해야 합니다(검증 가능성 유지).\n"
    "- 청크의 가장 식별력 높은 키워드(고유명사·수치)를 질문에 '그대로' 노출하지 마세요.\n"
    + _COMMON_RULES +
    "추가로 적용한 기법을 question_type에 'synonym'|'paraphrase'|'inference' 중 기재하세요.\n"
    '반드시 JSON으로만 출력: {"question": "...", "ground_truth": "...", "question_type": "..."}'
)

PROMPTS = {"easy": QGEN_EASY, "hard": QGEN_HARD}


def _iter_chunks(vectorstore):
    """FAISS docstore에서 청크를 순회"""
    docstore = vectorstore.docstore._dict  # {id: Document}
    for doc in docstore.values():
        yield doc


def _gen_qa(llm, chunk_text: str, difficulty: str) -> dict | None:
    resp = llm.invoke([
        SystemMessage(content=PROMPTS[difficulty]),
        HumanMessage(content=f"문서 청크:\n{chunk_text}"),
    ])
    raw = resp.content.strip()
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


def build_dataset(n: int, difficulty: str, out_path: str):
    vectorstore = load_vectorstore_if_exists()
    if vectorstore is None:
        raise SystemExit("FAISS 벡터스토어가 없습니다. 먼저 `python app.py`로 인덱스를 구축하세요.")

    # 길이 필터 후 결정적 균등 샘플링 (easy/hard 동일 청크 → 공정 비교)
    candidates = [d for d in _iter_chunks(vectorstore)
                  if len(d.page_content) >= MIN_CHUNK_LEN]
    if not candidates:
        raise SystemExit("적합한 청크가 없습니다. MIN_CHUNK_LEN을 낮춰보세요.")

    stride = max(1, len(candidates) // n)
    sampled = candidates[::stride][:n]
    print(f"[build_dataset:{difficulty}] 후보 {len(candidates)}개 → {len(sampled)}개 샘플링")

    llm = get_llm(temperature=0.0)
    dataset = []
    for i, doc in enumerate(sampled, 1):
        qa = _gen_qa(llm, doc.page_content, difficulty)
        if qa is None:
            print(f"  [{i}/{len(sampled)}] 생성 실패 — 스킵")
            continue
        dataset.append({
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            "difficulty": difficulty,
            "question_type": qa.get("question_type", "direct"),
            # IR 정답 식별용: 검색 결과의 page_content와 정확 매칭
            "gold_chunk": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
        })
        print(f"  [{i}/{len(sampled)}] OK  Q: {qa['question'][:50]}...")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n[build_dataset:{difficulty}] {len(dataset)}개 Q&A 저장 → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="생성할 Q&A 개수")
    parser.add_argument("--difficulty", choices=["easy", "hard"], default="easy")
    parser.add_argument("--out", default=None,
                        help="출력 경로 (기본: results/qa_dataset.json, hard는 권장 results/qa_dataset_hard.json)")
    args = parser.parse_args()

    out = args.out or DEFAULT_OUT
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(__file__), out)
    build_dataset(n=args.n, difficulty=args.difficulty, out_path=out)
