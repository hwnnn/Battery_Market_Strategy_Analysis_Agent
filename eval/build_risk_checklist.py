"""
토픽별 리스크/비판 체크리스트 생성 (리스크 recall 평가의 정답지).
web_topics.json 의 각 subject에 대해 '잘 알려진 리스크·한계·논쟁점'을
GPT-4o-mini로 4~7개 추출해 risk_checklist.json 으로 저장한다.

생성물은 라벨 아티팩트로 '커밋'하여 재현성을 확보한다(매번 재생성 금지).

실행:
    python -m eval.build_risk_checklist
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from agents.llm_config import get_llm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TOPICS = os.path.join(RESULTS_DIR, "web_topics.json")
OUT = os.path.join(RESULTS_DIR, "risk_checklist.json")

SYS = (
    "당신은 배터리 산업 애널리스트입니다. 주어진 대상에 대해 '실제로 보도·지적된' "
    "리스크·한계·논쟁점을 4~7개 한국어로 나열하세요. 홍보성 장점이 아니라 부정적 측면만. "
    "각 항목은 8~20자의 짧은 명사구로, 서로 중복되지 않게.\n"
    '반드시 JSON 배열로만 출력: ["항목1", "항목2", ...]'
)


def _gen(llm, subject: str) -> list[str]:
    raw = llm.invoke([
        SystemMessage(content=SYS),
        HumanMessage(content=f"대상: {subject}"),
    ]).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
    try:
        arr = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        return [s.strip() for s in arr if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, ValueError):
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="앞 N개 토픽만(스모크)")
    args = parser.parse_args()

    topics = json.load(open(TOPICS, encoding="utf-8"))
    if args.limit:
        topics = topics[:args.limit]
    llm = get_llm(temperature=0.0)

    checklist = {}
    for i, t in enumerate(topics, 1):
        items = _gen(llm, t["subject"])
        checklist[t["topic_id"]] = items
        print(f"  [{i}/{len(topics)}] {t['topic_id']}: {len(items)}개 — {items}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # --limit 스모크 시 기존 항목 보존 병합
    if args.limit and os.path.exists(OUT):
        existing = json.load(open(OUT, encoding="utf-8"))
        existing.update(checklist)
        checklist = existing
    json.dump(checklist, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장 → {OUT} ({len(checklist)} 토픽)")


if __name__ == "__main__":
    main()
