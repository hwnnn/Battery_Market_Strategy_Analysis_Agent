"""
LLM-judge 모듈 — web search 평가용 (temp=0).
순수 파서(parse_*)는 단위테스트, LLM 호출부는 스모크 검증.
"""
import re
from langchain_core.messages import HumanMessage, SystemMessage
from eval.web_metrics import STANCES

_STANCE_SYS = (
    "당신은 텍스트 태도 분류기입니다. 주어진 검색 스니펫이 '대상'에 대해 어떤 태도를 "
    "보이는지 한 단어로만 답하세요: 긍정 / 부정 / 중립.\n"
    "- 긍정: 성과·기대·강점 강조\n- 부정: 리스크·한계·문제점·비판\n- 중립: 사실·현황 위주\n"
    "반드시 '긍정', '부정', '중립' 중 한 단어만 출력."
)
_RISK_SYS = (
    "주어진 '리스크 항목'이 아래 검색 스니펫들에 의해 실제로 언급/뒷받침되는지 판정하세요. "
    "뒷받침되면 'YES', 아니면 'NO' 한 단어만 출력."
)
_EXTRACT_SYS = (
    "다음 보고서에서 언급된 '리스크·한계·부정적 요인'을 중복 없이 짧은 명사구로 나열하세요. "
    "성과·장점은 제외.\n반드시 JSON 배열로만 출력: [\"항목1\", ...]"
)
_BALANCE_SYS = (
    "다음 보고서가 '성과·기대'와 '한계·리스크'를 얼마나 균형 있게 다루는지 0.0~1.0 점수로 "
    "평가하세요. 1.0=완전 균형, 0.0=한쪽으로 완전 치우침. 숫자만 출력."
)


def parse_stance_label(raw: str) -> str:
    for s in STANCES:
        if s in raw:
            return s
    return "중립"


def parse_score(raw: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", raw)
    if not m:
        return 0.0
    return round(min(1.0, max(0.0, float(m.group()))), 4)


def classify_stance(snippet: str, subject: str, llm) -> str:
    raw = llm.invoke([
        SystemMessage(content=_STANCE_SYS),
        HumanMessage(content=f"대상: {subject}\n\n스니펫:\n{snippet[:800]}"),
    ]).content
    return parse_stance_label(raw)


def risk_covered(risk_item: str, snippets: list[str], llm) -> bool:
    joined = "\n---\n".join(s[:400] for s in snippets)
    raw = llm.invoke([
        SystemMessage(content=_RISK_SYS),
        HumanMessage(content=f"리스크 항목: {risk_item}\n\n검색 스니펫들:\n{joined}"),
    ]).content
    return "YES" in raw.upper()


def extract_risk_claims(report: str, llm) -> list[str]:
    import json
    raw = llm.invoke([
        SystemMessage(content=_EXTRACT_SYS),
        HumanMessage(content=report[:12000]),
    ]).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
    try:
        arr = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        return [s.strip() for s in arr if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, ValueError):
        return []


def report_balance_score(report: str, llm) -> float:
    raw = llm.invoke([
        SystemMessage(content=_BALANCE_SYS),
        HumanMessage(content=report[:12000]),
    ]).content
    return parse_score(raw)
