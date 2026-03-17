"""
Report Generator Agent — T6
목차 구조에 따라 최종 보고서 작성
SUMMARY(½p 이내) + REFERENCE 자동 생성 → Markdown 저장
"""

import os
from datetime import datetime
from typing import List, Dict
from langchain_core.messages import HumanMessage

from agents.state import BatteryAnalysisState
from agents.llm_config import get_llm

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def _load_prompt(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _format_references(references: List[Dict]) -> str:
    """
    수집된 references 목록을 과제 규정 형식으로 변환
    - 기관 보고서: 발행기관(YYYY). 보고서명. URL
    - 웹페이지: 기관명(YYYY-MM-DD). 제목. 사이트명, URL
    """
    if not references:
        return "※ 참고 자료 없음"

    pdf_refs = []
    web_refs = []
    seen = set()

    for ref in references:
        key = f"{ref.get('source', '')}_{ref.get('page', '')}"
        if key in seen:
            continue
        seen.add(key)

        ref_type = ref.get("type", "pdf")
        if ref_type == "pdf":
            source = ref.get("source", "Unknown")
            page = ref.get("page", "?")
            pdf_refs.append(f"  - {source} (p.{page})")
        elif ref_type == "web":
            title = ref.get("title", "제목 없음")
            url = ref.get("url", "URL 없음")
            date = ref.get("date", "날짜 미상")
            web_refs.append(f"  - 출처({date}). {title}. {url}")

    sections = []
    if pdf_refs:
        sections.append("### 기관 보고서 / PDF 문서\n" + "\n".join(pdf_refs))
    if web_refs:
        sections.append("### 웹페이지\n" + "\n".join(web_refs))

    return "\n\n".join(sections) if sections else "※ 참고 자료 없음"


def report_generation_node(state: BatteryAnalysisState) -> dict:
    """T6: 최종 보고서 생성 및 저장"""
    print("[ReportGenerator] 보고서 생성 시작...")
    llm = get_llm(temperature=0.1)  # 약간의 창의성 허용
    prompt_template = _load_prompt("report_generator.txt")

    # REFERENCE 섹션 생성
    references = state.get("references", [])
    formatted_refs = _format_references(references)

    # 프롬프트 조립
    prompt = prompt_template.format(
        market_background=state.get("market_background", "시장 배경 정보 없음"),
        lges_strategy=state.get("lges_strategy", "LGES 분석 결과 없음"),
        catl_strategy=state.get("catl_strategy", "CATL 분석 결과 없음"),
        comparison_result=state.get("comparison_result", "비교 분석 결과 없음"),
        references=formatted_refs,
        date=datetime.now().strftime("%Y년 %m월 %d일"),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    report_content = response.content

    # outputs/ 디렉터리에 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "report.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[ReportGenerator] 보고서 저장 완료: {output_path}")
    return {
        "report_draft": report_content,
        "error_messages": [],
    }
