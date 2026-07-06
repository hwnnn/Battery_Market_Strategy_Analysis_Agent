"""
Report Generator Agent — T6
목차 구조에 따라 최종 보고서 작성
SUMMARY(½p 이내) + REFERENCE 자동 생성 → Markdown/PDF 저장
"""

import os
from datetime import datetime
from typing import List, Dict
import markdown
import fitz  # PyMuPDF — 이미 설치됨 (PDF 파싱 용도)
from langchain_core.messages import HumanMessage

from agents.state import BatteryAnalysisState
from agents.llm_config import get_llm

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

_PDF_CSS = """
body  { font-family: sans-serif; font-size: 11pt; line-height: 1.7; color: #1a1a1a; }
h1    { font-size: 20pt; margin: 0 0 6pt 0; }
h2    { font-size: 15pt; margin: 18pt 0 6pt 0; }
h3    { font-size: 13pt; margin: 14pt 0 4pt 0; }
h4    { font-size: 11pt; margin: 10pt 0 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 10pt; }
th, td { border: 1px solid #bbb; padding: 5pt 7pt; text-align: left; }
th    { background: #f0f0f0; font-weight: bold; }
code  { font-family: monospace; font-size: 9.5pt; background: #f5f5f5; }
pre   { background: #f5f5f5; padding: 8pt; font-size: 9pt; }
ul, ol { margin: 4pt 0; padding-left: 20pt; }
li    { margin: 2pt 0; }
"""


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
        ref_type = ref.get("type", "pdf")
        if ref_type == "web":
            key = f"web_{ref.get('url', '')}"
        else:
            key = f"pdf_{ref.get('source', '')}_{ref.get('page', '')}"
        if key in seen:
            continue
        seen.add(key)

        if ref_type == "pdf":
            source = ref.get("source", "Unknown")
            page = ref.get("page", "?")
            pdf_refs.append(f"  - {source} (p.{page})")
        elif ref_type == "web":
            source = ref.get("source", "출처")
            title = ref.get("title", "제목 없음")
            url = ref.get("url", "URL 없음")
            date = ref.get("date", "날짜 미상")
            perspective = ref.get("perspective")
            suffix = f" [{perspective}]" if perspective else ""
            web_refs.append(f"  - {source}({date}). {title}. {url}{suffix}")

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
    md_path = os.path.join(OUTPUT_DIR, "report.md")
    pdf_path = os.path.join(OUTPUT_DIR, "report.pdf")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    # Markdown → HTML → PDF 변환 (fitz.Story, 시스템 라이브러리 불필요)
    html_body = markdown.markdown(
        report_content,
        extensions=["tables", "fenced_code", "nl2br"],
    )
    html_full = (
        f"<html><head><style>{_PDF_CSS}</style></head>"
        f"<body>{html_body}</body></html>"
    )

    story = fitz.Story(html=html_full)
    mediabox = fitz.paper_rect("a4")
    margin = 50  # points (~18mm)
    where = mediabox + (margin, margin, -margin, -margin)

    writer = fitz.DocumentWriter(pdf_path)
    more = 1
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()

    print(f"[ReportGenerator] 보고서 저장 완료: {md_path}, {pdf_path}")
    return {
        "report_draft": report_content,
        "error_messages": [],
    }
