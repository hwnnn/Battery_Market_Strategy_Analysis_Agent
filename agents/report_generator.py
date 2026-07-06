"""
Report Generator Agent — T6
목차 구조에 따라 최종 보고서 작성
SUMMARY(½p 이내) + REFERENCE 자동 생성 → Markdown/PDF 저장
"""

import os
import re
from datetime import datetime
from typing import List, Dict
import markdown
import fitz  # PyMuPDF — 이미 설치됨 (PDF 파싱 용도)
from langchain_core.messages import HumanMessage

from agents.state import BatteryAnalysisState
from agents.llm_config import get_llm

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:\*\*REFERENCE\*\*|#{1,6}\s*REFERENCE|REFERENCE)\s*$"
)
_INLINE_PDF_RE = re.compile(r"\[출처:\s*([^,\]]+),\s*p\.([0-9?]+)\]")
_NUMERIC_CITATION_RE = re.compile(r"\[[0-9]+\]")
_NON_CANONICAL_SOURCE_RE = re.compile(
    r"\[출처:\s*(?![^,\]]+,\s*p\.[0-9?]+\])[^\]]+\]"
)

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


def _pdf_key(ref: Dict) -> str:
    return f"{ref.get('source', 'Unknown')}#p{ref.get('page', '?')}"


def _pdf_references(references: List[Dict]) -> List[Dict]:
    pdf_refs = []
    seen = set()
    for ref in references:
        if ref.get("type", "pdf") != "pdf":
            continue
        key = _pdf_key(ref)
        if key in seen:
            continue
        seen.add(key)
        pdf_refs.append(ref)
    return pdf_refs


def _strip_reference_section(report: str) -> str:
    """LLM이 임의로 작성한 REFERENCE 섹션을 제거하고 코드가 다시 붙인다."""
    matches = list(_REFERENCE_HEADING_RE.finditer(report))
    return report[: matches[-1].start()].rstrip() if matches else report.rstrip()


def _remove_noncanonical_citations(report: str) -> str:
    """REFERENCE에 연결할 수 없는 모델 생성 citation 표기를 제거한다."""
    report = _NUMERIC_CITATION_RE.sub("", report)
    return _NON_CANONICAL_SOURCE_RE.sub("", report)


def _inline_pdf_keys(report: str) -> set[str]:
    return {
        f"{source.strip()}#p{page.strip()}"
        for source, page in _INLINE_PDF_RE.findall(report)
    }


def _merge_inline_pdf_references(report: str, references: List[Dict]) -> List[Dict]:
    """
    모델이 입력 분석의 정식 PDF citation을 본문에 유지했지만 state.references에
    같은 page가 없을 수 있다. 같은 source가 이미 수집된 PDF에 한해 REFERENCE에 보강한다.
    """
    merged = list(references)
    known_sources = {
        ref.get("source", "").strip()
        for ref in _pdf_references(references)
    }
    seen = {_pdf_key(ref) for ref in _pdf_references(merged)}

    for source, page in _INLINE_PDF_RE.findall(report):
        source = source.strip()
        page = page.strip()
        if source not in known_sources:
            continue
        inline_ref = {"type": "pdf", "source": source, "page": page}
        key = _pdf_key(inline_ref)
        if key in seen:
            continue
        seen.add(key)
        merged.append(inline_ref)

    return merged


def _format_inline_citation(ref: Dict) -> str:
    return f"[출처: {ref.get('source', 'Unknown')}, p.{ref.get('page', '?')}]"


def _append_missing_pdf_citations(report: str, references: List[Dict]) -> str:
    """
    최종 재작성 과정에서 PDF inline citation이 누락되면 REFERENCE 정합성이 무너진다.
    본문에 없는 PDF 출처는 별도 근거 매핑 섹션에 명시해 인용-참고문헌 연결을 보존한다.
    """
    pdf_refs = _pdf_references(references)
    if not pdf_refs:
        return report

    existing = _inline_pdf_keys(report)
    missing_refs = [ref for ref in pdf_refs if _pdf_key(ref) not in existing]
    if not missing_refs:
        return report

    lines = ["## 근거 출처 매핑"]
    for ref in missing_refs:
        source = ref.get("source", "Unknown")
        page = ref.get("page", "?")
        lines.append(f"- {source} p.{page}: {_format_inline_citation(ref)}")

    return f"{report.rstrip()}\n\n" + "\n".join(lines)


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


def _finalize_report_content(report_content: str, references: List[Dict]) -> str:
    """본문 정리 → 누락 PDF 인용 보강 → 결정적 REFERENCE 섹션 부착."""
    body = _strip_reference_section(report_content)
    body = _remove_noncanonical_citations(body)
    references = _merge_inline_pdf_references(body, references)
    body = _append_missing_pdf_citations(body, references)
    return f"{body}\n\n---\n\n**REFERENCE**\n\n{_format_references(references)}\n"


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
    report_content = _finalize_report_content(response.content, references)

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
