"""
REFERENCE 품질을 정적으로 점검하는 지표 함수.
LLM/API 없이 보고서 본문 인용과 REFERENCE 섹션의 정합성을 확인한다.
"""

import re

INLINE_PDF_RE = re.compile(r"\[출처:\s*([^,\]]+),\s*p\.([0-9?]+)\]")
REFERENCE_PDF_RE = re.compile(r"^\s*-\s+(.+?)\s+\(p\.([0-9?]+)\)\s*$", re.MULTILINE)
WEB_REFERENCE_RE = re.compile(r"^\s*-\s+.+?\(.+?\)\.\s+.+?\.\s+https?://\S+", re.MULTILINE)
REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:\*\*REFERENCE\*\*|#{1,6}\s*REFERENCE|REFERENCE)\s*$"
)


def _norm_pdf_ref(source: str, page: str) -> str:
    return f"{source.strip()}#p{page.strip()}"


def reference_section(report: str) -> str:
    """보고서에서 REFERENCE 섹션만 반환한다. 없으면 빈 문자열."""
    matches = list(REFERENCE_HEADING_RE.finditer(report))
    return report[matches[-1].start():] if matches else ""


def body_section(report: str) -> str:
    """REFERENCE 앞 본문만 반환한다. inline citation은 본문에서만 계산한다."""
    section = reference_section(report)
    if not section:
        return report
    return report[: report.rfind(section)]


def inline_pdf_refs(report: str) -> list[str]:
    return sorted({
        _norm_pdf_ref(source, page)
        for source, page in INLINE_PDF_RE.findall(body_section(report))
    })


def reference_pdf_refs(report: str) -> list[str]:
    section = reference_section(report)
    return sorted({
        _norm_pdf_ref(source, page)
        for source, page in REFERENCE_PDF_RE.findall(section)
    })


def web_reference_count(report: str) -> int:
    return len(WEB_REFERENCE_RE.findall(reference_section(report)))


def reference_metrics(report: str) -> dict:
    inline = set(inline_pdf_refs(report))
    listed = set(reference_pdf_refs(report))
    missing = sorted(inline - listed)
    orphan = sorted(listed - inline)

    inline_coverage = (
        round((len(inline) - len(missing)) / len(inline), 4)
        if inline else (0.0 if listed else 1.0)
    )
    used_rate = (
        round((len(listed) - len(orphan)) / len(listed), 4)
        if listed else (0.0 if inline else 1.0)
    )
    passes = (bool(inline) and not missing and not orphan) if listed else not inline

    return {
        "inline_pdf_citation_count": len(inline),
        "reference_pdf_count": len(listed),
        "web_reference_count": web_reference_count(report),
        "has_inline_pdf_citations": bool(inline),
        "inline_pdf_reference_coverage": inline_coverage,
        "reference_pdf_used_rate": used_rate,
        "passes_reference_check": passes,
        "missing_pdf_references": missing,
        "orphan_pdf_references": orphan,
    }
