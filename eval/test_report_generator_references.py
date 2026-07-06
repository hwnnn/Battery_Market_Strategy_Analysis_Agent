from agents.report_generator import _finalize_report_content
from eval.reference_metrics import reference_metrics


def test_finalize_report_replaces_model_reference_and_maps_missing_pdf_refs():
    draft = """
# 보고서

본문에 아직 PDF 인용이 없음.

**REFERENCE**

- hallucinated.example(날짜 미상). 모델이 만든 임의 출처. https://hallucinated.example
"""
    refs = [
        {"type": "pdf", "source": "01_market_background.pdf", "page": 3},
        {"type": "pdf", "source": "02_LGES_strategy.pdf", "page": 5},
        {"type": "pdf", "source": "01_market_background.pdf", "page": 3},
        {
            "type": "web",
            "source": "example.com",
            "date": "2026-01-01",
            "title": "Example title",
            "url": "https://example.com/a",
            "perspective": "중립",
        },
    ]

    report = _finalize_report_content(draft, refs)
    metrics = reference_metrics(report)

    assert "hallucinated.example" not in report
    assert report.count("**REFERENCE**") == 1
    assert "## 근거 출처 매핑" in report
    assert metrics["inline_pdf_citation_count"] == 2
    assert metrics["reference_pdf_count"] == 2
    assert metrics["web_reference_count"] == 1
    assert metrics["inline_pdf_reference_coverage"] == 1.0
    assert metrics["reference_pdf_used_rate"] == 1.0
    assert metrics["passes_reference_check"] is True


def test_finalize_report_removes_noncanonical_citations():
    draft = """
# 보고서

숫자형 주석 [1][2]과 불완전한 출처 [출처: 1] [출처: 03_CATL_strategy.pdf]는 제거되어야 한다.
정식 인용은 유지한다 [출처: 01_market_background.pdf, p.3].
"""
    refs = [
        {"type": "pdf", "source": "01_market_background.pdf", "page": 3},
    ]

    report = _finalize_report_content(draft, refs)
    metrics = reference_metrics(report)

    assert "[1]" not in report
    assert "[2]" not in report
    assert "[출처: 1]" not in report
    assert "[출처: 03_CATL_strategy.pdf]" not in report
    assert "[출처: 01_market_background.pdf, p.3]" in report
    assert metrics["malformed_citation_count"] == 0
    assert metrics["passes_reference_check"] is True


def test_finalize_report_adds_known_inline_refs_to_reference_section():
    draft = """
# 보고서

본문에서 모델이 유지한 정식 인용 [출처: 03_CATL_strategy.pdf, p.12].
"""
    refs = [
        {"type": "pdf", "source": "03_CATL_strategy.pdf", "page": 13},
    ]

    report = _finalize_report_content(draft, refs)
    metrics = reference_metrics(report)

    assert "  - 03_CATL_strategy.pdf (p.12)" in report
    assert "  - 03_CATL_strategy.pdf (p.13)" in report
    assert metrics["inline_pdf_reference_coverage"] == 1.0
    assert metrics["reference_pdf_used_rate"] == 1.0
    assert metrics["passes_reference_check"] is True
