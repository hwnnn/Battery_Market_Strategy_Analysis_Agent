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
