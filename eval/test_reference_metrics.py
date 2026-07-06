from eval.reference_metrics import (
    inline_pdf_refs,
    reference_metrics,
    reference_pdf_refs,
    web_reference_count,
)


def test_reference_metrics_detects_missing_and_orphan_refs():
    report = """
본문 [출처: 01_market_background.pdf, p.3]
본문 [출처: 02_LGES_strategy.pdf, p.5]

**REFERENCE**

### 기관 보고서 / PDF 문서
  - 01_market_background.pdf (p.3)
  - 03_CATL_strategy.pdf (p.9)

### 웹페이지
  - example.com(2026-01-01). Example title. https://example.com/a
"""

    assert inline_pdf_refs(report) == [
        "01_market_background.pdf#p3",
        "02_LGES_strategy.pdf#p5",
    ]
    assert reference_pdf_refs(report) == [
        "01_market_background.pdf#p3",
        "03_CATL_strategy.pdf#p9",
    ]
    assert web_reference_count(report) == 1

    metrics = reference_metrics(report)

    assert metrics["inline_pdf_reference_coverage"] == 0.5
    assert metrics["reference_pdf_used_rate"] == 0.5
    assert metrics["missing_pdf_references"] == ["02_LGES_strategy.pdf#p5"]
    assert metrics["orphan_pdf_references"] == ["03_CATL_strategy.pdf#p9"]
