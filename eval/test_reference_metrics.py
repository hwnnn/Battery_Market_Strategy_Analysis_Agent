from eval.reference_metrics import (
    inline_pdf_refs,
    malformed_citations,
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

    assert metrics["malformed_citation_count"] == 0
    assert metrics["passes_reference_check"] is False
    assert metrics["inline_pdf_reference_coverage"] == 0.5
    assert metrics["reference_pdf_used_rate"] == 0.5
    assert metrics["missing_pdf_references"] == ["02_LGES_strategy.pdf#p5"]
    assert metrics["orphan_pdf_references"] == ["03_CATL_strategy.pdf#p9"]


def test_reference_metrics_treats_zero_inline_citations_as_failure():
    report = """
본문에 출처 표기가 없음

**REFERENCE**

### 기관 보고서 / PDF 문서
  - 01_market_background.pdf (p.3)
"""

    metrics = reference_metrics(report)

    assert metrics["has_inline_pdf_citations"] is False
    assert metrics["passes_reference_check"] is False
    assert metrics["inline_pdf_reference_coverage"] == 0.0
    assert metrics["reference_pdf_used_rate"] == 0.0
    assert metrics["orphan_pdf_references"] == ["01_market_background.pdf#p3"]


def test_reference_metrics_rejects_malformed_citations():
    report = """
본문 [1][2]
본문 [출처: 1]
본문 [출처: 03_CATL_strategy.pdf]
정식 인용 [출처: 01_market_background.pdf, p.3]

**REFERENCE**

### 기관 보고서 / PDF 문서
  - 01_market_background.pdf (p.3)
"""

    metrics = reference_metrics(report)

    assert malformed_citations(report) == [
        "[1]",
        "[2]",
        "[출처: 1]",
        "[출처: 03_CATL_strategy.pdf]",
    ]
    assert metrics["malformed_citation_count"] == 4
    assert metrics["has_malformed_citations"] is True
    assert metrics["inline_pdf_reference_coverage"] == 1.0
    assert metrics["reference_pdf_used_rate"] == 1.0
    assert metrics["passes_reference_check"] is False
