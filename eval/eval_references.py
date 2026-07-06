"""
보고서 REFERENCE 정합성 평가.

실행:
    python -m eval.eval_references
    python -m eval.eval_references --report outputs/report.md
"""

import argparse
import json
import os

from eval.reference_metrics import reference_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_REPORT = os.path.join(os.path.dirname(__file__), "..", "outputs", "report.md")
DEFAULT_OUT = os.path.join(RESULTS_DIR, "reference_metrics.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as f:
        report = f.read()

    metrics = reference_metrics(report)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 55)
    print("REFERENCE 정합성 평가")
    print("=" * 55)
    print(f"  inline PDF citations : {metrics['inline_pdf_citation_count']}")
    print(f"  PDF references       : {metrics['reference_pdf_count']}")
    print(f"  web references       : {metrics['web_reference_count']}")
    print(f"  inline coverage      : {metrics['inline_pdf_reference_coverage']:.3f}")
    print(f"  reference used rate  : {metrics['reference_pdf_used_rate']:.3f}")
    print(f"  status               : {'PASS' if metrics['passes_reference_check'] else 'FAIL'}")
    print(f"\n상세 → {args.out}")


if __name__ == "__main__":
    main()
