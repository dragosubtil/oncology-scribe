import runpy
from pathlib import Path

import pytest


def helper():
    return runpy.run_path(str(Path(__file__).parents[1] / "tools/summarize_public_probe.py"))["error_spans"]


def test_error_analysis_preserves_numeric_and_negation_differences():
    spans = helper()("No fever. Dose 2.5 mg.", "Fever. Dose 25 mg.")
    assert any("no" in item["reference"] for item in spans)
    assert any("2.5" in item["reference"] and "25" in item["hypothesis"] for item in spans)


def test_equal_normalized_text_has_no_errors():
    assert helper()("No fever.", "NO FEVER") == []


@pytest.mark.parametrize("reference,hypothesis,expected", [
    ("No fever", "", "delete"), ("", "fever", "insert"),
])
def test_error_analysis_supports_empty_hypothesis_or_reference(reference, hypothesis, expected):
    assert helper()(reference, hypothesis)[0]["type"] == expected
