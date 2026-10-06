import pytest

from clinical_asr.benchmark import evaluate_predictions, paired_comparison
from clinical_asr.metrics import normalize_text, score_text
from clinical_asr.schema import Sample


def sample(sample_id="s1", reference="No fever", **changes):
    value = dict(sample_id=sample_id, speaker_id=sample_id, encounter_id=sample_id,
                 audio_path="absent.wav", language="en", target_policy="verbatim",
                 split="test", reference=reference, data_use="simulated",
                 source="fixture", license="project-generated")
    return Sample(**(value | changes))


def test_word_edit_counts_and_reference_denominator():
    result = score_text("no fever today", "fever today please", [], "en")
    assert result["normalized"]["deletions"] == 1
    assert result["normalized"]["insertions"] == 1
    assert result["normalized"]["wer"] == pytest.approx(2 / 3)
    assert "negation_change" in result["review_flags"]


def test_decimals_and_accents_are_not_erased():
    assert normalize_text("Dose: 2.5 mg, pas de fièvre.") == "dose 2.5 mg pas de fièvre"
    assert score_text("2.5 mg", "25 mg", [], "en")["normalized"]["wer"] == 0.5
    assert score_text("2,5 mg", "2.5 mg", [], "de")["normalized"]["wer"] == 0.5


def test_negative_numbers_and_percentages_survive():
    assert normalize_text("-2.5 +3 5%") == "-2.5 +3 5%"


def test_unicode_minus_is_not_erased_or_confused_with_a_positive_value():
    negative = "Value \u22122.5"
    assert normalize_text(negative) == "value -2.5"
    assert score_text(negative, "Value 2.5", [], "en")["normalized"]["wer"] == 0.5
    assert "numeric_or_unit_change" in score_text(negative, "Value 2.5", [], "en")["review_flags"]


@pytest.mark.parametrize("number", [".5", ",5", "-.5", "+,5"])
def test_leading_decimal_separators_remain_clinically_distinct(number):
    assert normalize_text(f"Dose {number} mg") == f"dose {number} mg"
    result = score_text(f"Dose {number} mg", "Dose 5 mg", [], "en")
    assert result["normalized"]["wer"] == 1 / 3
    assert "numeric_or_unit_change" in result["review_flags"]


def test_raw_and_normalized_scores_are_distinct():
    result = score_text("No fever.", "no fever", [], "en")
    assert result["raw"]["wer"] == 1.0
    assert result["normalized"]["wer"] == 0.0


def test_empty_reference_reports_insertions_not_zero_wer():
    result = score_text("", "thank you very much", [], "en")
    assert result["normalized"]["wer"] is None
    assert result["normalized"]["insertions"] == 4
    assert result["empty_reference"]


def test_term_matching_uses_boundaries_and_occurrence_counts():
    result = score_text("pain pain carcinoma", "pain painless", ["pain", "carcinoma"], "en")
    assert result["terms"]["reference_occurrences"] == 3
    assert result["terms"]["matched_occurrences"] == 1
    assert result["terms"]["recall"] == pytest.approx(1 / 3)
    assert result["terms"]["precision"] == 1.0


def test_term_not_present_in_reference_is_an_insertion():
    result = score_text("no pain", "no pain carcinoma", ["pain", "carcinoma"], "en")
    assert result["terms"]["precision"] == 0.5
    assert result["terms"]["recall"] == 1.0


def test_no_annotated_terms_has_no_claimed_perfect_f1():
    assert score_text("hello", "hello", [], "en")["terms"]["f1"] is None


def test_micro_and_macro_wer_have_different_denominators():
    rows = [sample("s1", "one"), sample("s2", "one two three four")]
    result = evaluate_predictions(rows, {"s1": "wrong", "s2": "one two three four"}, bootstrap=0)
    group = result["groups"][0]
    assert group["micro_wer"] == 0.2
    assert group["macro_wer"] == 0.5


def test_missing_and_unknown_predictions_fail():
    with pytest.raises(ValueError, match="Missing"):
        evaluate_predictions([sample()], {}, bootstrap=0)
    with pytest.raises(ValueError, match="Unknown"):
        evaluate_predictions([sample()], {"s1": "No fever", "alien": "x"}, bootstrap=0)


def test_swiss_targets_are_never_pooled():
    rows = [sample("s1", "Kein Fieber", language="gsw", target_policy="standard_de"),
            sample("s2", "Kei Fieber", language="gsw", target_policy="dialect")]
    result = evaluate_predictions(rows, {"s1": "Kein Fieber", "s2": "Kei Fieber"}, bootstrap=0)
    assert {g["target_policy"] for g in result["groups"]} == {"dialect", "standard_de"}
    assert len(result["groups"]) == 2


def test_one_independent_speaker_has_no_confidence_interval():
    rows = [sample("s1"), sample("s2", speaker_id="s1")]
    group = evaluate_predictions(rows, {"s1": "No fever", "s2": "No fever"}, bootstrap=100)["groups"][0]
    assert group["independent_groups"] == 1
    assert group["micro_wer_ci95"] is None


def test_bootstrap_seed_is_reproducible():
    rows = [sample("s1"), sample("s2"), sample("s3")]
    predictions = {"s1": "No fever", "s2": "fever", "s3": "fever"}
    a = evaluate_predictions(rows, predictions, bootstrap=100, seed=7)
    b = evaluate_predictions(rows, predictions, bootstrap=100, seed=7)
    assert a["groups"][0]["micro_wer_ci95"] == b["groups"][0]["micro_wer_ci95"]


def test_joint_encounter_dependencies_are_grouped():
    rows = [sample("s1", speaker_id="a", encounter_id="visit"),
            sample("s2", speaker_id="b", encounter_id="visit"),
            sample("s3", speaker_id="b", encounter_id="other")]
    result = evaluate_predictions(rows, {row.sample_id: "No fever" for row in rows}, bootstrap=0)
    assert result["groups"][0]["independent_groups"] == 1


def test_paired_comparison_requires_same_samples():
    with pytest.raises(ValueError, match="same"):
        paired_comparison([sample()], {"s1": "No fever"}, {}, bootstrap=100)


def test_paired_delta_direction_is_candidate_minus_baseline():
    rows = [sample("s1"), sample("s2")]
    result = paired_comparison(rows, {"s1": "fever", "s2": "fever"},
                               {"s1": "No fever", "s2": "No fever"}, bootstrap=100)
    assert result["groups"][0]["delta_micro_wer"] == -0.5


def test_paired_comparison_rejects_negative_bootstrap_iterations():
    with pytest.raises(ValueError, match="negative"):
        paired_comparison([sample()], {"s1": "No fever"}, {"s1": "No fever"}, bootstrap=-1)
