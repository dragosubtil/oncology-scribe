import pytest


def test_clinical_probe_excludes_uncertain_empty_and_bad_boundaries():
    from clinical_asr.public_data import select_clean_intervals

    entries = [
        (0, 4, "No fever was reported."),
        (4, 8, "<UNSURE>drug</UNSURE> was discussed."),
        (8, 12, ""),
        (12, 12, "A documented dose is present."),
        (12, 40, "This segment is too long."),
        (40, 45, "The patient reports persistent fatigue."),
    ]
    selected = select_clean_intervals(entries, limit=2, seed=42)
    assert {item[2] for item in selected} == {
        "No fever was reported.", "The patient reports persistent fatigue.",
    }


def test_clinical_probe_selection_is_repeatable_without_model_outputs():
    from clinical_asr.public_data import select_clean_intervals

    entries = [(i * 5, i * 5 + 4, f"This is source sentence number {i}.") for i in range(20)]
    assert select_clean_intervals(entries, 4, 42) == select_clean_intervals(entries, 4, 42)
    assert len(select_clean_intervals(entries, 4, 42)) == 4
    assert select_clean_intervals(entries, 4, 42) != select_clean_intervals(entries, 4, 43)


def test_clinical_probe_does_not_silently_deliver_fewer_samples():
    from clinical_asr.public_data import select_clean_intervals

    with pytest.raises(ValueError, match="Insufficient"):
        select_clean_intervals([(0, 4, "No fever was reported.")], 2, 42)


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_clinical_probe_acquisition_is_bounded(limit):
    from clinical_asr.public_data import select_clean_intervals

    with pytest.raises(ValueError, match="limit"):
        select_clean_intervals([], limit, 42)
def test_acquisition_cannot_overwrite_a_frozen_manifest(tmp_path):
    from clinical_asr.public_data import require_new_probe

    (tmp_path / "manifest.jsonl").write_text("frozen", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        require_new_probe(tmp_path)


def test_acquisition_allows_resuming_download_before_manifest_exists(tmp_path):
    from clinical_asr.public_data import require_new_probe

    (tmp_path / "audio").mkdir()
    require_new_probe(tmp_path)

