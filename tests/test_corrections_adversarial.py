"""Offline adversarial checks for proposals and audio-reviewed medical drafts.

Safety expectations are ordinary assertions, including actionable audit failures.
The source hash checks consistency, not authenticity. Self-consistent report
rewrites are characterized below; trusted issuing state and the UI current-source
guard must establish provenance. No test contacts a real inference service.
"""

import copy
import hashlib
import io
import json
import urllib.request

import pytest

from clinical_asr import corrections

LANGUAGES = ("en", "de", "gsw", "fr", "it")


@pytest.fixture(autouse=True)
def offline_wire(monkeypatch):
    state = {"proposals": [], "requests": []}

    def blocked(*args, **kwargs):
        raise AssertionError("Real inference is forbidden in the adversarial battery")

    def transport(opener, request, **kwargs):
        state["requests"].append(request)
        envelope = {
            "model": "audit-stub",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({"proposals": state["proposals"]}),
            },
        }
        return io.BytesIO(json.dumps(envelope).encode("utf-8"))

    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", blocked)
    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", blocked)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", transport)
    return state


def edit(before, after, span_id="s0"):
    return {
        "span_id": span_id,
        "before": before,
        "after": after,
        "reason": "Spelling candidate; listen to the source audio.",
    }


def propose(wire, source, proposals, language="en"):
    wire["proposals"] = proposals
    return corrections.ollama_corrections(source, language, model="audit-stub")


@pytest.mark.parametrize("language, context", [
    ("en", "no fever; 2.5 mg/m\u00b2 IV; <5%; T2N0M0; 1/2 dose; twice daily"),
    ("de", "kein Fieber; 2,5 mg/m\u00b2; Stadium II; nicht erh\u00f6hen; zwei Dosen"),
    ("gsw", "n\u00f6d erh\u00f6he; kei Fieber; 2,5 mg; dr\u00fc Dosis; ohni Schmerz"),
    ("fr", "pas de fi\u00e8vre; 2,5 mg; stade IIA; sans douleur; deux doses"),
    ("it", "non aumentare; 2,5 mg; stadio III; senza dolore; due dosi"),
])
def test_reviewed_spelling_preserves_raw_source_and_multilingual_context(
    offline_wire, language, context
):
    source = " \tTrastuzumap " + context + ".\r\n"
    expected = " \tTrastuzumab " + context + ".\r\n"
    report = propose(offline_wire, source, [edit("Trastuzumap", "Trastuzumab")], language)
    original = copy.deepcopy(report)
    assert report["raw_transcript"] == source
    assert report["status"] == "proposed"
    assert "reviewed_transcript" not in report
    assert report["proposals"][0]["protected"] is False
    with pytest.raises(ValueError, match="audio"):
        corrections.apply_reviewed_corrections(report, ["e0"], False)
    assert corrections.apply_reviewed_corrections(report, [], False)["reviewed_transcript"] == source
    draft = corrections.apply_reviewed_corrections(report, ["e0"], True)
    assert draft["reviewed_transcript"] == expected
    assert draft["raw_transcript"] == source
    assert draft["source_transcript_sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert draft["status"] == "reviewed_draft"
    assert draft["clinical_validation"] == "not_established"
    assert draft["multilingual_validation"] == "not_established"
    assert draft["requires_human_review"] is True
    assert draft["requires_clinician_final_review"] is True
    assert draft["clinician_final_review"] == "pending"
    assert report == original
    draft["proposals"][0]["after"] = "carboplatin"
    assert report == original


@pytest.mark.parametrize("language, before, after", [
    ("en", "0.25 mg", "25 mg"),
    ("de", "0,25 mg", "25 mg"),
    ("gsw", "\u22120,5 mg", "0,5 mg"),
    ("fr", "1e-3 g", "1e3 g"),
    ("it", ".5 mg", "5 mg"),
    ("en", "5 mg/kg", "5 mg"),
    ("de", "50 mg/m\u00b2", "50 mg/m"),
    ("gsw", "500 \u00b5g", "500 mg"),
    ("fr", "5 \u03bcg", "5 g"),
    ("it", "5 mL", "5 L"),
    ("en", "5 mEq", "5 Eq"),
    ("de", "2 Gy", "2 cGy"),
    ("gsw", "5 mmol", "5 mol"),
    ("fr", "5 UI", "5 IU"),
    ("it", "5 mg/dL", "5 mg/L"),
    ("en", "<5%", ">5%"),
    ("de", "10 Prozent", "10"),
    ("gsw", "1/2", "1 2"),
    ("fr", "10 pour cent", "10"),
    ("it", "stadio IIA", "stadio IIB"),
    ("en", "two doses", "ten doses"),
    ("de", "zwei Dosen", "zehn Dosen"),
    ("gsw", "dr\u00fc Dosis", "eis Dosis"),
    ("fr", "deux doses", "dix doses"),
    ("it", "due dosi", "dieci dosi"),
    ("en", "no fever", "fever"),
    ("en", "isn't indicated", "is indicated"),
    ("en", "denies pain", "reports pain"),
    ("de", "kein Fieber", "Fieber"),
    ("de", "nicht erh\u00f6hen", "erh\u00f6hen"),
    ("de", "ohne Schmerzen", "mit Schmerzen"),
    ("gsw", "n\u00f6d erh\u00f6he", "erh\u00f6he"),
    ("gsw", "kei Fieber", "Fieber"),
    ("gsw", "ohni Schmerz", "mit Schmerz"),
    ("fr", "n'est pas indiqu\u00e9", "est indiqu\u00e9"),
    ("fr", "sans douleur", "avec douleur"),
    ("fr", "aucune fi\u00e8vre", "fi\u00e8vre"),
    ("it", "non indicato", "indicato"),
    ("it", "senza dolore", "con dolore"),
    ("it", "nessun dolore", "dolore"),
    ("en", "no pain; fever", "pain; no fever"),
    ("de", "kein Schmerz; Fieber", "Schmerz; kein Fieber"),
    ("gsw", "kei Schmerz; Fieber", "Schmerz; kei Fieber"),
    ("fr", "sans douleur; fi\u00e8vre", "douleur; sans fi\u00e8vre"),
    ("it", "senza dolore; febbre", "dolore; senza febbre"),
])
def test_critical_token_edits_remain_blocked_even_with_audio(
    offline_wire, language, before, after
):
    source = before + "."
    report = propose(offline_wire, source, [edit(before, after)], language)
    original = copy.deepcopy(report)
    assert report["proposals"][0]["protected"] is True
    with pytest.raises(ValueError, match="protected"):
        corrections.apply_reviewed_corrections(report, ["e0"], True)
    assert report == original
    assert corrections.apply_reviewed_corrections(report, [], False)["reviewed_transcript"] == source
    report["proposals"][0]["protected"] = False
    with pytest.raises(ValueError, match="protected"):
        corrections.apply_reviewed_corrections(report, [], False)


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("before", ["cis plat", "cis plat in", "plat in", "cis\tplat in"])
def test_cisplatin_spacing_applies_complete_source_only_after_audio(
    offline_wire, language, before
):
    name = "cis\tplat in" if "\t" in before else "cis plat in"
    source = name + " 2,5 mg/m\u00b2; no fever; nicht; n\u00f6d; sans; senza."
    expected = "cisplatin 2,5 mg/m\u00b2; no fever; nicht; n\u00f6d; sans; senza."
    report = propose(offline_wire, source, [edit(before, "cisplatin")], language)
    item = report["proposals"][0]
    assert item["before"] == name
    assert (item["start"], item["end"]) == (0, len(name))
    assert item["protected"] is False
    if before != name:
        assert item["model_before"] == before
    with pytest.raises(ValueError, match="audio"):
        corrections.apply_reviewed_corrections(report, ["e0"], False)
    draft = corrections.apply_reviewed_corrections(report, ["e0"], True)
    assert draft["reviewed_transcript"] == expected
    assert draft["raw_transcript"] == source


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("before, after", [
    ("cisplatin", "carboplatin"),
    ("CISPLATIN", "carboplatin"),
    ("cis plat in", "carboplatin"),
    ("carboplatin", "cisplatin"),
    ("oxaliplatin", "cisplatin"),
    ("paclitaxel", "docetaxel"),
    ("trastuzumab", "pertuzumab"),
    ("letrozole", "anastrozole"),
])
def test_glossary_drug_substitutions_are_removed_from_selectable_proposals(
    offline_wire, language, before, after
):
    source = before + " 2.5 mg; no fever."
    report = propose(offline_wire, source, [edit(before, after)], language)
    assert report["proposals"] == []
    assert report["rejected_proposals"][0]["rejection"] == "known_medication_substitution"
    assert report["raw_transcript"] == source
    assert corrections.apply_reviewed_corrections(report, [], False)["reviewed_transcript"] == source
    with pytest.raises(ValueError, match="unknown|ID"):
        corrections.apply_reviewed_corrections(report, ["e0"], True)


@pytest.mark.parametrize("language", LANGUAGES)
def test_valid_afatinib_cannot_be_substituted_with_imatinib(offline_wire, language):
    # Afatinib is a valid medication, not a misspelling of imatinib.
    source = "afatinib 5 mg; no fever."
    report = propose(offline_wire, source, [edit("afatinib", "imatinib")], language)
    assert report["raw_transcript"] == source
    if report["proposals"]:
        with pytest.raises(ValueError):
            corrections.apply_reviewed_corrections(report, ["e0"], True)
    else:
        assert corrections.apply_reviewed_corrections(report, [], False)["reviewed_transcript"] == source


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("before", ["Cisplatin", "CISPLATIN"])
def test_known_valid_cisplatin_case_only_change_is_not_selectable(offline_wire, language, before):
    source = before + " 5 mg; no fever."
    report = propose(offline_wire, source, [edit(before, "cisplatin")], language)
    assert report["proposals"] == []
    draft = corrections.apply_reviewed_corrections(report, [], False)
    assert draft["raw_transcript"] == source
    assert draft["reviewed_transcript"] == source
    assert draft["clinical_validation"] == "not_established"
    assert draft["requires_human_review"] is True


@pytest.mark.parametrize("source, proposals", [
    ("Trastuzumap and Trastuzumap.", [edit("Trastuzumap", "Trastuzumab")]),
    ("cis plat in and cis plat in.", [edit("cis plat", "cisplatin")]),
    ("cis plat in and cisplatin.", [edit("cis", "cisplatin")]),
    ("Trastuzumap. Other source.", [edit("Trastuzumap", "Trastuzumab", "s1")]),
    ("Trastuzumap.", [edit("Trastuzumap", "Trastuzumab", "s99")]),
    ("Trastuzumap. Other source.", [edit("Trastuzumap. Other", "Trastuzumab")]),
    ("cis plat in.", [edit("cis plat", "cisplatin"), edit("plat in", "cisplatin")]),
])
def test_ambiguous_missing_or_overlapping_source_is_rejected(offline_wire, source, proposals):
    with pytest.raises(ValueError, match="ambiguous|source|span|overlap"):
        propose(offline_wire, source, proposals)


@pytest.mark.parametrize("before, after", [
    ("cisplatin", "cisplatin"),
    ("cisplatin", " cisplatin "),
    ("cis plat in", "cis plat in"),
    ("Trastuzumab", "Trastuzumab"),
])
def test_unchanged_proposals_preserve_source_without_clinical_approval(offline_wire, before, after):
    source = " \t" + before + " 5 mg.\r\n"
    report = propose(offline_wire, source, [edit(before, after)])
    assert report["proposals"] == []
    assert report["rejected_proposals"][0]["rejection"] == "unchanged_text"
    draft = corrections.apply_reviewed_corrections(report, [], False)
    assert draft["reviewed_transcript"] == source
    assert draft["raw_transcript"] == source
    assert draft["clinical_validation"] == "not_established"
    assert draft["clinician_final_review"] == "pending"


@pytest.mark.parametrize("source", ["No such name.", "cisplatin and cisplatin."])
def test_noop_does_not_bypass_unique_source_requirement(offline_wire, source):
    with pytest.raises(ValueError, match="unique|source"):
        propose(offline_wire, source, [edit("cisplatin", "cisplatin")])


@pytest.mark.parametrize("language", LANGUAGES)
def test_empty_model_response_is_no_suggestion_not_validation(offline_wire, language):
    source = "cisplatin 5 mg; no fever."
    report = propose(offline_wire, source, [], language)
    draft = corrections.apply_reviewed_corrections(report, [], False)
    assert draft["reviewed_transcript"] == source
    assert draft["clinical_validation"] == "not_established"
    assert draft["multilingual_validation"] == "not_established"
    assert draft["requires_human_review"] is True
    assert draft["requires_clinician_final_review"] is True


@pytest.mark.parametrize("selected_ids", [[], ["e0"]], ids=["unselected", "selected"])
@pytest.mark.parametrize("mutation", [
    {"raw_transcript": "Trastuzumap 50 mg."},
    {"source_transcript_sha256": "0" * 64},
    {"clinical_validation": "approved"},
    {"multilingual_validation": "approved"},
    {"requires_human_review": False},
    {"requires_clinician_final_review": False},
    {"clinician_final_review": "approved"},
    {"status": "reviewed_draft"},
])
def test_stale_source_and_forged_review_flags_are_rejected(
    offline_wire, selected_ids, mutation
):
    report = propose(offline_wire, "Trastuzumap 5 mg.", [edit("Trastuzumap", "Trastuzumab")])
    report.update(mutation)
    with pytest.raises(ValueError):
        corrections.apply_reviewed_corrections(report, selected_ids, bool(selected_ids))


@pytest.mark.parametrize("selected_ids", [[], ["e0"]], ids=["unselected", "selected"])
@pytest.mark.parametrize("mutation", [
    {"before": "cis plat", "end": 8},
    {"model_before": "carboplatin"},
    {"model_before": "cisplatin"},
    {"model_before": "not in the source"},
    {"start": 1},
    {"end": 10},
    {"requires_audio_confirmation": False},
])
def test_inconsistent_expanded_source_or_model_before_is_rejected(
    offline_wire, selected_ids, mutation
):
    report = propose(offline_wire, "cis plat in 5 mg.", [edit("cis plat", "cisplatin")])
    report["proposals"][0].update(mutation)
    with pytest.raises(ValueError):
        corrections.apply_reviewed_corrections(report, selected_ids, bool(selected_ids))


@pytest.mark.parametrize("selected_ids", [[], ["e0"]], ids=["unselected", "selected"])
@pytest.mark.parametrize(
    "forged_before", [None, "cis", "plat in"], ids=["removed", "cis", "plat-in"]
)
def test_consistent_model_before_rewrite_is_accepted_without_external_provenance(
    offline_wire, selected_ids, forged_before
):
    """Self-consistency cannot distinguish a rewritten trace from the issued trace."""
    report = propose(offline_wire, "cis plat in 5 mg.", [edit("cis plat", "cisplatin")])
    assert report["proposals"][0]["model_before"] == "cis plat"
    issued_report = copy.deepcopy(report)
    report["proposals"][0]["model_before"] = forged_before
    draft = corrections.apply_reviewed_corrections(report, selected_ids, bool(selected_ids))
    assert draft["proposals"][0]["model_before"] == forged_before
    assert draft["raw_transcript"] == "cis plat in 5 mg."
    assert draft["reviewed_transcript"] == (
        "cisplatin 5 mg." if selected_ids else "cis plat in 5 mg."
    )
    assert draft["accepted_ids"] == selected_ids
    assert draft["clinical_validation"] == "not_established"
    assert issued_report["proposals"][0]["model_before"] == "cis plat"


@pytest.mark.parametrize("selected_ids", [[], ["e0"]], ids=["unselected", "selected"])
def test_consistent_retargeted_report_needs_trusted_external_proposal_state(offline_wire, selected_ids):
    """Selection IDs refer to the supplied report, not an authenticated issued copy."""
    source = "Trastuzumap and Rituximap 5 mg."
    report = propose(offline_wire, source, [edit("Trastuzumap", "Trastuzumab")])
    issued_report = copy.deepcopy(report)
    report["proposals"][0].update(before="Rituximap", after="Rituximab", start=16, end=25)
    draft = corrections.apply_reviewed_corrections(report, selected_ids, bool(selected_ids))
    assert draft["raw_transcript"] == source
    assert draft["reviewed_transcript"] == (
        "Trastuzumap and Rituximab 5 mg." if selected_ids else source
    )
    assert draft["accepted_ids"] == selected_ids
    assert draft["clinical_validation"] == "not_established"
    assert issued_report["proposals"][0]["before"] == "Trastuzumap"


@pytest.mark.parametrize("selected_ids", [[], ["e0"]], ids=["unselected", "selected"])
def test_source_checksum_accepts_rewritten_text_with_recomputed_hash(offline_wire, selected_ids):
    """The UI must compare current source with trusted state; SHA is not a signature."""
    report = propose(offline_wire, "Trastuzumap 5 mg.", [edit("Trastuzumap", "Trastuzumab")])
    issued_report = copy.deepcopy(report)
    report["raw_transcript"] = "Trastuzumap 50 mg."
    report["source_transcript_sha256"] = hashlib.sha256(
        report["raw_transcript"].encode("utf-8")
    ).hexdigest()
    draft = corrections.apply_reviewed_corrections(report, selected_ids, bool(selected_ids))
    assert draft["raw_transcript"] == "Trastuzumap 50 mg."
    assert draft["reviewed_transcript"] == (
        "Trastuzumab 50 mg." if selected_ids else "Trastuzumap 50 mg."
    )
    assert draft["source_transcript_sha256"] == report["source_transcript_sha256"]
    assert draft["source_transcript_sha256"] != issued_report["source_transcript_sha256"]
    assert draft["clinical_validation"] == "not_established"
    assert issued_report["raw_transcript"] == "Trastuzumap 5 mg."


def test_reviewed_draft_cannot_be_reused_as_a_proposed_report(offline_wire):
    report = propose(offline_wire, "Trastuzumap 5 mg.", [edit("Trastuzumap", "Trastuzumab")])
    draft = corrections.apply_reviewed_corrections(report, ["e0"], True)
    with pytest.raises(ValueError, match="report"):
        corrections.apply_reviewed_corrections(draft, ["e0"], True)
