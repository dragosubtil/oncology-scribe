"""Local research workbench for raw ASR, review cues and extractive note drafts."""

from __future__ import annotations

import hashlib
import html
import json
import threading
import uuid
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from .cli import write_json
from .highlights import render_highlight_html, review_spans
from .metrics import score_text
from .notes import extractive_note, ollama_note, ollama_positional_note, validate_local_model
from .schema import file_sha256
from .timing import TimingHistory, begin, finish, processing, render
from .transcribe import ASRConfig, Transcriber, _validate_policy
from .word_audio import (
    continuation_excerpt,
    invalidate_playback,
    require_current_excerpt,
    reserve_playback_action,
    word_audio_excerpt,
)

LANGUAGES = [
    ("English", "en"),
    ("German", "de"),
    ("Swiss German", "gsw"),
    ("French", "fr"),
    ("Italian", "it"),
]
ASSETS = Path(__file__).parent / "assets"
_AUDIO_UNSPECIFIED = object()
_WORKFLOW_LOCK = threading.RLock()
_ASR_REVISIONS = {
    'turbo': '0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf',
    'large-v3': 'edaa852ec7e145841d8ffdb056a99866b5f0a478',
    'small': '536b0662742c02347bc0e980a01041f333bce120',
}


def _icon(name):
    path = ASSETS / "icons" / f"{name}.svg"
    return str(path) if path.exists() else None


def correction_preview(report):
    proposals = report.get("proposals", [])
    if not proposals:
        if any(p.get("rejection") == "known_medication_substitution"
               for p in report.get("rejected_proposals", [])):
            return '<div class="safety-band">Unsafe medication substitutions blocked. No corrections to apply.</div>'
        return '<div class="empty-result">No corrections proposed. This is not a correctness check.</div>'
    rows = []
    for item in proposals:
        tag = "Manual review - not applied" if item["protected"] else "Audio confirmation required"
        rows.append(
            '<article class="edit-row">'
            f'<div class="edit-heading"><strong>{html.escape(item["id"])}</strong>'
            f'<span class="risk-label">{tag}</span></div>'
            f'<div class="edit-pair"><del>{html.escape(item["before"])}</del>'
            f"<ins>{html.escape(item['after'])}</ins></div>"
            f"<p>{html.escape(item['reason'])}</p></article>"
        )
    return "".join(rows)


def _workflow_snapshot(value, slot):
    if not isinstance(value, dict) or "generation" not in value:
        return value
    record = value.get(slot)
    if not record:
        return None
    if slot == "published":
        return require_published_result(record["value"], value)
    return deepcopy(record["report"] if slot == "issued" else record)


def propose_corrections_ui(text, language, model):
    from .corrections import ollama_corrections

    validate_local_model(model, "http://127.0.0.1:11434", 45)
    report = ollama_corrections(text, language, model)
    choices = [
        (f"{p['id']}: {p['before']} -> {p['after']}", p["id"])
        for p in report["proposals"]
        if not p["protected"]
    ]
    return correction_preview(report), report, report, choices


def review_corrections_ui(source, report, selected, confirmed, tracker=None, audio=None,
                          configuration=None):
    report = _workflow_snapshot(report, "issued")
    if tracker is not None:
        report = require_issued_proposals(source, report, tracker, audio, configuration)
    if not report or source != report.get("raw_transcript"):
        raise ValueError("The transcript changed. Request new proposals before reviewing.")
    if not selected:
        raise ValueError("No correction selected. No changes applied. Select an eligible proposal; "
                         "blocked proposals require manual editing.")
    from .corrections import apply_reviewed_corrections

    reviewed = apply_reviewed_corrections(report, selected or [], confirmed)
    if tracker is not None:
        issued = tracker["issued"]
        reviewed["workflow"] = {key: issued["request"][key] for key in
                                ("request_id", "session_id", "generation", "audio_sha256")}
        reviewed["workflow"].update(kind="corrections", event_order=issued["request"].get("event_order"))
        reviewed["workflow"]["proposal_id"] = issued["proposal_id"]
        tracker["reviewed"] = deepcopy(reviewed)
    return reviewed["reviewed_transcript"], reviewed, reviewed, reviewed["reviewed_transcript"]


def note_source_for_review(source, report, selected, confirmed, reviewed=None,
                           tracker=None, audio=None, configuration=None):
    report = _workflow_snapshot(report, "issued")
    reviewed = _workflow_snapshot(reviewed, "reviewed")
    if selected:
        if not report or source != report.get("raw_transcript"):
            raise ValueError("The transcript changed. Request new proposals before reviewing.")
        if not reviewed:
            raise ValueError("Create a reviewed draft before transferring selected corrections.")
        if tracker is not None and reviewed != tracker.get("reviewed"):
            raise ValueError("The reviewed draft was not issued for this request.")
        if (reviewed.get("raw_transcript") != source or confirmed is not True
                or reviewed.get("audio_confirmed") is not True
                or set(reviewed.get("accepted_ids", [])) != set(selected)):
            raise ValueError("Review decisions changed. Create a new reviewed draft.")
        return review_corrections_ui(source, report, selected, confirmed, tracker, audio, configuration)[0]
    if not source or not source.strip():
        raise ValueError("A transcript is required before creating a note draft.")
    return source


@lru_cache(maxsize=1)
def _engine(model: str, device: str, experimental: bool):
    return Transcriber(
        ASRConfig(
            model=model,
            device=device,
            compute_type="int8" if device == "cpu" else "int8_float16",
            allow_experimental_dialect=experimental,
            local_files_only=True,
            revision=_ASR_REVISIONS.get(model),
        )
    )


def transcribe_ui(audio, language, policy, model, device, threshold, experimental=False):
    if not audio:
        raise ValueError("An audio file is required")
    source_hash = file_sha256(Path(audio))
    result = _engine(model, device, experimental).transcribe_file(audio, language, policy)
    if source_hash != file_sha256(Path(audio)):
        raise ValueError("The audio changed while processing. Transcribe it again.")
    result["source_audio_sha256"] = source_hash
    result["review_revision"] = uuid.uuid4().hex
    result["review_threshold"] = threshold
    result["review_spans"] = review_spans(result["text"], result["words"], language, threshold)
    marked = render_highlight_html(result["text"], result["review_spans"], source_hash,
                                    result["review_revision"], result['words'])
    return result["text"], marked, result, result["text"], result


def note_preview(text: str, method: str, model: str):
    if method == "extractive":
        note = extractive_note(text)
    elif method == "positional":
        note = ollama_positional_note(text, model)
    elif method == "ollama":
        note = ollama_note(text, model)
    else:
        raise ValueError("Unknown note method")
    parts = []
    if note.get("classification_warnings"):
        parts.append('<div class="safety-band">Attribution review pending.</div>')
    for category, items in note["sections"].items():
        if items:
            parts.append(f"<h3>{html.escape(category.title())}</h3><ul>")
            parts.extend(f"<li>{html.escape(item['quote'])}</li>" for item in items)
            parts.append("</ul>")
    if note["uncovered_source"]:
        parts.append("<h3>Uncovered source</h3><ul>")
        parts.extend(f"<li>{html.escape(item['quote'])}</li>" for item in note["uncovered_source"])
        parts.append("</ul>")
    return "".join(parts), note


def compare_texts(reference: str, hypothesis: str, language: str, policy: str):
    _validate_policy(language, policy, experimental=True)
    return {
        "language": language,
        "target_policy": policy,
        "scores": score_text(reference, hypothesis, [], language),
    }


def _export(value):
    if value is None or value.get("status") == "failed":
        raise ValueError("No successful result is available to export")
    path = Path(".build/demo-exports") / f"draft-{uuid.uuid4().hex}.json"
    write_json(path, value)
    return str(path.resolve())


def export_matching_source(value, current_source):
    if not value or current_source != value.get("raw_transcript"):
        raise ValueError("The transcript changed. Generate a new result before exporting.")
    return _export(value)


def export_reviewed_source(value, source, confirmed, selected=None):
    if not value or value.get("audio_confirmed") is not confirmed:
        raise ValueError(
            "Audio confirmation changed. Create a new reviewed draft before exporting."
        )
    if selected is not None and set(value.get("accepted_ids", [])) != set(selected):
        raise ValueError("The correction selection changed. Create a new reviewed draft before exporting.")
    return export_matching_source(value, source)


def _require_matching_audio(value, audio):
    if (not value or not audio or not Path(audio).is_file()
            or file_sha256(Path(audio)) != value.get("source_audio_sha256")):
        raise ValueError("The audio changed or was cleared. Transcribe the current audio again.")


def export_matching_audio(value, audio):
    _require_matching_audio(value, audio)
    return _export(value)


def current_request_result(completed, current_inputs, tracker=None, current_audio=_AUDIO_UNSPECIFIED):
    """Do not publish a slow job's output under a different source or configuration."""
    if not completed or completed.get("inputs") != list(current_inputs):
        raise ValueError("Inputs changed while processing. Generate a new result.")
    if tracker is not None and not _request_is_current(completed.get("request"), current_inputs, tracker):
        raise ValueError("Request changed or was superseded. Generate a new result.")
    if (current_audio is not _AUDIO_UNSPECIFIED
            and current_audio != completed.get("request", {}).get("audio_path")):
        raise ValueError("The audio changed before publication. Generate a new result.")
    if tracker is None and completed.get("request", {}).get("session_id"):
        raise ValueError("The request requires its issuing session before publication.")
    if "error" in completed:
        raise ValueError(completed["error"])
    return deepcopy(completed["result"])


def _run_request(function, inputs):
    snapshot = deepcopy(list(inputs))
    try:
        return {"inputs": snapshot, "result": deepcopy(function(*deepcopy(snapshot)))}
    except Exception as exc:
        return {"inputs": snapshot, "error": str(exc)}


def _inputs_digest(inputs):
    return hashlib.sha256(json.dumps(list(inputs), ensure_ascii=True).encode()).hexdigest()


def invalidate_task(tracker):
    with _WORKFLOW_LOCK:
        tracker["generation"] += 1
        tracker.update(active_request_id=None, active_request=None, claimed_request_id=None,
                       issued=None, reviewed=None, published=None, busy=False, pending_start=None,
                       pending_inputs_digest=None, pending_audio_path=None)
        return render(None, generation=tracker["generation"])


def _prepare_timed_request(kind, inputs, history, audio, event_order):
    labels = {"asr": "Transcription", "corrections": "Terminology review", "note": "Note draft"}
    if kind == "asr":
        import soundfile as sf

        try:
            units = sf.info(inputs[0]).duration if inputs[0] else None
        except (OSError, RuntimeError, TypeError):
            units = None
        configuration = [inputs[1], inputs[2], inputs[3], inputs[4], inputs[6]]
    else:
        units = len(inputs[0] or "") or None
        configuration = list(inputs[1:])
    key = json.dumps([kind, *configuration], ensure_ascii=True)
    state = begin(labels[kind], history.estimate(key, units))
    audio_path = inputs[0] if kind == "asr" else audio
    audio_hash = file_sha256(Path(audio_path)) if audio_path and Path(audio_path).is_file() else None
    state.update(kind=kind, request_id=uuid.uuid4().hex, session_id=None,
                 audio_path=audio_path, audio_sha256=audio_hash, history_key=key,
                 units=units, input_digest=_inputs_digest(inputs), event_order=event_order)
    return state


def start_timed_request(kind, inputs, history, tracker=None, audio=None, event_order=None):
    reservation = uuid.uuid4().hex
    with _WORKFLOW_LOCK:
        if event_order is not None:
            if (type(event_order) is not int or not 0 < event_order < 2**53
                    or (tracker is not None and event_order <= tracker.get("last_event_order", 0))):
                raise ValueError("Obsolete or invalid source event order; request not started.")
            if tracker is not None:
                tracker["last_event_order"] = event_order
        if tracker is not None:
            tracker["pending_start"] = reservation
            tracker["pending_inputs_digest"] = _inputs_digest(inputs)
            tracker["pending_audio_path"] = inputs[0] if kind == "asr" else audio
    try:
        state = _prepare_timed_request(kind, inputs, history, audio, event_order)
        with _WORKFLOW_LOCK:
            # Hashing is deliberately outside the lock; a newer event owns the commit.
            if tracker is not None:
                if (tracker.get("pending_start") != reservation
                        or (event_order is not None and tracker.get("last_event_order") != event_order)):
                    raise ValueError("Obsolete source event order after audio verification.")
                tracker["generation"] += 1
                state["generation"] = tracker["generation"]
                state["session_id"] = tracker.setdefault("session_id", uuid.uuid4().hex)
                tracker.update(active_request_id=state["request_id"], issued=None,
                               reviewed=None, published=None, claimed_request_id=None,
                               active_request=deepcopy(state), busy=True)
        return state
    finally:
        with _WORKFLOW_LOCK:
            if tracker is not None and tracker.get("pending_start") == reservation:
                tracker.update(pending_start=None, pending_inputs_digest=None, pending_audio_path=None)


def _request_identity_current(state, inputs, tracker=None):
    if not state or state.get("input_digest") != _inputs_digest(inputs):
        return False
    if tracker is not None and any([
        state.get("generation") != tracker.get("generation"),
        state.get("request_id") != tracker.get("active_request_id"),
        state.get("session_id") != tracker.get("session_id"),
    ]):
        return False
    if tracker is not None and tracker.get("active_request"):
        if state != tracker["active_request"]:
            return False
    return True


def _request_is_current(state, inputs, tracker=None):
    if not _request_identity_current(state, inputs, tracker):
        return False
    path = state.get("audio_path")
    try:
        audio_matches = not path or (Path(path).is_file() and file_sha256(Path(path)) == state.get("audio_sha256"))
        # Invalidation may run while the file hash is being computed.
        return audio_matches and _request_identity_current(state, inputs, tracker)
    except OSError:
        return False


def issue_proposals(completed, inputs, tracker, current_audio=_AUDIO_UNSPECIFIED):
    """The UI stores an independent, server-owned copy; IDs alone grant no authority."""
    preview, report, state, choices = current_request_result(completed, inputs, tracker, current_audio)
    if report != state or report.get("raw_transcript") != inputs[0]:
        raise ValueError("Proposal source changed before issuing.")
    with _WORKFLOW_LOCK:
        if not _request_is_current(completed.get("request"), inputs, tracker):
            raise ValueError("Request changed or was superseded. Generate a new result.")
        tracker["issued"] = {"report": deepcopy(report), "inputs": deepcopy(list(inputs)),
                             "request": deepcopy(completed["request"]),
                             "proposal_id": uuid.uuid4().hex, "status": "issued"}
    return preview, report, state, choices


def require_issued_proposals(source, report, tracker, audio=None, configuration=None):
    issued = tracker.get("issued")
    if (not issued or report != issued["report"] or source != issued["report"]["raw_transcript"]
            or audio != issued["request"].get("audio_path")
            or (configuration is not None and list(configuration) != issued["inputs"][1:])
            or not _request_is_current(issued["request"], issued["inputs"], tracker)):
        raise ValueError("The proposals were not issued for this source/session, or the request changed.")
    return deepcopy(issued["report"])


def require_published_result(value, tracker, current_inputs=None, current_audio=_AUDIO_UNSPECIFIED):
    published = tracker.get("published")
    if (not published or value != published["value"]
            or (current_inputs is not None and list(current_inputs) != published["inputs"])
            or (current_audio is not _AUDIO_UNSPECIFIED and current_audio != published["request"]["audio_path"])
            or not _request_is_current(published["request"], published["inputs"], tracker)):
        raise ValueError("The published result changed or belongs to an earlier request.")
    return deepcopy(published["value"])


def remember_published(value, completed, tracker):
    with _WORKFLOW_LOCK:
        if not _request_is_current(completed.get("request"), completed["inputs"], tracker):
            raise ValueError("Request changed or was superseded. Generate a new result.")
        tracker["published"] = {"value": deepcopy(value), "request": deepcopy(completed["request"]),
                                "inputs": deepcopy(completed["inputs"])}


def run_timed_request(function, inputs, state, history, tracker=None):
    state = deepcopy(state)
    inputs = deepcopy(list(inputs))

    def current():
        return _request_is_current(state, inputs, tracker)

    def failed_job(message, clock):
        job = {"inputs": deepcopy(inputs), "error": message, "request": deepcopy(state)}
        if clock:
            job["clock"] = finish(processing(clock), False)
        return job

    consumed = tracker is not None and state is not None and tracker.get("claimed_request_id") == state["request_id"]
    if not current() or consumed:
        job = failed_job("Inputs changed or request already consumed while queued. Generate a new result.", state)
        job["discarded"] = consumed or not _request_identity_current(state, inputs, tracker)
        yield job, state, None, False
        return
    if tracker is not None:
        tracker["claimed_request_id"] = state["request_id"]
    active = processing(state)
    yield None, active, render(active), True
    job = _run_request(function, inputs)
    if not current():
        yield failed_job("Inputs changed while processing. Generate a new result.", active), state, None, False
        return
    done = finish(active, "error" not in job)
    job["request"] = deepcopy(state)
    job["clock"] = deepcopy(done)
    job["timing"] = {key: done[key] for key in ("status", "processing_seconds")}
    job["timing"]["total_seconds"] = done["ended_at"] - done["queued_at"]
    if "error" not in job and done["units"] and done["processing_seconds"] > 0:
        history.record(done["history_key"], done["units"], done["processing_seconds"])
    yield job, done, render(done), False


def update_highlights(value, cutoff):
    if value is None:
        return "", None, None, None
    updated = deepcopy(value)
    updated["review_threshold"] = cutoff
    updated["review_revision"] = uuid.uuid4().hex
    updated["review_spans"] = review_spans(
        value["text"], value["words"], value["spoken_language"], cutoff
    )
    return render_highlight_html(value["text"], updated["review_spans"],
                                 value.get("source_audio_sha256"), updated["review_revision"],
                                 value['words']), updated, updated, None


def update_matching_highlights(value, cutoff, audio):
    if value is None:
        return "", None, None, None
    _require_matching_audio(value, audio)
    return update_highlights(value, cutoff)


def build_app():
    import gradio as gr

    from .demo_examples import choices, load_example

    history = TimingHistory(Path(".build/task-timing.json"))
    def sequence_js(kind, force=False):
        source = (ASSETS / "source-order.js").read_text(encoding="utf-8")
        return f"({source})({json.dumps(kind)}, {json.dumps(force)})"

    def delivery_js(kind, indexes, correction_index=-1):
        source = (ASSETS / "result-gate.js").read_text(encoding="utf-8")
        return f"({source})({json.dumps(kind)}, {json.dumps(indexes)}, {correction_index})"

    def task_panel(label):
        # A regular component is snapshotted in the queued event payload. A State
        # would be resolved at dequeue and could adopt a newer request's clock.
        clock = gr.JSON(visible=False)
        tracker = gr.State({"generation": 0})
        status = gr.HTML(render(None), label=label, min_height=52,
                         js_on_load=(ASSETS / "task-clock.js").read_text(encoding="utf-8"))
        return clock, status, tracker

    def start_clock(kind, with_audio=False):
        def start(tracker, event_order, *args):
            current_audio, args = (args[0], args[1:]) if with_audio else (None, args)
            value = start_timed_request(kind, args, history, tracker, audio=current_audio,
                                        event_order=event_order)
            return value, render(value)
        return start

    def timed_worker(function):
        def worker(clock, tracker, *args):
            for job, value, markup, active in run_timed_request(function, args, clock, history, tracker):
                # Terminal clocks publish with the result, after the final identity check.
                current = _request_identity_current(clock, args, tracker)
                yield (job if job is not None and current and not job.get("discarded") else gr.skip(),
                       markup if active and markup is not None and current else gr.skip())
        return worker

    def policy_for(language):
        choices = ["standard_de", "dialect"] if language == "gsw" else ["verbatim"]
        return gr.update(choices=choices, value=choices[0])

    def invalidate_inputs(with_audio=False):
        def invalidate(tracker, event_order=0, *args):
            with _WORKFLOW_LOCK:
                if event_order:
                    if type(event_order) is not int or not 0 < event_order < 2**53:
                        raise ValueError("Invalid source event order")
                    if event_order <= tracker.get("last_event_order", 0):
                        return gr.skip()
                    if args and tracker.get("pending_start"):
                        current_audio, values = (args[0], args[1:]) if with_audio else (args[0], args)
                        if (tracker.get("pending_inputs_digest") == _inputs_digest(values)
                                and current_audio == tracker.get("pending_audio_path")):
                            return gr.skip()
                    tracker["last_event_order"] = event_order
                generation = tracker["generation"]
                receipt = deepcopy(tracker.get("active_request"))
            unchanged = False
            if args:
                current_audio, values = (args[0], args[1:]) if with_audio else (None, args)
                unchanged = bool(receipt and _request_is_current(receipt, values, tracker)
                                 and (not with_audio or current_audio == receipt.get("audio_path")))
            with _WORKFLOW_LOCK:
                if (tracker["generation"] != generation
                        or (event_order and tracker.get("last_event_order") != event_order)
                        or unchanged):
                    return gr.skip()
                return invalidate_task(tracker)
        return invalidate

    def completion_clock(completed, args, tracker, current_audio=_AUDIO_UNSPECIFIED):
        if (completed and completed.get("clock")
                and _request_identity_current(completed.get("request"), args, tracker)
                and (current_audio is _AUDIO_UNSPECIFIED
                     or current_audio == completed["request"].get("audio_path"))
                and ("error" in completed or _request_is_current(completed["request"], args, tracker))):
            return render(completed["clock"])
        return gr.skip()

    def workflow_evidence(completed):
        return {key: completed["request"][key] for key in
                ("request_id", "session_id", "generation", "input_digest", "audio_sha256", "kind", "event_order")}

    def publish_transcription(completed, tracker, *args):
        try:
            raw_text, markup, details, source, state = current_request_result(completed, args, tracker)
            details["workflow"] = workflow_evidence(completed)
            state = deepcopy(details)
            with _WORKFLOW_LOCK:
                clock = completion_clock(completed, args, tracker)
                remember_published(state, completed, tracker)
                return [raw_text, markup, details, source, state, args[1], clock]
        except Exception as exc:
            gr.Warning(str(exc))
            return [*((gr.skip(),) * 6), completion_clock(completed, args, tracker)]

    def publish_note(completed, tracker, current_audio, *args):
        try:
            markup, note = current_request_result(completed, args, tracker, current_audio)
            note["workflow"] = workflow_evidence(completed)
            with _WORKFLOW_LOCK:
                clock = completion_clock(completed, args, tracker, current_audio)
                remember_published(note, completed, tracker)
                return [markup, note, clock]
        except Exception as exc:
            gr.Warning(str(exc))
            return [gr.skip(), gr.skip(), completion_clock(completed, args, tracker, current_audio)]

    def publish_proposals(completed, tracker, current_audio, *args):
        try:
            with _WORKFLOW_LOCK:
                clock = completion_clock(completed, args, tracker, current_audio)
                preview, report, state, choices = issue_proposals(completed, args, tracker, current_audio)
                return [
                    preview,
                    report,
                    state,
                    gr.update(choices=choices, value=[]),
                    "",
                    None,
                    None,
                    False,
                    None,
                    clock,
                ]
        except Exception as exc:
            gr.Warning(str(exc))
            return [*((gr.skip(),) * 9), completion_clock(completed, args, tracker, current_audio)]

    def safe_review(source, report, selected, confirmed, tracker, current_audio,
                    current_language=None, current_model=None):
        try:
            return list(review_corrections_ui(source, report, selected, confirmed, tracker, current_audio,
                                             [current_language, current_model])[:3])
        except Exception as exc:
            gr.Warning(str(exc))
            return ["", {"status": "failed", "error": str(exc)}, None]

    def transfer_note(source, report, selected, confirmed, reviewed, tracker, current_audio,
                      current_language, current_model):
        try:
            return (note_source_for_review(source, report, selected, confirmed, reviewed,
                                          tracker, current_audio, [current_language, current_model]),
                    gr.update(selected="notes"))
        except Exception as exc:
            gr.Warning(str(exc))
            return gr.skip(), gr.skip()

    def export_asr_result(value, tracker, *args):
        value = _workflow_snapshot(value, "published")
        return export_matching_audio(require_published_result(value, tracker, args), args[0])

    def export_note_result(value, tracker, source, method, model, current_audio):
        return export_matching_source(require_published_result(value, tracker, [source, method, model],
                                                               current_audio), source)

    def export_proposal_result(value, source, tracker, current_audio, current_language, current_model):
        value = _workflow_snapshot(value, "issued")
        report = require_issued_proposals(source, value, tracker, current_audio, [current_language, current_model])
        report["workflow"] = {key: tracker["issued"]["request"][key] for key in
                              ("request_id", "session_id", "generation", "audio_sha256")}
        report["workflow"]["proposal_id"] = tracker["issued"]["proposal_id"]
        return _export(report)

    def export_reviewed_result(value, source, confirmed, selected, tracker, current_audio,
                               current_language, current_model):
        value = _workflow_snapshot(value, "reviewed")
        require_issued_proposals(source, (tracker.get("issued") or {}).get("report"), tracker,
                                current_audio, [current_language, current_model])
        if value != tracker.get("reviewed"):
            raise ValueError("The reviewed result changed or belongs to an earlier request.")
        return export_reviewed_source(value, source, confirmed, selected)

    def refresh_highlights(value, cutoff, current_audio, tracker):
        value = _workflow_snapshot(value, "published")
        if value is None:
            return ["", None, None, None]
        require_published_result(value, tracker)
        markup, details, state, old_export = update_matching_highlights(value, cutoff, current_audio)
        tracker["generation"] += 1
        state["workflow"]["generation"] = tracker["generation"]
        published = tracker["published"]
        published["inputs"][5] = cutoff
        published["request"]["generation"] = tracker["generation"]
        published["request"]["input_digest"] = _inputs_digest(published["inputs"])
        tracker["active_request"] = deepcopy(published["request"])
        tracker["published"]["value"] = deepcopy(state)
        return [markup, details, state, old_export]

    def fictional_example():
        text = "The patient reports fatigue after the last cycle. She denies fever. Treatment with cis plat in was discussed. The documented dose is 2.5 mg."
        return text

    def load_dialogue(identifier):
        path, spoken_language, script, speech_reference = load_example(identifier)
        return path, spoken_language, gr.update(choices=["verbatim"], value="verbatim"), script, speech_reference

    def prepare_word(value, current_audio, tracker, evt):
        generation = None
        try:
            value = _workflow_snapshot(value, "published")
            data = evt._data
            if not value or not data.get('revision') or data['revision'] != value.get('review_revision'):
                raise ValueError('This word belongs to an earlier transcript.')
            invalidate_playback(tracker)
            generation = tracker['generation']
            clip = word_audio_excerpt(value, current_audio, data.get('start'), data.get('end'),
                                      data.get('source_sha256'), data.get('revision'))
            if tracker['generation'] != generation:
                return (gr.skip(),) * 4
            clip['playback_generation'] = generation
            audio_update, label, published = publish_excerpt(clip, value, current_audio, tracker)
            return clip, audio_update, label, published
        except Exception as exc:
            gr.Warning(str(exc))
            if generation is not None and tracker['generation'] == generation:
                return None, None, '', None
            return (gr.skip(),) * 4

    prepare_word.__annotations__['evt'] = gr.EventData

    def publish_excerpt(clip, value, current_audio, tracker):
        try:
            value = _workflow_snapshot(value, "published")
            require_current_excerpt(clip, value, current_audio, tracker)
            label = (f'<div class="playback-selection" aria-live="polite">'
                     f'<strong>{html.escape(clip["word"])}</strong>'
                     f'<span>{clip["start"]:.2f}-{clip["end"]:.2f} s | Estimated alignment</span></div>')
            return gr.update(value=clip['audio'], playback_position=0), label, clip
        except ValueError as exc:
            if clip:
                gr.Warning(str(exc))
            return gr.skip(), gr.skip(), gr.skip()

    def replay_excerpt(clip, value, current_audio, tracker):
        try:
            value = _workflow_snapshot(value, "published")
            clip = reserve_playback_action(clip, value, tracker)
            require_current_excerpt(clip, value, current_audio, tracker)
            yield None, ''
            yield publish_excerpt(clip, value, current_audio, tracker)[:2]
        except (ValueError, OSError) as exc:
            gr.Warning(str(exc))
            yield gr.skip(), gr.skip()

    def continue_audio(clip, value, current_audio, tracker):
        try:
            value = _workflow_snapshot(value, "published")
            clip = reserve_playback_action(clip, value, tracker)
            with continuation_excerpt(clip, value, current_audio, tracker) as tail:
                cached = excerpt_audio.move_resource_to_block_cache(tail)
            require_current_excerpt(clip, value, current_audio, tracker)
            yield None, ''
            require_current_excerpt(clip, value, current_audio, tracker)
            yield (gr.update(value=cached, playback_position=0),
                   f"Consultation audio from {clip['end']:.2f} s")
        except (ValueError, OSError) as exc:
            gr.Warning(str(exc))
            yield gr.skip(), gr.skip()

    def reset_word_audio(tracker):
        invalidate_playback(tracker)
        return None, '', None

    with gr.Blocks(
        title="Oncology Scribe | Clinical Workspace",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as app:
        event_sequence = gr.Number(value=0, precision=0, visible=False)
        transcription_job = gr.State()
        proposal_job = gr.State()
        note_job = gr.State()
        gr.HTML(
            '<header class="workspace-header"><div><span class="brand-label">CLINICAL WORKSPACE</span>'
            '<h1>Oncology Scribe</h1></div><div class="header-status">'
            '<span class="local-status">Local processing</span>'
            '<span class="prototype-status">Research prototype</span></div></header>',
            js_on_load="\n".join((ASSETS / name).read_text(encoding="utf-8")
                                 for name in ("field-labels.js", "workflow-lock.js")),
        )
        with gr.Tabs() as tabs:
            with gr.Tab("Consultation", id="consultation"):
                word_job = gr.State()
                excerpt_state = gr.State()
                playback_tracker = gr.State({'generation': 0})
                with gr.Row(elem_classes="consultation-layout"):
                    with gr.Column(scale=1, min_width=280, elem_classes="input-column"):
                        gr.Markdown("### Consultation")
                        with gr.Accordion("Fictional dialogues", open=False):
                            dialogue = gr.Dropdown(choices(), label="Fictional consultation",
                                                   elem_classes='dialogue-selector')
                            load_dialogue_button = gr.Button("Load dialogue", icon=_icon("audio-lines"))
                        audio = gr.Audio(
                            sources=["upload"], type="filepath", label="Consultation audio"
                        )
                        language = gr.Dropdown(LANGUAGES, value="en", label="Spoken language")
                        policy = gr.Dropdown(
                            ["verbatim"], value="verbatim", label="Reference policy"
                        )
                        with gr.Accordion("ASR settings", open=False):
                            model = gr.Dropdown(
                                [
                                    ("Whisper Small", "small"),
                                    ("Whisper Large-v3", "large-v3"),
                                    ("Whisper Turbo", "turbo"),
                                ],
                                value="turbo",
                                label="ASR model",
                            )
                            device = gr.Radio(
                                [("CPU", "cpu"), ("GPU / CUDA", "cuda")],
                                value="cpu",
                                label="Device",
                            )
                            experimental = gr.Checkbox(False, label="Experimental dialect output")
                        transcribe = gr.Button(
                            "Transcribe audio", variant="primary", icon=_icon("audio-lines")
                        )
                        asr_clock, asr_status, asr_tracker = task_panel("Transcription timing")
                    with gr.Column(scale=2, min_width=280, elem_classes="transcript-column"):
                        gr.Markdown("### Transcript")
                        with gr.Column(elem_classes='transcript-playback', min_width=0):
                            excerpt_label = gr.HTML('', elem_id='transcript-audio-selection')
                            excerpt_audio = gr.Audio(label="Selected audio excerpt", interactive=False,
                                                     autoplay=True, buttons=[], editable=False,
                                                     elem_id='transcript-audio-player')
                            with gr.Row():
                                replay_word = gr.Button('Replay excerpt', size='sm', icon=_icon('audio-lines'))
                                continue_audio_button = gr.Button('Continue audio', size='sm', icon=_icon('arrow-right'))
                        marked = gr.HTML(label="Audio-linked transcript", elem_classes="transcript-cues",
                                         js_on_load=(ASSETS / "word-audio.js").read_text(encoding="utf-8"))
                        with gr.Accordion("Raw text copy", open=False):
                            raw = gr.Textbox(
                                label="Raw transcript", lines=5, interactive=False, buttons=["copy"],
                                elem_id="raw-transcript-field",
                            )
                        with gr.Accordion("Review cue settings", open=False):
                            threshold = gr.Slider(
                                0, 1, value=0.6, step=0.05, label="Uncalibrated word-score cutoff"
                            )
                        with gr.Accordion("Run details", open=False):
                            details = gr.JSON(label="ASR result")
                        with gr.Row():
                            continue_medical = gr.Button(
                                "Medical review", variant="primary", icon=_icon("arrow-right")
                            )
                            export_asr = gr.Button(
                                "Export transcript", size="sm", icon=_icon("download")
                            )
                        asr_download = gr.File(label="Transcript export", interactive=False)
                        with gr.Accordion("Synthetic script reference", open=False):
                            gr.HTML('<div class="safety-band">Fictional dialogue / Synthetic speech / '
                                    'Clinical and native-speaker review pending.</div>')
                            dialogue_script = gr.Textbox(label="Dialogue script", lines=7, interactive=False)
                            speech_reference = gr.Textbox(label="Synthetic speech reference", lines=5,
                                                          interactive=False, buttons=["copy"])
            with gr.Tab("Medical Review", id="medical"):
                gr.HTML(
                    '<div class="safety-band">Draft proposals only. Clinical approval pending.</div>'
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        correction_source = gr.Textbox(
                            label="Transcript for medical review", lines=8,
                            elem_id="medical-transcript-field",
                        )
                        correction_language = gr.Dropdown(
                            LANGUAGES, value="en", label="Review language"
                        )
                        correction_model = gr.Dropdown(
                            [
                                ("Qwen3.5 4B (English candidate)", "qwen3.5:4b"),
                                ("MedGemma 4B (local)", "medgemma:4b"),
                                ("Llama 3.1 (comparison)", "llama3.1:latest"),
                            ],
                            value="qwen3.5:4b",
                            label="Medical correction model",
                        )
                        propose = gr.Button(
                            "Propose terminology edits", variant="primary", icon=_icon("sparkles")
                        )
                        proposal_clock, proposal_status, proposal_tracker = task_panel(
                            "Terminology review timing"
                        )
                        demo_text = gr.Button("Load fictional example", size="sm")
                    with gr.Column(scale=2, min_width=280):
                        gr.Markdown("### Proposed changes")
                        correction_markup = gr.HTML(
                            '<div class="empty-result">No proposals yet.</div>'
                        )
                        # Choices are delivered by the client gate. Authorization
                        # comes from the server-owned report, not widget metadata.
                        selected_edits = gr.Dropdown(
                            [], value=[], multiselect=True, allow_custom_value=True,
                            max_choices=16, label="Changes to include",
                            info="Select proposals from the current transcript. Unissued edits are rejected.",
                        )
                        audio_confirmed = gr.Checkbox(
                            False, label="I checked the selected changes against the audio"
                        )
                        apply_edits = gr.Button(
                            "Create reviewed draft", variant="primary", icon=_icon("check")
                        )
                reviewed_text = gr.Textbox(
                    label="Reviewed transcript draft", lines=6, interactive=False, buttons=["copy"],
                    elem_id="reviewed-transcript-field",
                )
                with gr.Row():
                    to_note = gr.Button("Note draft", icon=_icon("arrow-right"))
                    export_corrections = gr.Button(
                        "Export proposals", icon=_icon("download"), size="sm"
                    )
                    export_reviewed = gr.Button(
                        "Export reviewed draft", icon=_icon("download"), size="sm"
                    )
                with gr.Row():
                    correction_download = gr.File(label="Proposal export", interactive=False)
                    reviewed_download = gr.File(label="Reviewed draft export", interactive=False)
                with gr.Accordion("Correction audit", open=False):
                    correction_json = gr.JSON(label="Proposal evidence")
                    reviewed_json = gr.JSON(label="Review decisions")
            with gr.Tab("Note Draft", id="notes"):
                gr.HTML('<div class="safety-band">Clinical approval pending.</div>')
                note_source = gr.Textbox(label="Transcript for note", lines=7,
                                         elem_id="note-transcript-field")
                with gr.Row():
                    method = gr.Radio(
                        [("SOAP categories", "positional"), ("Extractive points", "extractive")],
                        value="positional",
                        label="Note method",
                    )
                    llm = gr.Dropdown(
                        ["llama3.1:latest", "medgemma:4b", "qwen3.5:4b", "mistral:latest"],
                        value="llama3.1:latest",
                        label="Local note model",
                    )
                generate = gr.Button(
                    "Generate note draft", variant="primary", icon=_icon("clipboard-list")
                )
                note_clock, note_status, note_tracker = task_panel("Note draft timing")
                preview = gr.HTML(label="Categorized draft", elem_classes="note-preview")
                with gr.Accordion("Evidence and uncovered source", open=False):
                    note_json = gr.JSON(label="Draft evidence")
                export_note = gr.Button("Export note draft", size="sm", icon=_icon("download"))
                note_download = gr.File(label="Note export", interactive=False)
            with gr.Tab("Evaluation", id="evaluation"):
                with gr.Row():
                    eval_language = gr.Dropdown(LANGUAGES, value="en", label="Language")
                    eval_policy = gr.Dropdown(
                        ["verbatim"], value="verbatim", label="Reference policy"
                    )
                with gr.Row():
                    reference = gr.Textbox(label="Human reference", lines=7)
                    hypothesis = gr.Textbox(label="ASR hypothesis", lines=7)
                score = gr.Button("Evaluate", variant="primary")
                metrics = gr.JSON(label="Lexical metrics")

        gr.HTML(
            '<footer class="workspace-footer"><span>Oncology research workspace</span>'
            "<span>EN / DE / GSW / FR / IT</span><span>Human review required</span></footer>"
        )

        locked_controls = [
            audio, language, policy, model, device, experimental, threshold,
            correction_source, correction_language, correction_model, selected_edits,
            audio_confirmed, note_source, method, llm, transcribe, propose, apply_edits,
            demo_text, to_note, generate, export_asr, export_note, export_corrections,
            export_reviewed,
            dialogue, load_dialogue_button,
        ]
        for control in locked_controls:
            control.elem_classes = [*(control.elem_classes or []), "workflow-controlled"]

        def lock_inputs():
            return [gr.update(interactive=False) for _ in locked_controls]

        def unlock_inputs(completed, tracker, *all_trackers):
            if not completed or not _request_identity_current(completed.get("request"), completed.get("inputs", []), tracker):
                return [gr.skip() for _ in locked_controls]
            tracker["busy"] = False
            return unlock_review(*all_trackers)

        def unlock_review(*trackers):
            with _WORKFLOW_LOCK:
                if any(tracker.get("busy") or tracker.get("pending_start") for tracker in trackers):
                    return [gr.skip() for _ in locked_controls]
                return [gr.update(interactive=True) for _ in locked_controls]

        all_trackers = [asr_tracker, proposal_tracker, note_tracker]
        # Reports remain server-owned; only versioned delivery frames cross to the UI.
        state = asr_tracker
        correction_state = reviewed_state = proposal_tracker
        asr_delivery = gr.JSON(visible=False, elem_classes="delivery-frame")
        proposal_delivery = gr.JSON(visible=False, elem_classes="delivery-frame")
        note_delivery = gr.JSON(visible=False, elem_classes="delivery-frame")
        review_delivery = gr.JSON(visible=False, elem_classes="delivery-frame")
        highlight_delivery = gr.JSON(visible=False, elem_classes="delivery-frame")

        language.change(policy_for, language, policy)
        load_dialogue_button.click(load_dialogue, dialogue,
                                   [audio, language, policy, dialogue_script, speech_reference],
                                   queue=False)
        for event in (audio.upload, audio.clear):
            event(lambda: ("", ""), outputs=[dialogue_script, speech_reference], queue=False)
        eval_language.change(policy_for, eval_language, eval_policy)
        asr_inputs = [audio, language, policy, model, device, threshold, experimental]
        asr_start = transcribe.click(lock_inputs, outputs=locked_controls, queue=False).then(
            start_clock("asr"), [asr_tracker, event_sequence, *asr_inputs], [asr_clock, asr_status],
            queue=False, js=sequence_js("asr", force=True),
        )
        asr_start.then(unlock_review, all_trackers, locked_controls, queue=False)
        asr_start.success(
            lambda: ("", "", None, "", "", None, None, None, ""),
            outputs=[raw, marked, details, note_source, preview, note_json,
                     asr_download, note_download, correction_source],
            queue=False,
        ).then(
            timed_worker(transcribe_ui),
            [asr_clock, asr_tracker, *asr_inputs],
            [transcription_job, asr_status],
            concurrency_limit=1,
            concurrency_id="asr",
            show_progress="hidden",
        ).then(
            publish_transcription,
            [transcription_job, asr_tracker, *asr_inputs],
            asr_delivery,
            queue=False,
        ).then(
            fn=None, inputs=asr_delivery,
            outputs=[raw, marked, details, correction_source, correction_language, asr_status],
            js=delivery_js("asr", [0, 1, 2, 3, 5, 6]), queue=False,
        ).then(unlock_inputs, [transcription_job, asr_tracker, *all_trackers], outputs=locked_controls, queue=False)
        continue_medical.click(lambda: gr.update(selected="medical"), outputs=tabs)
        to_note.click(lock_inputs, outputs=locked_controls, queue=False).then(
            transfer_note,
            [correction_source, correction_state, selected_edits, audio_confirmed, reviewed_state,
             proposal_tracker, audio, correction_language, correction_model],
            [note_source, tabs], queue=False,
        ).then(unlock_review, all_trackers, outputs=locked_controls, queue=False)
        proposal_start = propose.click(lock_inputs, outputs=locked_controls, queue=False).then(
            start_clock("corrections", with_audio=True),
            [proposal_tracker, event_sequence, audio, correction_source, correction_language, correction_model],
            [proposal_clock, proposal_status], queue=False, js=sequence_js("corrections", force=True),
        )
        proposal_start.then(unlock_review, all_trackers, locked_controls, queue=False)
        proposal_start.success(
            lambda: ("", None, gr.update(choices=[], value=[]), "", None,
                     False, None, None, ""),
            outputs=[correction_markup, correction_json, selected_edits,
                     reviewed_text, reviewed_json, audio_confirmed,
                     correction_download, reviewed_download, note_source],
            queue=False,
        ).then(
            timed_worker(propose_corrections_ui),
            [proposal_clock, proposal_tracker, correction_source, correction_language, correction_model],
            [proposal_job, proposal_status],
            concurrency_limit=1,
            concurrency_id="llm",
            show_progress="hidden",
        ).then(
            publish_proposals,
            [proposal_job, proposal_tracker, audio, correction_source, correction_language, correction_model],
            proposal_delivery,
            queue=False,
        ).then(
            fn=None, inputs=proposal_delivery,
            outputs=[correction_markup, correction_json, selected_edits, reviewed_text,
                     reviewed_json, audio_confirmed, reviewed_download, proposal_status],
            js=delivery_js("corrections", [0, 1, 3, 4, 5, 7, 8, 9], 2), queue=False,
        ).then(unlock_inputs, [proposal_job, proposal_tracker, *all_trackers], outputs=locked_controls, queue=False)
        apply_edits.click(lock_inputs, outputs=locked_controls, queue=False).then(
            safe_review,
            [correction_source, correction_state, selected_edits, audio_confirmed, proposal_tracker,
             audio, correction_language, correction_model],
            review_delivery,
            queue=False,
        ).then(
            fn=None, inputs=review_delivery, outputs=[reviewed_text, reviewed_json],
            js=delivery_js("corrections", [0, 1]), queue=False,
        ).then(unlock_review, all_trackers, outputs=locked_controls, queue=False)
        demo_text.click(fictional_example, outputs=correction_source)

        threshold.change(
            refresh_highlights, [state, threshold, audio, asr_tracker],
            highlight_delivery, api_name="update_highlights", queue=False,
        ).then(
            fn=None, inputs=highlight_delivery, outputs=[marked, details, asr_download],
            js=delivery_js("asr", [0, 1, 3]), queue=False,
        )
        marked.play_word(
            prepare_word, [state, audio, playback_tracker], [word_job, excerpt_audio, excerpt_label, excerpt_state],
            api_name='listen_word', queue=False,
        )
        replay_word.click(
            replay_excerpt, [excerpt_state, state, audio, playback_tracker],
            [excerpt_audio, excerpt_label], queue=True, concurrency_limit=None)
        continue_audio_button.click(
            continue_audio, [excerpt_state, state, audio, playback_tracker],
            [excerpt_audio, excerpt_label], queue=True, concurrency_limit=None)
        note_start = generate.click(lock_inputs, outputs=locked_controls, queue=False).then(
            start_clock("note", with_audio=True), [note_tracker, event_sequence, audio, note_source, method, llm],
            [note_clock, note_status], queue=False, js=sequence_js("note", force=True),
        )
        note_start.then(unlock_review, all_trackers, locked_controls, queue=False)
        note_start.success(
            lambda: ("", None, None), outputs=[preview, note_json, note_download], queue=False,
        ).then(
            timed_worker(note_preview),
            [note_clock, note_tracker, note_source, method, llm],
            [note_job, note_status],
            concurrency_limit=1,
            concurrency_id="llm",
            show_progress="hidden",
        ).then(publish_note, [note_job, note_tracker, audio, note_source, method, llm],
               note_delivery, queue=False
               ).then(fn=None, inputs=note_delivery, outputs=[preview, note_json, note_status],
                      js=delivery_js("note", [0, 1, 2]), queue=False
               ).then(unlock_inputs, [note_job, note_tracker, *all_trackers], outputs=locked_controls, queue=False)
        score.click(compare_texts, [reference, hypothesis, eval_language, eval_policy], metrics)
        export_asr.click(export_asr_result, [state, asr_tracker, *asr_inputs], asr_download)
        export_note.click(export_note_result, [note_json, note_tracker, note_source, method, llm, audio], note_download)
        export_corrections.click(
            export_proposal_result, [correction_state, correction_source, proposal_tracker, audio,
                                    correction_language, correction_model], correction_download
        )
        export_reviewed.click(
            export_reviewed_result,
            [reviewed_state, correction_source, audio_confirmed, selected_edits, proposal_tracker,
             audio, correction_language, correction_model],
            reviewed_download,
        )
        def invalidate_asr_result(tracker, event_order, *args):
            status = invalidate_inputs()(tracker, event_order, *args)
            if status == gr.skip():
                return (gr.skip(),) * 10
            return "", "", None, "", "", None, None, None, "", status

        for asr_input in (audio, language, policy, model, device, experimental):
            asr_input.change(
                invalidate_asr_result,
                inputs=[asr_tracker, event_sequence, *asr_inputs],
                outputs=[raw, marked, details, note_source, preview, note_json,
                         asr_download, note_download, correction_source, asr_status],
                queue=False, show_progress="hidden", js=sequence_js("asr"),
            ).then(unlock_review, all_trackers, locked_controls, queue=False)
        for note_input in (note_source, method, llm):
            note_input.change(lambda: ("", None, None),
                              outputs=[preview, note_json, note_download], queue=False)
        for source_input in (*asr_inputs, transcribe):
            event = source_input.click if source_input is transcribe else source_input.change
            event(reset_word_audio, playback_tracker, outputs=[excerpt_audio, excerpt_label, excerpt_state],
                  queue=False)
        correction_outputs = [
            correction_markup,
            correction_json,
            selected_edits,
            reviewed_text,
            reviewed_json,
            audio_confirmed,
            correction_download,
            reviewed_download,
            note_source,
        ]

        def reset_corrections():
            return (
                "",
                None,
                gr.update(choices=[], value=[]),
                "",
                None,
                False,
                None,
                None,
                "",
            )

        correction_source.change(reset_corrections, outputs=correction_outputs, queue=False)
        correction_language.change(reset_corrections, outputs=correction_outputs, queue=False)
        correction_model.change(reset_corrections, outputs=correction_outputs, queue=False)
        selected_edits.change(
            lambda: ("", None, False, None, ""),
            outputs=[
                reviewed_text,
                reviewed_json,
                audio_confirmed,
                reviewed_download,
                note_source,
            ],
            queue=False,
        )

        def confirmation_changed(confirmed, reviewed):
            reviewed = _workflow_snapshot(reviewed, "reviewed")
            if not reviewed or reviewed.get("audio_confirmed") is confirmed:
                return (gr.skip(),) * 4
            return "", None, None, ""

        audio_confirmed.change(
            confirmation_changed,
            [audio_confirmed, reviewed_state],
            [reviewed_text, reviewed_json, reviewed_download, note_source],
            queue=False,
        )
        for control in (correction_source, correction_language, correction_model):
            control.change(invalidate_inputs(True),
                           inputs=[proposal_tracker, event_sequence, audio, correction_source, correction_language, correction_model],
                           outputs=proposal_status,
                           queue=False, show_progress="hidden", js=sequence_js("corrections")).then(
                               unlock_review, all_trackers, locked_controls, queue=False)
        for control in (note_source, method, llm):
            control.change(invalidate_inputs(True), inputs=[note_tracker, event_sequence, audio, note_source, method, llm],
                           outputs=note_status,
                           queue=False, show_progress="hidden", js=sequence_js("note")).then(
                               unlock_review, all_trackers, locked_controls, queue=False)
        audio.change(invalidate_inputs(True),
                     inputs=[proposal_tracker, event_sequence, audio, correction_source, correction_language, correction_model],
                     outputs=proposal_status,
                     queue=False, show_progress="hidden", js=sequence_js("corrections")).then(
                         unlock_review, all_trackers, locked_controls, queue=False)
        audio.change(invalidate_inputs(True), inputs=[note_tracker, event_sequence, audio, note_source, method, llm], outputs=note_status,
                     queue=False, show_progress="hidden", js=sequence_js("note")).then(
                         unlock_review, all_trackers, locked_controls, queue=False)
    return app


def launch(port=7860):
    import gradio as gr

    app = build_app()
    theme = gr.themes.Default(primary_hue="emerald", neutral_hue="gray")
    tokens = theme.to_dict()["theme"]
    theme.set(
        **{key: tokens[key[:-5]] for key in tokens if key.endswith("_dark") and key[:-5] in tokens}
    )
    english_controls = {
        "audio.drop_to_upload": "Drop an audio file here to upload",
        "upload_text.drop_audio": "Drop audio here",
        "upload_text.click_to_upload": "Click to upload",
        "common.or": "or",
        "common.empty": "Empty",
        "common.remove": "Remove",
        "common.loading": "Loading",
        "file.uploading": "Uploading...",
        "common.clear": "Clear",
        "common.download": "Download",
        "common.share": "Share",
        "audio.play": "Play",
        "audio.pause": "Pause",
        "audio.stop": "Stop",
    }
    i18n = gr.I18n(
        **{locale: english_controls for locale in ["en", "pt", "pt-BR", "de", "fr", "it"]}
    )
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_error=False,
        footer_links=[],
        max_file_size="100mb",
        theme=theme,
        i18n=i18n,
        css=(ASSETS / "workbench.css").read_text(encoding="utf-8"),
    )
