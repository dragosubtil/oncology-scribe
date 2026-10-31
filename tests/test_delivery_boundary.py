"""Regression for publication delivery after server callback validation.

Run from the candidate oncology-scribe repository with its existing environment:
    .venv/Scripts/python.exe -B -m pytest -c pyproject.toml -q -p no:cacheprovider \
        "<absolute path to this file>"

Only Gradio's postprocessing is paused. Request validation, invalidation,
publication, serialization, and component/event registration remain real.
No server is launched and no model is called. JavaScript/browser tests must
separately verify that the configured client gate rejects stale delivery frames.
"""

import asyncio
import hashlib
from copy import deepcopy
from html.parser import HTMLParser

import pytest
from gradio.state_holder import SessionState

from clinical_asr import demo
from clinical_asr.timing import TimingHistory, finish, processing

OLD_SOURCE = "No fever."
NEW_SOURCE = "Temperature is 39 C."
OLD_PREVIEW = "<p>OLD NOTE: No fever.</p>"


def _dependency(app, api_name):
    matches = [d for d in app.config["dependencies"] if d["api_name"] == api_name]
    assert len(matches) == 1, f"Expected one {api_name} event, found {len(matches)}"
    return matches[0]


def _component_id(app, label):
    matches = [c["id"] for c in app.config["components"] if c["props"].get("label") == label]
    assert len(matches) == 1, f"Expected one component labelled {label!r}"
    return matches[0]


def _client_delivery(app, publisher):
    """Publish one complete JSON frame, then deliver it through client-only code."""
    components = {c["id"]: c for c in app.config["components"]}
    assert len(publisher["outputs"]) == 1, (
        f"{publisher['api_name']} must publish one complete delivery frame"
    )
    frame_id = publisher["outputs"][0]
    frame_component = components[frame_id]
    assert frame_component["type"] == "json", "Serialize the whole publisher tuple as JSON"
    assert not app.blocks[frame_id].stateful, "The client gate must receive the serialized frame"
    assert frame_component["props"].get("visible", True) is False, (
        f"{publisher['api_name']} binds its late server response directly to a visible "
        "component; publish into a hidden JSON delivery frame instead"
    )

    gates = [
        dependency for dependency in app.config["dependencies"]
        if dependency.get("trigger_after") == publisher["id"]
        and dependency.get("backend_fn") is False
        and isinstance(dependency.get("js"), str)
        and dependency["js"].strip()
        and dependency["inputs"] == [frame_id]
        and dependency["outputs"]
    ]
    assert len(gates) == 1, (
        f"{publisher['api_name']} needs one JS-only delivery successor consuming "
        "only its single hidden JSON frame"
    )
    return frame_id, gates[0]


class _ClockStamp(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stamps = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-task-clock" in attrs:
            self.stamps.append(attrs)


def test_late_postprocessing_stages_old_note_without_targeting_visible_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    app = demo.build_app()
    publisher = _dependency(app, "publish_note")
    preview_id = _component_id(app, "Categorized draft")
    evidence_id = _component_id(app, "Draft evidence")
    status_id = _component_id(app, "Note draft timing")
    source_id = _component_id(app, "Transcript for note")
    invalidators = [
        d for d in app.config["dependencies"]
        if d["api_name"].startswith("invalidate")
        and [source_id, "change"] in [list(target) for target in d["targets"]]
    ]
    assert len(invalidators) == 1, "Use the real note-source invalidation event"
    invalidator = invalidators[0]

    async def exercise():
        session = SessionState(app)
        tracker = session[publisher["inputs"][1]]
        args = [OLD_SOURCE, "extractive", "llama3.1:latest"]
        request = demo.start_timed_request(
            "note", args, TimingHistory(tmp_path / "timing.json"), tracker, event_order=1
        )
        session[publisher["inputs"][0]] = {
            "inputs": deepcopy(args),
            "request": deepcopy(request),
            "clock": finish(processing(deepcopy(request)), True),
            "result": (OLD_PREVIEW, demo.extractive_note(OLD_SOURCE)),
        }

        callback_returned = asyncio.Event()
        release_postprocessing = asyncio.Event()
        real_postprocess = app.postprocess_data

        async def paused_postprocess(block_fn, predictions, state):
            if block_fn is app.fns[publisher["id"]]:
                callback_returned.set()
                await release_postprocessing.wait()
            return await real_postprocess(block_fn, predictions, state)

        monkeypatch.setattr(app, "postprocess_data", paused_postprocess)
        publication = asyncio.create_task(
            app.process_api(publisher["id"], [None, None, None, *args], state=session)
        )
        try:
            await asyncio.wait_for(callback_returned.wait(), timeout=10)
            assert tracker["generation"] == 1
            assert tracker["published"]["value"]["raw_transcript"] == OLD_SOURCE

            # This is a different event, run after the publisher released its lock
            # but before Gradio processes that publisher's returned output tuple.
            await asyncio.wait_for(
                app.process_api(
                    invalidator["id"],
                    [None, 2, None, NEW_SOURCE, *args[1:]],
                    state=session,
                ),
                timeout=10,
            )
            assert tracker["generation"] == 2
            assert tracker["last_event_order"] == 2
            assert tracker["published"] is None

            release_postprocessing.set()
            response = await asyncio.wait_for(publication, timeout=10)
        finally:
            release_postprocessing.set()
            if not publication.done():
                publication.cancel()
            await asyncio.gather(publication, return_exceptions=True)

        # Preserve the reproduction: the entire old tuple survives real Gradio
        # postprocessing in ONE frame. The JS gate receives this coherent array,
        # rather than separately updated preview/evidence/clock component values.
        assert len(response["data"]) == 1, "Return one serialized delivery frame"
        frame = response["data"][0]
        if hasattr(frame, "model_dump"):
            frame = frame.model_dump()
        assert isinstance(frame, list), "The publisher tuple must serialize as a JSON array"
        assert len(frame) == 3, "Keep preview, note evidence, and terminal clock together"
        old_preview, old_note, terminal_clock = frame
        assert old_preview == OLD_PREVIEW, "The old preview must survive postprocessing"
        assert isinstance(old_note, dict)
        assert old_note["raw_transcript"] == OLD_SOURCE
        assert old_note["workflow"]["generation"] == 1
        assert old_note["workflow"]["kind"] == "note"
        assert old_note["workflow"]["event_order"] == 1
        assert old_note["workflow"]["request_id"] == request["request_id"]
        assert old_note["workflow"]["session_id"] == request["session_id"]
        assert tracker["generation"] == 2
        assert tracker["last_event_order"] == 2
        assert tracker["published"] is None, "Delivery must not restore export authorization"
        with pytest.raises(ValueError, match="changed|earlier"):
            demo.require_published_result(old_note, tracker)

        frame_id, gate = _client_delivery(app, publisher)
        assert publisher["outputs"] == [frame_id]
        assert {preview_id, evidence_id, status_id}.isdisjoint(publisher["outputs"]), (
            "The server publisher must not bypass the gate for preview, JSON, or status"
        )
        assert gate["inputs"] == [frame_id], "Consume one coherent frame, not scattered values"
        assert gate["outputs"] == [preview_id, evidence_id, status_id], (
            "The client-only successor must deliver preview, evidence, and clock together"
        )
        assert isinstance(terminal_clock, str)
        stamp = _ClockStamp()
        stamp.feed(terminal_clock)
        assert len(stamp.stamps) == 1
        assert stamp.stamps[0].get("data-task-kind") == "note"
        assert stamp.stamps[0].get("data-generation") == "1"
        assert stamp.stamps[0].get("data-event-order") == "1", (
            "Stamp the old request's order, never the newer source event's order"
        )
        assert stamp.stamps[0].get("data-status") == "completed"

    asyncio.run(exercise())


@pytest.mark.parametrize("publisher_name", [
    "publish_transcription", "publish_proposals", "publish_note",
])
def test_all_publishers_stage_browser_outputs_before_js_only_delivery(
    monkeypatch, tmp_path, publisher_name
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    app = demo.build_app()
    _client_delivery(app, _dependency(app, publisher_name))


@pytest.mark.parametrize(("selection", "confirmed", "success"), [
    (["e0"], True, True), (["not-issued"], True, False), (["e0"], False, False),
])
def test_client_delivered_choices_use_server_issued_report_validation(
    monkeypatch, tmp_path, selection, confirmed, success
):
    monkeypatch.chdir(tmp_path)
    app = demo.build_app()
    event = _dependency(app, "safe_review")
    source = "Trastuzumap 5 mg."
    report = {"status": "proposed", "raw_transcript": source,
        "source_transcript_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "language": "en", "model": "qwen3.5:4b",
        "proposals": [{"id": "e0", "span_id": "s0", "before": "Trastuzumap",
            "after": "Trastuzumab", "start": 0, "end": 11, "reason": "Spelling.",
            "protected": False, "requires_audio_confirmation": True}],
        "rejected_proposals": [], "clinical_validation": "not_established",
        "multilingual_validation": "not_established", "requires_human_review": True,
        "requires_clinician_final_review": True, "clinician_final_review": "pending"}

    async def exercise():
        session = SessionState(app)
        tracker = session[event["inputs"][4]]
        args = [source, "en", "qwen3.5:4b"]
        request = demo.start_timed_request("corrections", args,
            TimingHistory(tmp_path / "times.json"), tracker, event_order=1)
        job = list(demo.run_timed_request(lambda *_: ("preview", report, report, []),
                                         args, request, TimingHistory(tmp_path / "times.json"),
                                         tracker))[-1][0]
        demo.issue_proposals(job, args, tracker)
        # Choices reach the browser through JS; the original server component
        # still has no choices. Only the canonical issued report grants authority.
        assert app.blocks[event["inputs"][2]].choices == []
        response = await app.process_api(event["id"],
            [source, None, selection, confirmed, None, None, "en", "qwen3.5:4b"], state=session)
        frame = response["data"][0]
        if hasattr(frame, "model_dump"):
            frame = frame.model_dump()
        if success:
            assert frame[0] == "Trastuzumab 5 mg."
            assert frame[1]["accepted_ids"] == ["e0"]
            assert tracker["reviewed"] == frame[1]
        else:
            assert frame[0] == ""
            assert frame[1]["status"] == "failed"
            assert not tracker.get("reviewed")

    asyncio.run(exercise())
