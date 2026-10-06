"""Exercise actual Gradio preprocessing, queue receipts and publication boundaries."""

import asyncio
import copy
import json

from gradio.state_holder import SessionState

from clinical_asr import demo


def test_serialized_queued_ticket_cannot_adopt_retry_or_replace_completed_job(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []

    def backend(text, method, model):
        calls.append(text)
        return "current draft", {"raw_transcript": text, "model": model}

    monkeypatch.setattr(demo, "note_preview", backend)
    app = demo.build_app()
    deps = {d["id"]: d for d in app.config["dependencies"]}
    pub = next(d for d in deps.values() if d["api_name"] == "publish_note")
    worker = deps[pub["trigger_after"]]
    start = deps[deps[worker["trigger_after"]]["trigger_after"]]
    args = ["same source", "extractive", "llama3.1:latest"]

    async def exercise():
        session = SessionState(app)

        async def begin():
            tracker = session[start["inputs"][0]]
            order = tracker.get("last_event_order", 0) + 1
            response = await app.process_api(start["id"], [None, order, None, *args], state=session)
            value = response["data"][0]
            return json.loads(value.model_dump_json()) if hasattr(value, "model_dump_json") else copy.deepcopy(value)

        async def run(ticket):
            payload = [copy.deepcopy(ticket), None, *args]
            response = await app.process_api(worker["id"], payload, state=session)
            while response["is_generating"]:
                response = await app.process_api(worker["id"], payload, state=session,
                                                 iterator=response["iterator"])
            return copy.deepcopy(session[worker["outputs"][0]])

        first = await begin()
        latest = await begin()
        assert first["request_id"] != latest["request_id"]
        assert await run(first) is None
        assert calls == []
        forged = copy.deepcopy(latest)
        forged["units"] = 9999
        assert await run(forged) is None
        assert calls == []
        current = await run(latest)
        assert calls == ["same source"]
        assert current["request"]["request_id"] == latest["request_id"]
        response = await app.process_api(pub["id"], [None, None, None, *args], state=session)
        assert response["data"][0] == "current draft"
        assert await run(first) == current
        assert await run(latest) == current
        assert calls == ["same source"]

    asyncio.run(exercise())
