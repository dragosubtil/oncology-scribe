from clinical_asr.demo import build_app


def test_dialogue_loads_audio_and_language_without_inserting_reference_into_asr():
    app = build_app()
    components = app.config["components"]
    selector = next((c for c in components
                     if c["props"].get("label") == "Fictional consultation"), None)
    assert selector is not None, "Fictional audio selector is missing"
    assert len(selector["props"]["choices"]) == 8
    button = next(c["id"] for c in components
                  if c["type"] == "button" and c["props"].get("value") == "Load dialogue")
    event = next(d for d in app.config["dependencies"]
                 if any(t[0] == button and t[1] == "click" for t in d["targets"]))
    labels = {c["props"].get("label") for c in components if c["id"] in event["outputs"]}
    assert {"Consultation audio", "Spoken language", "Dialogue script",
            "Synthetic speech reference"} <= labels
    assert not {"Raw transcript", "Transcript for medical review", "Transcript for note"} & labels


def test_new_upload_or_clear_removes_old_demo_reference():
    app = build_app()
    components = app.config["components"]
    audio = next(c["id"] for c in components if c["props"].get("label") == "Consultation audio")
    refs = {c["id"] for c in components
            if c["props"].get("label") in {"Dialogue script", "Synthetic speech reference"}}
    assert len(refs) == 2, "Separate demo references are missing"
    for trigger in ["upload", "clear"]:
        assert any(any(t[0] == audio and t[1] == trigger for t in d["targets"])
                   and refs <= set(d["outputs"]) for d in app.config["dependencies"])
