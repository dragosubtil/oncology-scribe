from pathlib import Path

import pytest

from clinical_asr import demo


@pytest.mark.parametrize('model,revision', [
    ('turbo', '0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf'),
    ('large-v3', 'edaa852ec7e145841d8ffdb056a99866b5f0a478'),
])
def test_demo_loads_explicitly_prepared_revision_without_main_cache_ref(
        tmp_path, monkeypatch, model, revision):
    from faster_whisper import utils

    snapshot = tmp_path / revision
    snapshot.mkdir()
    calls = []

    def cached_download(name, **options):
        calls.append((name, options))
        assert options['local_files_only'] is True
        assert options['revision'] == revision, 'Prepared pinned cache must not require refs/main'
        return str(snapshot)

    monkeypatch.setattr(utils, 'download_model', cached_download)
    monkeypatch.setattr('faster_whisper.WhisperModel', lambda *args, **kwargs: object())
    demo._engine.cache_clear()
    try:
        engine = demo._engine(model, 'cpu', False)
        engine._load()
        assert engine._resolved_revision == Path(snapshot).name
        assert len(calls) == 1
    finally:
        demo._engine.cache_clear()
