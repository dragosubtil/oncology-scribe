import copy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from clinical_asr import demo, highlights
from clinical_asr.schema import file_sha256
from clinical_asr.word_audio import require_current_excerpt


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'audio.wav'
    sf.write(path, np.linspace(-0.2, 0.2, 10000), 1000, subtype='FLOAT')
    words = [dict(text='pain', char_start=0, char_end=4, start=1.0, end=1.5, model_score=0.2),
             dict(text='pain', char_start=5, char_end=9, start=8.0, end=8.5, model_score=0.2)]
    result = dict(status='transcribed', text='pain pain', words=words, spoken_language='en',
                  review_revision='revision-a',
                  source_audio_sha256=file_sha256(path))
    result['review_spans'] = highlights.review_spans(result['text'], words, 'en')
    return result, str(path)


def select(source, start=5, end=9):
    result, path = source
    assert hasattr(demo, 'word_audio_excerpt'), 'Word-linked audio is not implemented'
    return demo.word_audio_excerpt(result, path, start, end, result['source_audio_sha256'],
                                   result['review_revision'])


def test_repeated_word_uses_exact_offsets_and_original_pcm(source):
    original = copy.deepcopy(source[0])
    clip = select(source)
    assert (clip['start'], clip['end']) == (6.0, 10.0)
    assert clip['word'] == 'pain'
    assert clip['audio'][0] == 1000
    samples, _ = sf.read(source[1], dtype='float32', start=6000, stop=10000)
    np.testing.assert_allclose(clip['audio'][1].astype(float) / 32768, samples, atol=1 / 32768)
    assert source[0] == original


def test_context_clamped_at_start(source):
    clip = select(source, 0, 4)
    assert (clip['start'], clip['end']) == (0, 3.5)


@pytest.mark.parametrize('start,end', [(True,4), (0,False), (-1,4), (0,100), (5,8)])
def test_invalid_character_selection_rejected(source, start, end):
    with pytest.raises(ValueError):
        select(source, start, end)


@pytest.mark.parametrize('start,end', [(None,8.5), (float('nan'),8.5), (True,8.5),
                                     (-1,8.5), (8,8), (9,8), (8,11)])
def test_missing_or_invalid_time_cannot_play(source, start, end):
    source[0]['words'][1].update(start=start, end=end)
    with pytest.raises(ValueError):
        select(source)


def test_stale_source_or_cleared_audio_rejected(source):
    result, path = source
    assert hasattr(demo, 'word_audio_excerpt')
    with pytest.raises(ValueError):
        demo.word_audio_excerpt(result, None, 5, 9, result['source_audio_sha256'], result['review_revision'])
    with pytest.raises(ValueError):
        demo.word_audio_excerpt(result, path, 5, 9, 'stale', result['review_revision'])
    sf.write(path, np.zeros(10000), 1000)
    with pytest.raises(ValueError):
        select(source)


def test_unhighlighted_aligned_word_can_be_requested(source):
    source[0]['review_spans'] = []
    assert select(source)['word'] == 'pain'


def test_highlight_time_and_source_identity_are_exposed_safely(source):
    result, _ = source
    rendered = highlights.render_highlight_html(result['text'], result['review_spans'],
                                                 result['source_audio_sha256'], result['review_revision'])
    assert 'data-word-start="5"' in rendered
    assert 'data-word-end="9"' in rendered
    assert result['source_audio_sha256'] in rendered
    assert 'aria-label="Listen to pain"' in rendered
    assert 'data-review-revision="revision-a"' in rendered


def test_untimed_word_remains_plain_review_cue():
    words = [dict(text='<x>', char_start=0, char_end=3, model_score=0.2)]
    spans = highlights.review_spans('<x>', words, 'en')
    rendered = highlights.render_highlight_html('<x>', spans)
    assert '&lt;x&gt;' in rendered
    assert 'data-word-start' not in rendered


def test_old_click_cannot_select_a_new_word_at_same_offsets(source):
    result, path = source
    result['text'] = 'pain rash'
    result['words'][1]['text'] = 'rash'
    result['review_revision'] = 'revision-b'
    result['review_spans'] = highlights.review_spans(result['text'], result['words'], 'en')
    with pytest.raises(ValueError):
        demo.word_audio_excerpt(result, path, 5, 9, result['source_audio_sha256'], 'revision-a')


def test_cutoff_revision_invalidates_pending_clip_even_if_cue_retained(source):
    clip = select(source)
    _, _, updated, _ = demo.update_highlights(source[0], 0.4)
    assert updated['review_spans']
    with pytest.raises(ValueError):
        require_current_excerpt(clip, updated, source[1])


def test_playback_reset_or_newer_click_invalidates_old_completion(source):
    from clinical_asr import word_audio

    assert hasattr(word_audio, 'invalidate_playback'), 'Playback generation guard not implemented'
    clip = select(source)
    clip['playback_generation'] = 1
    tracker = {'generation': 1}
    require_current_excerpt(clip, *source, tracker)
    word_audio.invalidate_playback(tracker)
    with pytest.raises(ValueError):
        require_current_excerpt(clip, *source, tracker)


def callbacks():
    app = demo.build_app()
    functions = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
    for cell in functions['prepare_word'].__closure__ or ():
        value = cell.cell_contents
        if callable(value) and getattr(value, '__name__', '') == 'publish_excerpt':
            functions['publish_excerpt'] = value
    return functions


def consume(value):
    import types

    return list(value) if isinstance(value, types.GeneratorType) else [value]


def test_reset_during_extraction_does_not_claim_new_generation(source, monkeypatch):
    import gradio as gr

    functions = callbacks()
    tracker = {'generation': 0}
    original = demo.word_audio_excerpt

    def interrupted(*args):
        clip = original(*args)
        functions['reset_word_audio'](tracker)
        return clip

    monkeypatch.setattr(demo, 'word_audio_excerpt', interrupted)
    result, path = source
    evt = gr.EventData(None, dict(start=5, end=9, source_sha256=result['source_audio_sha256'],
                                  revision=result['review_revision']))
    response = functions['prepare_word'](result, path, tracker, evt)
    assert all(value == gr.skip() for value in response)


@pytest.mark.parametrize('handler', ['publish_excerpt', 'replay_excerpt', 'continue_audio'])
def test_reset_during_publication_hash_cannot_restore_old_playback(source, monkeypatch, handler):
    import gradio as gr

    from clinical_asr import word_audio

    functions = callbacks()
    tracker = {'generation': 3}
    clip = select(source)
    clip['playback_generation'] = 3
    original = word_audio.file_sha256

    def interrupted(path):
        digest = original(path)
        functions['reset_word_audio'](tracker)
        return digest

    monkeypatch.setattr(word_audio, 'file_sha256', interrupted)
    responses = consume(functions[handler](clip, *source, tracker))
    assert all(value == gr.skip() for response in responses for value in response)


@pytest.mark.parametrize('channels,level', [(1,0.0001), (1,0.2), (2,0.0001), (2,0)])
def test_gradio_playback_preserves_gain(source, channels, level):
    from gradio.processing_utils import convert_to_16_bit_audio

    samples = np.full((10000, channels), level, dtype=np.float32)
    sf.write(source[1], samples, 1000, subtype='FLOAT')
    source[0]['source_audio_sha256'] = file_sha256(Path(source[1]))
    clip = select(source)
    rendered = convert_to_16_bit_audio(clip['audio'][1]).astype(float) / 32768
    np.testing.assert_allclose(rendered, level, atol=1 / 32768)


@pytest.mark.parametrize('change', ['text', 'timing', 'cue', 'failure'])
def test_late_clip_publication_rejects_new_transcript_state(source, change):
    clip = select(source)
    result, path = source
    require_current_excerpt(clip, result, path)
    if change == 'text':
        result['text'] = 'new transcript'
    elif change == 'timing':
        result['words'][1]['start'] = 7.5
    elif change == 'cue':
        result['review_spans'] = []
        result['review_revision'] = 'new-cutoff-revision'
    else:
        result['status'] = 'failed'
    with pytest.raises(ValueError):
        require_current_excerpt(clip, result, path)


def test_app_has_word_event_and_resets_excerpt_on_new_audio():
    app = demo.build_app()
    components = app.config['components']
    players = [c for c in components if c['props'].get('label') == 'Selected audio excerpt']
    assert len(players) == 1
    assert players[0]['props']['autoplay']
    event = next(d for d in app.config['dependencies'] if d['api_name'] == 'listen_word')
    assert event['targets'][0][1] == 'play_word'
    audio_id = next(c['id'] for c in components if c['props'].get('label') == 'Consultation audio')
    assert any(d['targets'] == [(audio_id, 'change')] and players[0]['id'] in d['outputs']
               for d in app.config['dependencies'])


def test_continue_uses_remaining_source_pcm_not_an_unreliable_player_seek(source):
    result, path = source
    clip = select(source, 0, 4)
    tracker = {'generation': 1}
    clip['playback_generation'] = 1
    update, label = consume(callbacks()['continue_audio'](clip, result, path, tracker))[-1]
    assert update['value'] != path
    samples, rate = sf.read(update['value'], dtype='float32')
    original, _ = sf.read(path, dtype='float32', start=3500)
    assert rate == 1000 and len(samples) == 6500
    np.testing.assert_allclose(samples, original, atol=1 / 32768)
    assert update['playback_position'] == 0
    assert '3.50 s' in label


def test_continue_at_audio_end_abstains_without_resetting_player(source):
    import gradio as gr

    clip = select(source)
    clip['playback_generation'] = 1
    responses = consume(callbacks()['continue_audio'](clip, *source, {'generation': 1}))
    assert all(value == gr.skip() for response in responses for value in response)

    app = demo.build_app()
    button = next(c['id'] for c in app.config['components']
                  if c['props'].get('value') == 'Continue audio')
    dependency = next(d for d in app.config['dependencies'] if d['targets'] == [(button, 'click')])
    assert app.fns[dependency['id']].fn.__name__ == 'continue_audio'


def test_continue_reset_during_cache_copy_cannot_publish_old_audio(source, monkeypatch):
    import gradio as gr

    functions = callbacks()
    tracker = {'generation': 1}
    clip = select(source, 0, 4)
    clip['playback_generation'] = 1
    original = gr.Audio.move_resource_to_block_cache

    def interrupted(player, resource):
        cached = original(player, resource)
        functions['reset_word_audio'](tracker)
        return cached

    monkeypatch.setattr(gr.Audio, 'move_resource_to_block_cache', interrupted)
    responses = consume(functions['continue_audio'](clip, *source, tracker))
    assert all(value == gr.skip() for response in responses for value in response)


def test_slow_continue_cannot_overwrite_a_newer_replay(source, monkeypatch):
    import gradio as gr

    functions = callbacks()
    tracker = {'generation': 1}
    clip = select(source, 0, 4)
    clip['playback_generation'] = 1
    original = gr.Audio.move_resource_to_block_cache
    replayed = []

    def interrupted(player, resource):
        cached = original(player, resource)
        replayed.extend(consume(functions['replay_excerpt'](clip, *source, tracker)))
        return cached

    monkeypatch.setattr(gr.Audio, 'move_resource_to_block_cache', interrupted)
    responses = consume(functions['continue_audio'](clip, *source, tracker))
    assert replayed[-1][0].get('value') is not None
    assert all(value == gr.skip() for response in responses for value in response)


@pytest.mark.parametrize('channels,level', [(1,0.0001), (2,0.2), (2,0)])
def test_continuation_preserves_multiblock_pcm_and_cleans_temporary_file(source, channels, level):
    from clinical_asr.word_audio import continuation_excerpt

    samples = np.full((150000, channels), level, dtype=np.float32)
    sf.write(source[1], samples, 1000, subtype='FLOAT')
    source[0]['source_audio_sha256'] = file_sha256(Path(source[1]))
    clip = select(source, 0, 4)
    clip['playback_generation'] = 1
    with continuation_excerpt(clip, *source, {'generation': 1}) as path:
        temporary = path.parent
        actual, rate = sf.read(path, dtype='float32', always_2d=True)
        assert rate == 1000 and actual.shape == (146500, channels)
        np.testing.assert_allclose(actual, level, atol=1 / 32768)
    assert not temporary.exists()


@pytest.mark.parametrize('bad', ['empty', 'nan', 'clipping'])
def test_invalid_tail_read_abstains_and_cleans_temporary_file(source, monkeypatch, bad):
    import tempfile

    from clinical_asr import word_audio

    clip = select(source, 0, 4)
    clip['playback_generation'] = 1
    directories = []
    original = tempfile.TemporaryDirectory

    def tracked(*args, **kwargs):
        directory = original(*args, **kwargs)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(word_audio.tempfile, 'TemporaryDirectory', tracked)
    values = {'empty': np.empty((0, 1)), 'nan': np.array([[np.nan]]),
              'clipping': np.array([[1.1]])}
    monkeypatch.setattr(sf.SoundFile, 'read', lambda *args, **kwargs: values[bad])
    with pytest.raises(ValueError):
        with word_audio.continuation_excerpt(clip, *source, {'generation': 1}):
            pytest.fail('Invalid tail must not be published')
    assert directories and not any(path.exists() for path in directories)


def test_cache_copy_failure_keeps_previous_player_and_cleans_temporary_file(source, monkeypatch):
    import gradio as gr

    functions = callbacks()
    clip = select(source, 0, 4)
    clip['playback_generation'] = 1
    paths = []

    def failed(player, resource):
        paths.append(Path(resource))
        raise OSError('Test-only cache failure')

    monkeypatch.setattr(gr.Audio, 'move_resource_to_block_cache', failed)
    responses = consume(functions['continue_audio'](clip, *source, {'generation': 1}))
    assert all(value == gr.skip() for response in responses for value in response)
    assert paths and not any(path.exists() for path in paths)


def test_continuation_preserves_nonuniform_stereo_pcm16_bit_for_bit(source):
    from clinical_asr.word_audio import continuation_excerpt

    left = ((np.arange(150000, dtype=np.int64) * 31) % 60001 - 30000).astype(np.int16)
    original = np.column_stack([left, -left])
    sf.write(source[1], original, 1000, subtype='PCM_16')
    source[0]['source_audio_sha256'] = file_sha256(Path(source[1]))
    clip = select(source, 0, 4)
    clip['playback_generation'] = 1
    with continuation_excerpt(clip, *source, {'generation': 1}) as path:
        actual, rate = sf.read(path, dtype='int16', always_2d=True)
        assert rate == 1000
        np.testing.assert_array_equal(actual, original[3500:])


def test_source_replacement_during_tail_streaming_rejects_and_cleans(source, monkeypatch):
    from clinical_asr import word_audio

    clip = select(source, 0, 4)
    clip['playback_generation'] = 1
    original = word_audio.require_current_excerpt
    calls = []

    def changed(*args):
        calls.append(True)
        if len(calls) == 2:
            sf.write(source[1], np.zeros(10000), 1000)
        return original(*args)

    monkeypatch.setattr(word_audio, 'require_current_excerpt', changed)
    with pytest.raises(ValueError):
        with word_audio.continuation_excerpt(clip, *source, {'generation': 1}):
            pytest.fail('Changed source must not be published')
