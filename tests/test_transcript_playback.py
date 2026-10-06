import copy

import numpy as np
import pytest
import soundfile as sf

from clinical_asr import demo
from clinical_asr.schema import file_sha256
from clinical_asr.word_audio import require_current_excerpt


@pytest.fixture
def transcript(tmp_path):
    audio = tmp_path / 'source.wav'
    sf.write(audio, np.linspace(-0.1, 0.1, 12000), 1000, subtype='FLOAT')
    return {
        'status': 'transcribed', 'text': 'No new pain.', 'spoken_language': 'en',
        'source_audio_sha256': file_sha256(audio), 'review_revision': 'original',
        'review_spans': [],
        'words': [
            dict(text='No', char_start=0, char_end=2, start=2.0, end=2.4, model_score=0.9),
            dict(text='new', char_start=3, char_end=6, start=2.4, end=2.8, model_score=0.9),
            dict(text='pain.', char_start=7, char_end=12, start=2.8, end=3.5, model_score=0.9),
        ],
    }, str(audio)


def excerpt(transcript, start, end):
    result, audio = transcript
    try:
        return demo.word_audio_excerpt(result, audio, start, end,
                                       result['source_audio_sha256'], result['review_revision'])
    except ValueError as exc:
        pytest.fail(f'A current aligned selection must be playable: {exc}')


def test_high_confidence_words_are_also_audio_linked(transcript):
    markup, _, _, _ = demo.update_highlights(transcript[0], 0.6)
    assert markup.count('class="word-audio-cue') == 3
    assert 'data-word-start="3" data-word-end="6"' in markup
    assert 'aria-label="Listen to new"' in markup


def test_nonhighlighted_word_uses_its_original_timestamps(transcript):
    clip = excerpt(transcript, 3, 6)
    assert clip['word'] == 'new'
    assert (clip['start'], clip['end']) == (0.4, 4.8)
    original, _ = sf.read(transcript[1], start=400, stop=4800, dtype='float32')
    np.testing.assert_allclose(clip['audio'][1] / 32768, original, atol=1 / 32768)


def test_phrase_replays_all_selected_words_at_original_gain(transcript):
    before = copy.deepcopy(transcript[0])
    clip = excerpt(transcript, 0, 12)
    assert clip['word'] == 'No new pain.'
    assert (clip['start'], clip['end']) == (0, 5.5)
    assert len(clip['alignments']) == 3
    original, _ = sf.read(transcript[1], stop=5500, dtype='float32')
    np.testing.assert_allclose(clip['audio'][1] / 32768, original, atol=1 / 32768)
    assert transcript[0] == before
    require_current_excerpt(clip, *transcript)


@pytest.mark.parametrize('change', ['missing_middle_time', 'backwards_time', 'duplicate', 'unaligned_gap'])
def test_phrase_cannot_bridge_ambiguous_alignment(transcript, change):
    result, audio = transcript
    if change == 'missing_middle_time':
        result['words'][1]['start'] = None
    elif change == 'backwards_time':
        result['words'][1]['start'] = 1.0
    elif change == 'duplicate':
        result['words'].insert(1, copy.deepcopy(result['words'][0]))
    else:
        result['words'].pop(1)
    with pytest.raises(ValueError):
        demo.word_audio_excerpt(result, audio, 0, 12,
                                result['source_audio_sha256'], result['review_revision'])


def test_phrase_is_invalidated_when_any_middle_alignment_changes(transcript):
    clip = excerpt(transcript, 0, 12)
    transcript[0]['words'][1]['end'] = 2.7
    with pytest.raises(ValueError, match='alignment'):
        require_current_excerpt(clip, *transcript)


def test_untimed_word_is_not_a_clickable_audio_target(transcript):
    transcript[0]['words'][1]['start'] = None
    markup, _, _, _ = demo.update_highlights(transcript[0], 0.6)
    assert markup.count('class="word-audio-cue') == 2
    assert 'data-word-start="3"' not in markup
    assert 'new' in markup


def test_phrase_rejects_partial_word_and_oversized_audio_window(transcript):
    result, audio = transcript
    for start, end in [(1, 12), (0, 11)]:
        with pytest.raises(ValueError):
            demo.word_audio_excerpt(result, audio, start, end,
                                    result['source_audio_sha256'], result['review_revision'])
    result['words'][-1]['end'] = 34
    with pytest.raises(ValueError):
        demo.word_audio_excerpt(result, audio, 0, 12,
                                result['source_audio_sha256'], result['review_revision'])


@pytest.mark.parametrize('missing', [0, 2])
def test_untimed_selection_boundaries_remain_identifiable(transcript, missing):
    transcript[0]['words'][missing]['start'] = None
    markup, _, _, _ = demo.update_highlights(transcript[0], 0.6)
    token = transcript[0]['words'][missing]['text']
    assert 'class="untimed-audio-word"' in markup
    assert f'>{token}</mark>' in markup or f'>{token}</span>' in markup


@pytest.mark.parametrize('newer', [False, True])
def test_rejected_selection_clears_only_its_own_invalidated_player(transcript, monkeypatch, newer):
    import gradio as gr

    app = demo.build_app()
    prepare = next(f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == 'prepare_word')
    tracker = {'generation': 0}

    def reject(*args):
        if newer:
            tracker['generation'] += 1
        raise ValueError('Ambiguous selection')

    monkeypatch.setattr(demo, 'word_audio_excerpt', reject)
    event = gr.EventData(None, dict(start=0, end=12, revision='original',
                                   source_sha256=transcript[0]['source_audio_sha256']))
    output = prepare(*transcript, tracker, event)
    assert output == ((gr.skip(),) * 4 if newer else (None, None, '', None))


def test_word_publication_is_not_chained_to_retained_state():
    app = demo.build_app()
    event = next(d for d in app.config['dependencies'] if d['api_name'] == 'listen_word')
    assert not any(d.get('trigger_after') == event['id'] for d in app.config['dependencies'])
