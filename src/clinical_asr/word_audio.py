"""Replay original PCM around aligned words or a bounded phrase, never corrected text."""

import math
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

from .schema import file_sha256


def valid_times(start, end):
    return all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
               for t in (start, end)) and 0 <= start < end and end - start <= 30


def invalidate_playback(tracker):
    tracker['generation'] = tracker.get('generation', 0) + 1


def reserve_playback_action(clip, result, tracker):
    if (not clip or not result or clip.get('playback_generation') != tracker['generation']
            or clip.get('review_revision') != result.get('review_revision')):
        raise ValueError('This excerpt belongs to an earlier transcript.')
    tracker['action_generation'] = tracker.get('action_generation', 0) + 1
    return clip | {'playback_action': tracker['action_generation']}


def _aligned_selection(result, start, end):
    text = result['text']
    if (any(not isinstance(i, int) or isinstance(i, bool) for i in (start, end))
            or not 0 <= start < end <= len(text)):
        raise ValueError('Invalid text selection')
    words = result.get('words', [])
    first = [i for i, w in enumerate(words) if w.get('char_start') == start]
    last = [i for i, w in enumerate(words) if w.get('char_end') == end]
    if len(first) != 1 or len(last) != 1 or first[0] > last[0]:
        raise ValueError('The selection must contain complete aligned words.')
    selected = words[first[0]:last[0] + 1]
    cursor, previous_time = start, 0
    for word in selected:
        a, b = word.get('char_start'), word.get('char_end')
        if (any(not isinstance(i, int) or isinstance(i, bool) for i in (a, b))
                or not cursor <= a < b <= end or text[a:b] != word.get('text')
                or text[cursor:a].strip()):
            raise ValueError('Selection contains ambiguous word alignment.')
        t0, t1 = word.get('start'), word.get('end')
        if not valid_times(t0, t1) or t0 < previous_time:
            raise ValueError('Word-level timing is unavailable or ambiguous for this selection.')
        cursor, previous_time = b, t1
    if cursor != end or not valid_times(selected[0]['start'], selected[-1]['end']):
        raise ValueError('Select an aligned phrase of at most 30 seconds.')
    return [{key: w[key] for key in ('text', 'char_start', 'char_end', 'start', 'end')}
            for w in selected]


def word_audio_excerpt(result, audio, start, end, source_sha256, revision):
    if (not result or result.get('status') != 'transcribed' or not audio
            or source_sha256 != result.get('source_audio_sha256')
            or not revision or revision != result.get('review_revision')):
        raise ValueError('The audio or transcript changed. Transcribe the current audio again.')
    path = Path(audio)
    if not path.is_file() or file_sha256(path) != source_sha256:
        raise ValueError('The audio changed. Transcribe it again.')
    alignments = _aligned_selection(result, start, end)
    first, last = alignments[0]['start'], alignments[-1]['end']
    with sf.SoundFile(path) as stream:
        duration = stream.frames / stream.samplerate
        if last > duration:
            raise ValueError('Word timing lies outside the audio.')
        frame_start = max(0, math.floor(first * stream.samplerate) - 2 * stream.samplerate)
        frame_end = min(stream.frames, math.ceil(last * stream.samplerate) + 2 * stream.samplerate)
        stream.seek(frame_start)
        samples = stream.read(frame_end - frame_start, dtype='float32')
        rate = stream.samplerate
    if (not len(samples) or not np.isfinite(samples).all()
            or np.max(np.abs(samples)) > 1):
        raise ValueError('The selected audio excerpt is empty or invalid.')
    if file_sha256(path) != source_sha256:
        raise ValueError('Audio changed while preparing the excerpt.')
    # Fixed PCM scale prevents Gradio from peak-normalizing quiet excerpts.
    playback = np.clip(np.rint(samples.astype(np.float64) * 32768), -32768, 32767).astype(np.int16)
    return dict(audio=(rate, playback), start=frame_start / rate, end=frame_end / rate,
                review_revision=revision,
                word=result['text'][start:end], char_start=start, char_end=end,
                alignment=alignments[0], alignments=alignments,
                source_audio_sha256=source_sha256, source_text=result['text'])


def require_current_excerpt(clip, result, audio, tracker=None):
    if (not clip or not result or result.get('status') != 'transcribed'
            or result.get('text') != clip.get('source_text')
            or result.get('review_revision') != clip.get('review_revision')
            or (tracker is not None and clip.get('playback_generation') != tracker['generation'])
            or (tracker is not None and clip.get('playback_action') is not None
                and clip['playback_action'] != tracker.get('action_generation'))):
        raise ValueError('This excerpt belongs to an earlier transcript.')
    if (not audio or not Path(audio).is_file()
            or result.get('source_audio_sha256') != clip.get('source_audio_sha256')
            or file_sha256(Path(audio)) != clip.get('source_audio_sha256')):
        raise ValueError('This excerpt belongs to an earlier audio file.')
    if _aligned_selection(result, clip['char_start'], clip['char_end']) != clip['alignments']:
        raise ValueError('The word alignment changed. Select the current word again.')
    if tracker is not None and (clip.get('playback_generation') != tracker['generation']
                               or (clip.get('playback_action') is not None
                                   and clip['playback_action'] != tracker.get('action_generation'))):
        raise ValueError('Playback was reset while validating this excerpt.')


@contextmanager
def continuation_excerpt(clip, result, audio, tracker):
    """Stream remaining original PCM into a temporary WAV; never depend on player seeking."""
    require_current_excerpt(clip, result, audio, tracker)
    with tempfile.TemporaryDirectory(prefix='oncology-word-audio-') as temporary:
        output = Path(temporary) / 'continuation.wav'
        with sf.SoundFile(audio) as source:
            first = math.ceil(clip['end'] * source.samplerate)
            if first >= source.frames:
                raise ValueError('The selected excerpt already reaches the end of the audio.')
            source.seek(first)
            with sf.SoundFile(output, mode='w', samplerate=source.samplerate,
                              channels=source.channels, subtype='PCM_16') as destination:
                while source.tell() < source.frames:
                    samples = source.read(65536, dtype='float32', always_2d=True)
                    if (not len(samples) or not np.isfinite(samples).all()
                            or np.max(np.abs(samples)) > 1):
                        raise ValueError('The remaining source audio is invalid.')
                    pcm = np.clip(np.rint(samples.astype(np.float64) * 32768),
                                  -32768, 32767).astype(np.int16)
                    destination.write(pcm)
        require_current_excerpt(clip, result, audio, tracker)
        yield output

