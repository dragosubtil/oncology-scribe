import io
import json
import urllib.request

import pytest

from clinical_asr import demo, notes


@pytest.fixture
def wire(monkeypatch):
    state = {'label': 'assessment'}

    class Opener:
        def open(self, request, **options):
            if request.full_url.endswith('/api/show'):
                return io.BytesIO(json.dumps({'details': {'format': 'gguf'},
                    'model_info': {'general.architecture': 'llama'}, 'template': ''}).encode())
            count = len(json.loads(json.loads(request.data)['messages'][1]['content'])['evidence_spans'])
            return io.BytesIO(json.dumps({'done': True, 'done_reason': 'stop',
                'message': {'content': json.dumps({'labels': [state['label']] * count})}}).encode())

    monkeypatch.setattr(urllib.request, 'build_opener', lambda *args: Opener())
    return state


@pytest.mark.parametrize('source', [
    'Clinician: Has the tumour progressed?',
    'Arzt: Ist der Tumor fortgeschritten?',
    'Medecin: La tumeur a-t-elle progresse?',
    'Medico: Il tumore e progredito?',
])
@pytest.mark.parametrize('label', ['assessment', 'subjective'])
def test_explicit_question_cannot_be_presented_as_a_finding(wire, source, label):
    wire['label'] = label
    report = notes.ollama_positional_note(source)
    assert report['sections']['unassigned'] == [{'quote': source, 'start': 0, 'end': len(source)}]
    assert not report['sections'][label]
    assert report['raw_transcript'] == source
    assert report['uncovered_source'] == []
    assert report['category_overrides'] == [{'start': 0, 'end': len(source),
        'model_category': label, 'category': 'unassigned', 'reason': 'explicit_question'}]
    assert report['clinician_final_review'] == 'pending'


def test_nonquestion_does_not_reclassify_an_explicit_assessment(wire):
    source = 'Clinician: My assessment remains uncertain.'
    report = notes.ollama_positional_note(source)
    assert report['sections']['assessment'][0]['quote'] == source
    assert report['category_overrides'] == []


def test_already_unassigned_question_is_not_counted_as_an_override(wire):
    wire['label'] = 'unassigned'
    report = notes.ollama_positional_note('Clinician: Any new pain?')
    assert report['sections']['unassigned']
    assert report['category_overrides'] == []


def test_question_in_middle_of_line_keeps_answer_separate(wire):
    source = 'Clinician: Any new pain? Patient: No new pain. No fever.'
    report = notes.ollama_positional_note(source)
    assert report['sections']['unassigned'] == [
        {'quote': 'Clinician: Any new pain?', 'start': 0, 'end': 24}]
    assert [p['quote'] for p in report['sections']['assessment']] == [
        'Patient: No new pain.', 'No fever.']
    assert report['uncovered_source'] == []


def test_unicode_question_preserves_exact_offsets(wire):
    source = 'M\u00e9decin: La fi\u00e8vre persiste?\nPatient: Non.'
    report = notes.ollama_positional_note(source)
    point = report['sections']['unassigned'][0]
    assert point == {'quote': 'M\u00e9decin: La fi\u00e8vre persiste?', 'start': 0, 'end': 28}
    assert source[point['start']:point['end']] == point['quote']
    assert report['sections']['assessment'][0]['quote'] == 'Patient: Non.'


def test_embedded_quoted_question_does_not_move_the_surrounding_statement(wire):
    source = 'Clinician: She asked "Any pain?" but now denies pain.'
    report = notes.ollama_positional_note(source)
    assert report['sections']['assessment'] == [{'quote': source, 'start': 0, 'end': len(source)}]
    assert report['category_overrides'] == []
    assert report['classification_warnings'] == [
        {'start': 0, 'end': len(source), 'reason': 'embedded_question_requires_review'}]


def test_embedded_question_warning_is_visible_in_note_preview(wire):
    source = 'Clinician: She asked "Any pain?" but now denies pain.'
    markup, report = demo.note_preview(source, "positional", "llama3.1:latest")
    assert "Attribution review pending" in markup
    assert report['classification_warnings']
