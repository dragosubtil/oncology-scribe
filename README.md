# Oncology Scribe

A local multilingual research prototype for oncology consultation documentation.
The workflow retains raw ASR text, links aligned words and selections to original
audio, proposes limited terminology edits, and prepares a source-grounded note
for human review.

**Not validated for clinical use.** ASR, medication spelling, negation,
attribution and note categorization can be wrong. Human review is mandatory.

## Run Locally

Follow [the quickstart](docs/QUICKSTART_EN.md). Requires Python 3.12; CPU is the
tested inference path. Optional local LLMs use Ollama. No paid API key is needed.
Models must be acquired explicitly after checking their terms and disk space;
weights are not bundled and the UI does not download them automatically.

```powershell
uv sync --locked --python 3.12 --extra asr --extra ui --extra dev --link-mode copy
uv run --no-sync clinical-asr demo --port 7860
```

Open http://127.0.0.1:7860/ in Chrome. Transcription needs the pinned model cache
described in the quickstart. Without an LLM, extractive notes remain available.

## Workspace

- Consultation: upload eligible audio or load one of eight fictional TTS dialogues.
- Audio review: click an aligned word, or select words and use Play selection.
  The excerpt uses the original recording, with context and estimated timestamps.
- Medical Review: inspect spelling proposals, review eligible selections against
  audio, and retain the original separately. Medication substitutions are guarded.
- Note Draft: positional SOAP categories or extractive points, with source quotes.
- Evaluation: compare reference and hypothesis text using WER/CER and lexical metrics.

EN/DE/FR/IT are implemented. Swiss German is exploratory through Whisper's German
token; native dialect output and clinical accuracy are not established.

## Development

Main modules: `demo.py` (UI/workflow), `transcribe.py` (ASR), `word_audio.py`
(original-audio excerpts), `corrections.py` (proposals), `notes.py` (source-grounded
categories), and `metrics.py` (text metrics). Assets are under `src/clinical_asr/assets`.

```powershell
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests tools
node --test tests/task_clock.test.cjs tests/word_audio.test.cjs tests/workflow_lock.test.cjs
```

Use branches and reviewed changes. Add regression tests for failures; do not
weaken source identity, audio hashes or clinical review requirements for a pass.
See [verification scope](docs/SHARE_VERIFICATION_EN.md) and
[third-party notices](docs/THIRD_PARTY_NOTICES.md).

## Sharing

Prepared for private team collaboration, not public clinical hosting. No
open-source license has been selected for the project's code or examples.
Third-party components retain their own license terms. See the
[collaboration guide](docs/GITHUB_COLLABORATION_EN.md) for working together.

No patient recordings, external datasets, model weights, environments,
credentials or raw evaluation captures are included. Only the
explicitly catalogued synthetic fictional WAVs are bundled. Keep the server on
loopback; it has no authentication, EHR integration or production deployment.
