# Run Oncology Scribe

Research prototype for fictional or explicitly authorized non-patient recordings.
The browser interface is English. Transcription supports EN/DE/FR/IT; Swiss German
uses Whisper's German token and requires appropriate references and human review.

## Install

Use Python 3.12 and the free [uv tool](https://docs.astral.sh/uv/getting-started/installation/).
Extract or clone into a normal local folder, preferably outside a sync service.
Open a terminal in the folder containing `pyproject.toml`:

```powershell
uv sync --locked --python 3.12 --extra asr --extra ui --extra dev --link-mode copy
uv run --no-sync clinical-asr doctor
```

The lockfile pins Python packages. Installation requires internet; no paid API
key is used. Windows is the tested platform. Other platforms need verification.

## Prepare ASR Explicitly

Model weights are not included. Check the model terms and your disk space before
running a download. Reuse a compatible existing cache rather than downloading
the same weights again. No model is downloaded when the UI opens or when a
missing-model transcription fails.

For Whisper Turbo, approximately 1.62 GB of converted weights:

```powershell
uv run --no-sync hf download mobiuslabsgmbh/faster-whisper-large-v3-turbo --revision 0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf --cache-dir models
```

The upstream repository now redirects to `dropbox-dash`; the pinned revision and
the cache namespace above match this adapter. Keep `--cache-dir models`
and run the application from this same project folder. A normal global HF cache
or a standalone `--local-dir` is not the documented layout.

Optional Large-v3, approximately 3.09 GB:

```powershell
uv run --no-sync hf download Systran/faster-whisper-large-v3 --revision edaa852ec7e145841d8ffdb056a99866b5f0a478 --cache-dir models
```

Select only a prepared model. Turbo and Large-v3 are the documented
configurations; Small is experimental and also requires a compatible cache.
GPU inference needs working CUDA libraries; start with CPU.

## Open the Workspace

```powershell
uv run --no-sync clinical-asr demo --port 7860
```

Keep the terminal open, then open http://127.0.0.1:7860/ in Chrome. Choose a
different free port if needed. Ctrl+C stops the server. The URL is local to each
computer; GitHub Pages cannot run this Python backend.

In Consultation, load a fictional dialogue or upload eligible audio, select its
language and transcribe. The main transcript is audio-linked: click a word to
hear original audio with two seconds of context on either side. Select several
words and use **Play selection** for a phrase. Replay excerpt repeats it;
Continue audio uses the rest of the original recording. Raw text copy retains
the untouched ASR text separately. Native selection and keyboard activation work.

Timing is estimated by ASR, not verified word alignment or clinical certainty.
Untimed words and ambiguous/over-30-second phrases are rejected. Changing the
recording, transcript identity or cue configuration invalidates old playback.
Corrected text and the final note are not independently aligned to the recording.

## Optional LLMs

Install [Ollama](https://ollama.com/download) separately. No paid service is used.
Medical Review uses installed local models; extractive notes work without Ollama.
After checking model terms/storage, colleagues may explicitly install:

```powershell
ollama pull qwen3.5:4b
ollama pull llama3.1:latest
ollama list
```

Keep Ollama on `127.0.0.1:11434`. Tags can change; a new download is not guaranteed
to match the evaluated digest. MedGemma remains an optional comparison with
unresolved exact upstream provenance. No LLM edit is clinical approval. Review
selected eligible changes against audio before confirming them.

## Edit and Test

```powershell
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests tools
node --test tests/task_clock.test.cjs tests/word_audio.test.cjs tests/workflow_lock.test.cjs
```

Node is needed only for JS tests. Restart the server and refresh the page after
source changes. Do not edit `.venv` or remove guards to improve model scores.
Keep recordings, generated exports, environments, caches and credentials out of Git.
