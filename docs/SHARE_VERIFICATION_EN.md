# Verification Scope

Checked on Windows 11 / Python 3.12.13 on 30–31 October 2026.

## Automated Checks

- Complete Python suite: 799 passed, 10 skipped. The skips concern optional
  tooling and Windows symlink support; they are not successful tests.
- JavaScript: 31 passed. Ruff checks passed.
- The suite includes the reproduced publication races, quoted questions with
  European punctuation and spacing, source offsets, medication substitutions,
  negations, doses, stale audio and competing requests. Negative-path tests
  produced 18 expected warnings. Tests are not proof of clinical accuracy.
- Publisher tests pause real Gradio postprocessing while the source changes.
  Hidden JSON frames pass through a client source-order gate before display;
  server-owned reports remain authoritative for review and export. Identical
  source events retain their order, while retries and changed inputs advance it.
- Fresh installation used the locked uv environment outside OneDrive. No model
  weights were downloaded. Security-floor tests and RGB PNG/WebP image-cache
  roundtrips passed with Gradio 6.31.0 and Pillow 13.0.0.
- The known-advisory query returned no matches for 68 installed PyPI dependencies
  and 74 registry packages in the lockfile on 30 October. This is not a
  comprehensive security audit. See [the dependency policy](DEPENDENCIES_EN.md),
  including the explicit temporary Pillow metadata exception.

## Workflow Checks

Chrome completed the following workflow with a fictional English recording:
load dialogue, local Whisper Turbo transcription, aligned-word playback, phrase
selection, Replay excerpt, Continue audio, Medical Review, extractive note and
JSON export. Loading and elapsed time were visible. ETA correctly indicated
insufficient history before three comparable successful runs.

Desktop and 390px/320px layouts were inspected without horizontal overflow,
JavaScript errors or HTTP errors in the normal workflow. The export required
human review and retained the source without uncovered text. Source fidelity
does not prove that the source transcript is correct.

A short fictional note request also completed through the local Llama 3.1
positional-category path, with no omitted source and its explicit question kept
unassigned. This is an integration smoke test, not a model ranking or clinical
validation. Cold-start time and inference speed depend on the machine.

The local Qwen 3.5 4B path proposed the fictional spelling edit
`trastuzumap` to `trastuzumab`; selecting it produced the reviewed text in Chrome.
This was a UI integration exercise with simulated confirmation, not clinical
audio verification. Real API tests separately reject unknown proposal IDs and
missing confirmation. The multiselect picker accepts client-delivered choices;
the current server-issued report determines which edits are eligible.

One repeat request encountered an unavailable Ollama service. It displayed a
failed task and no proposed edits; local LLM availability is a prerequisite,
separate from the ASR and extractive paths.

## Remaining Limits

Only eight catalogued fictional TTS WAVs are bundled. Their audio and script
hashes were checked. They are not independent human references. ASR can still
misrecognize medication names and biomarkers; extractive notes retain such errors
until a person reviews them. Review cues and estimated timestamps are not certainty.

No new clinical benchmark, weight training or patient-data processing was
performed. Other languages and LLMs were not revalidated by this release. A real
phone, other operating systems, independent colleague installation and every
browser autoplay policy remain unverified. Keep the server on localhost and
review third-party terms before redistribution. Clinical approval remains pending.
