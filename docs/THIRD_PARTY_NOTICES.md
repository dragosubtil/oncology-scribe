# Third-Party Notices

Project code/examples: no new open-source grant is made by this package.
Publication terms for the project's code and examples have not been selected.

## Bundled Assets

Lucide SVG icons retain ISC and applicable Feather MIT notices in
`src/clinical_asr/assets/icons/LICENSE`.

Eight WAVs and their generation scripts/catalogue are synthetic fiction authored
for this prototype, not patient recordings or independent clinical references.
They were generated using eSpeak NG 1.52.0. No eSpeak engine binary or voice-data
package is distributed here. The catalogue retains engine attribution,
GPL-3.0-or-later engine license information and generation hashes. Confirm the
project's own example redistribution terms before public release; this document
does not assert that generated audio inherits or is exempt from any engine terms.
[eSpeak NG license](https://github.com/espeak-ng/espeak-ng/blob/master/COPYING).

## Separately Installed Models

Weights are not bundled. Check the exact upstream artifact and its current terms
before acquiring it. A free local download is not unrestricted redistribution.

- Whisper Turbo conversion: [model card](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo),
  MIT-labelled upstream. The tested cache uses the former mobiuslabsgmbh namespace
  at revision `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`.
- Whisper Large-v3 conversion: [model card](https://huggingface.co/Systran/faster-whisper-large-v3),
  revision `edaa852ec7e145841d8ffdb056a99866b5f0a478`.
- Local Qwen/Llama/MedGemma tags are not proof of exact upstream provenance or
  licensing. Do not redistribute their weights from this project. Match installed
  digests and review upstream terms when reproducing a model comparison.

No paid APIs, compute services or automatic model pulls are part of the app.
Python dependencies are installed separately from the versions in `uv.lock`;
their own licenses remain applicable. Raw public datasets and lecture audio are
not distributed with this source package.
