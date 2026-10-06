# Verification Scope

Regression checks repeated on Windows / Python 3.12 on 6 October 2026.
Browser and model-cache checks below were performed on 5 October 2026.

- Complete suite: 766 passed, 10 skipped in the separately installed source
  copy, including the final playback event-chain fix. Skips concern optional
  tooling and Windows symlink support; these are not successful tests.
- The suite includes playback regressions for ambiguous selections, untimed
  words, stale transcripts and competing requests. It completed without failures;
  13 warnings came from deliberately exercised rejection paths.
- JavaScript: 19 passed; Ruff checks passed on 6 October 2026.
- Clean dependency installation used the locked dependencies and an existing
  offline package cache, outside OneDrive. No model acquisition was performed.
- The clean installation loaded the pinned Turbo and Large-v3 caches; Turbo
  transcribed an eight-second synthetic fixture with word timestamps offline.
- In-browser checks confirmed word playback, phrase selection, Replay excerpt
  and Continue audio. Desktop and 320px layouts were inspected. A real phone,
  independent colleague installation and all browser autoplay policies remain
  unverified.

Review identified untimed selection boundaries, retained audio after rejected
selection and a stale event restarting a retained clip. Regression tests and
guarded playback publication address these cases. Timings remain approximate;
playback does not prove a medication name or clinical interpretation is correct.

Only eight explicitly catalogued fictional TTS WAVs are included. Their audio
and source hashes were checked. Source selection and credential-pattern checks
were performed; these do not constitute a comprehensive security audit.

No new clinical benchmark, weight training, patient-data processing or clinical
validation was performed in this UI/sharing phase. Review the third-party
notices before redistribution. Keep the server on localhost.
