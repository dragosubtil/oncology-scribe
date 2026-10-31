# Dependency Policy

Use the locked environment from the quickstart:

```powershell
uv sync --locked --python 3.12 --extra asr --extra ui --extra dev --link-mode copy
```

Keep `pyproject.toml` and `uv.lock` together. Do not mix pip or Conda upgrades into
this environment. Model weights and Ollama are separate from Python dependencies.

## Security Floors

The resolver policy excludes the affected versions identified during review:
Brotli below 1.2.1, certifi below 2026.10.18, filelock below 4.2.4,
OpenTelemetry API below 1.45.1 and Pillow below 13.0.0. The lockfile records the
actual tested versions and artifact hashes. The dependency regression tests
check these floors; they do not detect newly disclosed vulnerabilities.

## Temporary Pillow Compatibility Rule

Stable Gradio 6.31.0 still declares `pillow<13`. Pillow 13 contains security fixes,
and upstream Gradio has widened its bound on the next-major development branch.
This project keeps stable Gradio 6 and uses uv's explicit
`override-dependencies` rule for `pillow>=13.0.0,<14`.

This is a tested project-level compatibility exception, not a claim that Gradio
6 officially supports every Pillow 13 feature. A generic `pip check` will report
the upstream metadata mismatch. Do not downgrade Pillow to silence that warning.
The supported installation path is the locked uv command above.

Validation covers the application's audio workflow, Gradio UI, and RGB PNG/WebP
image-cache roundtrips. General image editing and arbitrary Pillow plugins are
not application features or part of this compatibility claim. Remove the
override when a stable Gradio release permits Pillow 13 and passes the same tests.

References: [Gradio's compatibility change](https://github.com/gradio-app/gradio/pull/14005),
[uv dependency overrides](https://docs.astral.sh/uv/concepts/resolution/#dependency-overrides).

## Updating

Make targeted changes in a branch, regenerate the lockfile, create a fresh
environment and run the complete Python/JavaScript suites and browser workflow.
Recheck installed and locked versions against current advisories. A clean
advisory query is not a comprehensive security audit or clinical validation.
