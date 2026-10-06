"""Research CLI with explicit errors, provenance and separate report artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import evaluate_predictions, paired_comparison
from .highlights import review_spans
from .notes import extractive_note, ollama_note, ollama_positional_note, validate_local_model
from .schema import file_sha256, load_manifest
from .transcribe import ASRConfig, Transcriber


def write_json(path: Path | str, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_predictions(path: Path | str) -> dict[str, str]:
    path = Path(path)
    results = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key, text = record["sample_id"], record["text"]
            if not isinstance(key, str) or not isinstance(text, str):
                raise ValueError()
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid prediction on line {number}") from exc
        if key in results:
            raise ValueError(f"Duplicate prediction ID: {key}")
        results[key] = text
    return results


def _asr_options(parser):
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--condition-on-previous-text", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--experimental-dialect", action="store_true")
    parser.add_argument("--initial-prompt-file", type=Path)


def _config(args) -> ASRConfig:
    prompt = args.initial_prompt_file.read_text(encoding="utf-8") if args.initial_prompt_file else None
    return ASRConfig(model=args.model, device=args.device, compute_type=args.compute_type,
                     cpu_threads=args.threads, beam_size=args.beam_size, vad_filter=not args.no_vad,
                     condition_on_previous_text=args.condition_on_previous_text,
                     initial_prompt=prompt, revision=args.revision, local_files_only=args.local_files_only,
                     allow_experimental_dialect=args.experimental_dialect)


def _doctor() -> dict:
    packages = {}
    for name in ["clinical-transcription-lab", "numpy", "scipy", "soundfile", "jiwer",
                 "pydantic", "faster-whisper", "ctranslate2", "gradio"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": packages, "paid_services_configured": False,
              "gpu_inference_verified": False}
    try:
        import ctranslate2
        result["cuda_device_count"] = ctranslate2.get_cuda_device_count()
        result["cpu_compute_types"] = sorted(ctranslate2.get_supported_compute_types("cpu"))
    except (ImportError, RuntimeError):
        result["cuda_device_count"] = None
    lock = Path("uv.lock")
    if lock.exists():
        result["dependency_lock_sha256"] = file_sha256(lock)
    return result


def _benchmark(args):
    all_samples = load_manifest(args.manifest, require_audio=True)
    samples = [s for s in all_samples if s.split == args.split]
    if not samples:
        raise ValueError("No samples in the requested split")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_sha256": file_sha256(args.manifest), "config": _config(args).model_dump(),
                "environment": _doctor(), "status": "running", "samples": len(samples)}
    write_json(output / "run.json", metadata)
    engine = Transcriber(_config(args))
    partial = output / "predictions.partial.jsonl"
    predictions = {}
    try:
        with partial.open("w", encoding="utf-8") as stream:
            for sample in samples:
                result = engine.transcribe_file(sample.audio_path, sample.language, sample.target_policy)
                result["sample_id"] = sample.sample_id
                result["source_audio_sha256"] = file_sha256(Path(sample.audio_path))
                result["review_spans"] = review_spans(result["text"], result["words"], sample.language)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                predictions[sample.sample_id] = result["text"]
        partial.replace(output / "predictions.jsonl")
        report = evaluate_predictions(samples, predictions, args.bootstrap, args.seed)
        write_json(output / "report.json", report)
        metadata["status"] = "complete"
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", metadata)
    except (Exception, KeyboardInterrupt) as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["completed_samples"] = len(predictions)
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", metadata)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local multilingual oncology transcription research")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    validation = commands.add_parser("validate")
    validation.add_argument("manifest", type=Path)
    validation.add_argument("--require-audio", action="store_true")
    transcription = commands.add_parser("transcribe")
    transcription.add_argument("audio", type=Path)
    transcription.add_argument("--language", choices=["en", "de", "gsw", "fr", "it", "pt"], required=True)
    transcription.add_argument("--target-policy", choices=["verbatim", "standard_de", "dialect"], default="verbatim")
    transcription.add_argument("--output", type=Path, required=True)
    _asr_options(transcription)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("manifest", type=Path)
    evaluation.add_argument("predictions", type=Path)
    evaluation.add_argument("--baseline", type=Path)
    evaluation.add_argument("--split", choices=["train", "dev", "test"], default="test")
    evaluation.add_argument("--bootstrap", type=int, default=1000)
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("manifest", type=Path)
    benchmark.add_argument("--split", choices=["train", "dev", "test"], default="test")
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--bootstrap", type=int, default=1000)
    benchmark.add_argument("--seed", type=int, default=42)
    _asr_options(benchmark)
    notes = commands.add_parser("notes")
    notes.add_argument("transcript", type=Path)
    notes.add_argument("--method", choices=["extractive", "ollama", "positional"], default="extractive")
    notes.add_argument("--model", default="llama3.1:latest")
    notes.add_argument("--endpoint", default="http://127.0.0.1:11434")
    notes.add_argument("--output", type=Path, required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            print(json.dumps(_doctor(), indent=2))
        elif args.command == "validate":
            samples = load_manifest(args.manifest, args.require_audio)
            print(json.dumps({"samples": len(samples), "languages": sorted({s.language for s in samples}),
                              "manifest_sha256": file_sha256(args.manifest)}))
        elif args.command == "transcribe":
            result = Transcriber(_config(args)).transcribe_file(args.audio, args.language, args.target_policy)
            result["source_audio_sha256"] = file_sha256(args.audio)
            result["review_spans"] = review_spans(result["text"], result["words"], args.language)
            write_json(args.output, result)
        elif args.command == "evaluate":
            samples = [s for s in load_manifest(args.manifest) if s.split == args.split]
            predictions = read_predictions(args.predictions)
            report = evaluate_predictions(samples, predictions, args.bootstrap, args.seed)
            if args.baseline:
                report["paired_comparison"] = paired_comparison(samples, read_predictions(args.baseline),
                                                               predictions, args.bootstrap, args.seed)
            write_json(args.output, report)
        elif args.command == "benchmark":
            _benchmark(args)
        elif args.command == "notes":
            raw = args.transcript.read_text(encoding="utf-8")
            text = json.loads(raw)["text"] if args.transcript.suffix == ".json" else raw
            if args.method == "extractive":
                result = extractive_note(text)
            elif args.method == "positional":
                result = ollama_positional_note(text, args.model, args.endpoint)
            else:
                validate_local_model(args.model, args.endpoint, 45)
                result = ollama_note(text, args.model, args.endpoint)
            write_json(args.output, result)
        elif args.command == "demo":
            from .demo import launch
            launch(args.port)
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, urllib.error.URLError) as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0
