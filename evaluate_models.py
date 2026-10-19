from pathlib import Path
import re
import csv
import sys

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

try:
    import soundfile as sf
except ImportError:
    sf = None


# ------------------------- Configuration -------------------------

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "references"
HYPOTHESIS_DIR = BASE_DIR / "hypotheses"

# Follow your benchmark protocol if it specifies a different collar/overlap policy.
COLLAR = 0.25
SKIP_OVERLAP = False

CATEGORIES = {
    "call": {
        "reference": "call_inputs",
        "output": "call_output",
        "report": "call_observation.csv",
    },
    "liveTranscription": {
        "reference": "liveTranscription_inputs",
        "output": "liveTranscription_output",
        "report": "liveTranscription_observation.csv",
    },
    "meeting": {
        "reference": "meeting_inputs",
        "output": "meeting_output",
        "report": "meeting_observation.csv",
    },
}


# ------------------------- Discover models -------------------------

def discover_models():
    if not HYPOTHESIS_DIR.is_dir():
        raise FileNotFoundError(f"Hypotheses folder not found: {HYPOTHESIS_DIR}")

    models = []
    for folder in HYPOTHESIS_DIR.iterdir():
        if not folder.is_dir():
            continue
        match = re.fullmatch(r"model\s*(\d+)", folder.name.strip(), re.IGNORECASE)
        if match:
            models.append((int(match.group(1)), folder))

    models.sort(key=lambda item: item[0])
    if not models:
        raise FileNotFoundError(
            f"No model folders like 'model 1' or 'model 10' found in {HYPOTHESIS_DIR}"
        )

    print(f"Found {len(models)} model folder(s):")
    for number, folder in models:
        print(f"  Model {number}: {folder.name}")
    return models


# ------------------------- Read RTTM -------------------------

def read_rttm(path: Path) -> Annotation:
    annotation = Annotation(uri=path.stem)

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if fields[0].upper() != "SPEAKER":
                continue
            if len(fields) < 8:
                raise ValueError(f"Malformed RTTM row at {path}:{line_number}: {line}")

            try:
                start = float(fields[3])
                duration = float(fields[4])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid start/duration at {path}:{line_number}: {line}"
                ) from exc

            if duration < 0:
                raise ValueError(f"Negative duration at {path}:{line_number}")
            if duration == 0:
                continue

            annotation[Segment(start, start + duration)] = fields[7]

    return annotation


# ------------------------- UEM / scoring region -------------------------

def read_uem(path: Path, expected_uri: str) -> Timeline:
    segments = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed UEM row at {path}:{line_number}: {line}")

            if fields[0] != expected_uri:
                continue

            try:
                start, end = float(fields[2]), float(fields[3])
            except ValueError as exc:
                raise ValueError(f"Invalid UEM times at {path}:{line_number}") from exc

            if end > start:
                segments.append(Segment(start, end))

    if not segments:
        raise ValueError(
            f"No UEM segment in {path} matched recording ID '{expected_uri}'. "
            "The UEM first column should match the reference RTTM filename stem."
        )

    return Timeline(segments).support()


def find_uem(reference_file: Path, reference_folder: Path):
    direct = reference_file.with_suffix(".uem")
    if direct.is_file():
        return direct
    matches = list(reference_folder.rglob(reference_file.stem + ".uem"))
    return matches[0] if matches else None


def find_audio(reference_file: Path, reference_folder: Path):
    extensions = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".sph"}
    for parent in (reference_folder, BASE_DIR / "resources"):
        if not parent.exists():
            continue
        direct = [
            parent / (reference_file.stem + ext)
            for ext in extensions
        ]
        for candidate in direct:
            if candidate.is_file():
                return candidate
        for candidate in parent.rglob(reference_file.stem + ".*"):
            if candidate.is_file() and candidate.suffix.lower() in extensions:
                return candidate
    return None


def get_scoring_region(reference_file: Path, reference_folder: Path):
    uem_file = find_uem(reference_file, reference_folder)
    if uem_file:
        return read_uem(uem_file, reference_file.stem), f"UEM: {uem_file.name}"

    audio_file = find_audio(reference_file, reference_folder)
    if audio_file:
        if sf is None:
            raise ImportError(
                "Install soundfile to read audio duration: python -m pip install soundfile"
            )
        duration = float(sf.info(str(audio_file)).duration)
        print(
            f"  WARNING: No UEM for {reference_file.name}; scoring the full "
            f"{duration:.2f}s duration of {audio_file.name}."
        )
        return Timeline([Segment(0.0, duration)]).support(), f"Full audio: {audio_file.name}"

    raise FileNotFoundError(
        f"No UEM or matching audio found for {reference_file.name}. "
        f"Add {reference_file.stem}.uem beside the RTTM, or matching audio under "
        f"{reference_folder} or {BASE_DIR / 'resources'}."
    )


# ------------------------- Match hypotheses -------------------------

def find_hypothesis(reference_file: Path, model_dir: Path,
                    output_folder_name: str, model_number: int):
    expected_name = f"{reference_file.stem}_model{model_number:02d}.rttm"

    preferred = model_dir / output_folder_name / expected_name
    if preferred.is_file():
        return preferred

    matches = list(model_dir.rglob(expected_name))
    return matches[0] if matches else None


# ------------------------- Evaluate -------------------------

def safe_percent(value, denominator):
    return 100.0 * value / denominator if denominator else 0.0


def evaluate_pair(reference_file: Path, hypothesis_file: Path,
                  reference_folder: Path, model_name: str, category_name: str):
    reference = read_rttm(reference_file)
    hypothesis = read_rttm(hypothesis_file)
    uem, uem_source = get_scoring_region(reference_file, reference_folder)

    der_metric = DiarizationErrorRate(collar=COLLAR, skip_overlap=SKIP_OVERLAP)
    jer_metric = JaccardErrorRate(collar=COLLAR, skip_overlap=SKIP_OVERLAP)

    der_value = float(der_metric(reference, hypothesis, uem=uem))
    jer_value = float(jer_metric(reference, hypothesis, uem=uem))
    components = der_metric.compute_components(reference, hypothesis, uem=uem)

    total = float(components.get("total", 0.0))
    correct = float(components.get("correct", 0.0))
    missed = float(components.get("missed detection", 0.0))
    false_alarm = float(components.get("false alarm", 0.0))
    confusion = float(components.get("confusion", 0.0))

    # Supplementary speech activity estimates; not a substitute for dedicated SAD evaluation.
    precision = correct / (correct + false_alarm) if correct + false_alarm else 0.0
    recall = correct / (correct + missed) if correct + missed else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "category": category_name,
        "model": model_name,
        "recording": reference_file.stem,
        "reference_file": str(reference_file.relative_to(BASE_DIR)),
        "hypothesis_file": str(hypothesis_file.relative_to(BASE_DIR)),
        "uem_source": uem_source,
        "collar_seconds": COLLAR,
        "skip_overlap": SKIP_OVERLAP,
        "reference_speech_seconds": total,
        "correct_seconds": correct,
        "missed_speech_seconds": missed,
        "false_alarm_seconds": false_alarm,
        "speaker_confusion_seconds": confusion,
        "missed_speech_percent": safe_percent(missed, total),
        "false_alarm_percent": safe_percent(false_alarm, total),
        "speaker_confusion_percent": safe_percent(confusion, total),
        "DER": der_value,
        "JER": jer_value,
        "speech_precision": precision,
        "speech_recall": recall,
        "speech_F1": f1,
        "reference_speaker_count": len(reference.labels()),
        "hypothesis_speaker_count": len(hypothesis.labels()),
    }


def aggregate_model_rows(rows):
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    summaries = []
    for model_name, model_rows in by_model.items():
        total_ref = sum(r["reference_speech_seconds"] for r in model_rows)
        total_correct = sum(r["correct_seconds"] for r in model_rows)
        total_missed = sum(r["missed_speech_seconds"] for r in model_rows)
        total_fa = sum(r["false_alarm_seconds"] for r in model_rows)
        total_confusion = sum(r["speaker_confusion_seconds"] for r in model_rows)

        def mean(key):
            return sum(float(r[key]) for r in model_rows) / len(model_rows)

        summary = {
            "category": model_rows[0]["category"],
            "model": model_name,
            "recording": "AVERAGE",
            "recordings_evaluated": len(model_rows),
            "reference_speech_seconds": total_ref,
            "correct_seconds": total_correct,
            "missed_speech_seconds": total_missed,
            "false_alarm_seconds": total_fa,
            "speaker_confusion_seconds": total_confusion,
            "missed_speech_percent": safe_percent(total_missed, total_ref),
            "false_alarm_percent": safe_percent(total_fa, total_ref),
            "speaker_confusion_percent": safe_percent(total_confusion, total_ref),
            "DER": safe_percent(total_missed + total_fa + total_confusion, total_ref) / 100,
            "JER": mean("JER"),  # macro-average across recordings
            "speech_precision": mean("speech_precision"),
            "speech_recall": mean("speech_recall"),
            "speech_F1": mean("speech_F1"),
            "reference_speaker_count": mean("reference_speaker_count"),
            "hypothesis_speaker_count": mean("hypothesis_speaker_count"),
            "collar_seconds": COLLAR,
            "skip_overlap": SKIP_OVERLAP,
            "uem_source": "Per-recording UEM/audio",
            "reference_file": "",
            "hypothesis_file": "",
        }
        summaries.append(summary)

    return summaries


def write_csv(path: Path, rows):
    if not rows:
        print(f"  No rows to write for {path.name}")
        return

    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ------------------------- Main -------------------------

def main():
    models = discover_models()
    errors = []

    for category_name, category in CATEGORIES.items():
        reference_folder = REFERENCE_DIR / category["reference"]
        if not reference_folder.is_dir():
            print(f"\nWARNING: Missing reference folder: {reference_folder}")
            continue

        reference_files = sorted(
            p for p in reference_folder.rglob("*.rttm")
            if "_model" not in p.stem.lower()
        )
        if not reference_files:
            print(f"\nWARNING: No reference RTTM files in {reference_folder}")
            continue

        print(f"\n{'=' * 60}\nCategory: {category_name}\n"
              f"Reference files: {len(reference_files)}\n{'=' * 60}")

        category_rows = []
        for model_number, model_dir in models:
            print(f"\nEvaluating {model_dir.name}...")
            count = 0

            for reference_file in reference_files:
                hypothesis_file = find_hypothesis(
                    reference_file, model_dir, category["output"], model_number
                )
                if hypothesis_file is None:
                    message = (
                        f"Expected {reference_file.stem}_model{model_number:02d}.rttm "
                        f"under {model_dir}"
                    )
                    print(f"  WARNING: Missing hypothesis: {message}")
                    errors.append({
                        "category": category_name,
                        "model": model_dir.name,
                        "recording": reference_file.stem,
                        "error": "Missing hypothesis",
                        "details": message,
                    })
                    continue

                try:
                    row = evaluate_pair(
                        reference_file, hypothesis_file, reference_folder,
                        model_dir.name, category_name
                    )
                    row["model_number"] = model_number
                    category_rows.append(row)
                    count += 1
                    print(
                        f"  {reference_file.stem}: DER={row['DER']:.4f}, "
                        f"JER={row['JER']:.4f}"
                    )
                except Exception as exc:
                    print(
                        f"  ERROR: {model_dir.name} / {reference_file.name}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    errors.append({
                        "category": category_name,
                        "model": model_dir.name,
                        "recording": reference_file.stem,
                        "error": type(exc).__name__,
                        "details": str(exc),
                    })

            print(f"  Evaluated {count}/{len(reference_files)} recording(s).")

        report = BASE_DIR / category["report"]
        write_csv(report, category_rows + aggregate_model_rows(category_rows))
        print(f"\nSaved: {report}")

    if errors:
        error_report = BASE_DIR / "evaluation_errors.csv"
        write_csv(error_report, errors)
        print(f"\nSome recordings were skipped. See: {error_report}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
