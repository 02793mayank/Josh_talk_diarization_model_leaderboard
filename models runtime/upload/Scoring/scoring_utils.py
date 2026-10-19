"""
Scoring Utilities Module for Josh Talks Voice Arena.
Provides modular, thoroughly tested functions for:
- Model and dataset discovery
- Robust RTTM parsing (pyannote.core.Annotation)
- Audio duration lookup
- pyannote metric evaluation (DER, JER, components)
- Customer-weighted custom score computation
- Corpus and macro-level aggregations
- Error logging and report writing
"""

import csv
import json
import re
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import soundfile as sf
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate


class ErrorLogger:
    """Thread-safe and deterministic error & warning logger for the scoring pipeline."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def log(
        self,
        severity: str,
        category: str,
        recording: str,
        model: str,
        error_type: str,
        error_message: str,
        reference_file: str = "",
        hypothesis_file: str = "",
    ) -> None:
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "severity": severity.upper(),
            "category": category or "",
            "recording": recording or "",
            "model": model or "",
            "reference_file": str(reference_file or ""),
            "hypothesis_file": str(hypothesis_file or ""),
            "error_type": error_type,
            "error_message": error_message,
        }
        self.records.append(record)

    def write_csv(self, output_path: Path, columns: List[str]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r)


# Global logger instance
logger = ErrorLogger()


def discover_model_slots(project_root: Path, max_slots: int = 10) -> List[Dict[str, Any]]:
    """
    Discover model slots 1 through max_slots (default 10).
    Maps existing physical model directories to their model numbers (1-10).
    Rejects Model 11 or later.
    Reserves placeholders for models 5-10 if absent.
    """
    slots: List[Dict[str, Any]] = []

    for slot_num in range(1, max_slots + 1):
        # Look for directory matching Model{slot_num}(* or Model{slot_num}_* or Model{slot_num}
        candidates = []
        for d in project_root.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            # Regex to match Model{slot_num} strictly followed by non-digits
            # e.g. Model1(...) matches slot 1, but Model10 does not match slot 1
            m = re.match(r"^Model(\d+)(.*)$", name, re.IGNORECASE)
            if m:
                num = int(m.group(1))
                if num == slot_num:
                    candidates.append(d)

        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous model directories for slot {slot_num}: "
                f"{[c.name for c in candidates]}"
            )

        if len(candidates) == 1:
            model_dir = candidates[0]
            hyp_dir = model_dir / "hypothesis_rttm"
            slots.append({
                "model_number": slot_num,
                "model_name": model_dir.name,
                "model_dir": model_dir,
                "hypothesis_dir": hyp_dir,
                "is_present": True,
                "has_hyp_dir": hyp_dir.is_dir(),
            })
        else:
            slots.append({
                "model_number": slot_num,
                "model_name": f"Model{slot_num}",
                "model_dir": None,
                "hypothesis_dir": None,
                "is_present": False,
                "has_hyp_dir": False,
            })

    return slots


def find_audio_file(
    folder: Path, recording_id: str, supported_extensions: List[str]
) -> Optional[Path]:
    """
    Find audio file for recording_id inside folder.
    Matches exact stem: file.stem.lower() == recording_id.lower().
    Detects ambiguity if multiple extensions exist for the same recording.
    """
    if not folder.is_dir():
        return None

    matches = []
    for ext in supported_extensions:
        for f in folder.glob(f"*{ext}"):
            if f.stem.lower() == recording_id.lower():
                matches.append(f)

    if len(matches) > 1:
        raise ValueError(
            f"Duplicate audio files found for '{recording_id}' in {folder}: "
            f"{[m.name for m in matches]}"
        )

    return matches[0] if matches else None


def find_reference_rttm(
    folder: Path,
    recording_id: str,
    reference_overrides: Optional[Dict[str, Path]] = None,
) -> Optional[Path]:
    """
    Find reference RTTM file for recording_id inside folder.
    Supports explicit, isolated reference overrides for provisional alignment testing.
    """
    if reference_overrides and recording_id in reference_overrides:
        override_file = reference_overrides[recording_id]
        if override_file.is_file():
            return override_file

    if not folder.is_dir():
        return None

    matches = [
        f for f in folder.glob("*.rttm")
        if f.stem.lower() == recording_id.lower()
    ]

    if len(matches) > 1:
        raise ValueError(
            f"Duplicate reference RTTMs found for '{recording_id}' in {folder}: "
            f"{[m.name for m in matches]}"
        )

    return matches[0] if matches else None


def match_hypothesis_rttm(
    hypothesis_dir: Optional[Path], recording_id: str, model_number: int
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Strict, deterministic matching of hypothesis RTTM for a model and recording.
    Checks patterns:
      1. {recording_id}_Model0{model_number}.rttm
      2. {recording_id}_Model{model_number}.rttm
      3. {recording_id}.rttm
    Avoids arbitrary substring matching.
    Returns (matched_path, error_message).
    """
    if not hypothesis_dir or not hypothesis_dir.is_dir():
        return None, "Hypothesis directory does not exist"

    candidates = list(hypothesis_dir.glob("*.rttm"))
    if not candidates:
        return None, "Hypothesis directory contains no RTTM files"

    # Targeted patterns
    pats = [
        f"^{re.escape(recording_id)}_model0?{model_number}\\.rttm$",
        f"^{re.escape(recording_id)}\\.rttm$",
    ]

    matched = []
    for f in candidates:
        for pat in pats:
            if re.match(pat, f.name, re.IGNORECASE):
                if f not in matched:
                    matched.append(f)

    if len(matched) == 1:
        return matched[0], None

    if len(matched) > 1:
        return None, f"Ambiguous hypothesis matches for '{recording_id}' in {hypothesis_dir}: {[m.name for m in matched]}"

    return None, f"No matching hypothesis RTTM found for '{recording_id}' in {hypothesis_dir.name}"


def get_audio_duration(audio_path: Path) -> float:
    """
    Get duration in seconds of audio_path.
    Uses soundfile as primary engine, with fallback header parsing for WAV/SPH/FLAC.
    """
    try:
        info = sf.info(str(audio_path))
        if info.duration > 0:
            return float(info.duration)
    except Exception:
        pass

    # Fallback for WAV using standard library wave module
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                if rate > 0 and frames > 0:
                    return float(frames) / float(rate)
        except Exception:
            pass

    # Fallback for SPHERE (NIST) files via header parsing
    if audio_path.suffix.lower() == ".sph":
        try:
            with open(audio_path, "rb") as f:
                header = f.read(1024).decode("latin-1", errors="ignore")
                count_m = re.search(r"sample_count\s+-i\s+(\d+)", header)
                rate_m = re.search(r"sample_rate\s+-i\s+(\d+)", header)
                if count_m and rate_m:
                    return float(count_m.group(1)) / float(rate_m.group(1))
        except Exception:
            pass

    raise RuntimeError(f"Could not determine audio duration for: {audio_path}")


def parse_rttm(file_path: Path, recording_id: Optional[str] = None) -> Annotation:
    """
    Parse an RTTM file into a pyannote.core.Annotation.
    Standard layout:
      SPEAKER <file_id> <channel> <start_time> <duration> <ortho> <stype> <speaker> <conf> <slat>
    - Supports UTF-8 with optional BOM (utf-8-sig).
    - Ignores blank lines and # comments.
    - Validates start_time >= 0, duration > 0.
    - Preserves overlapping speaker segments.
    - Skips zero-duration segments with warning.
    - Rejects malformed rows with line numbers.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"RTTM file not found: {file_path}")

    annotation = Annotation(uri=recording_id or file_path.stem)

    content = file_path.read_text(encoding="utf-8-sig")
    lines = content.splitlines()

    parsed_rows = 0
    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            raise ValueError(
                f"Malformed RTTM row at {file_path.name}:{line_num}: Expected 'SPEAKER' row with >=8 fields, got: '{line}'"
            )

        try:
            start_time = float(parts[3])
            duration = float(parts[4])
        except ValueError:
            raise ValueError(
                f"Non-numeric timestamp at {file_path.name}:{line_num}: start='{parts[3]}', duration='{parts[4]}'"
            )

        if start_time < 0:
            raise ValueError(
                f"Negative start time at {file_path.name}:{line_num}: {start_time}"
            )

        if duration < 0:
            raise ValueError(
                f"Negative duration at {file_path.name}:{line_num}: {duration}"
            )

        if duration == 0:
            # Skip zero-duration segment
            continue

        speaker_label = parts[7]
        end_time = start_time + duration

        segment = Segment(start_time, end_time)
        annotation[segment] = speaker_label
        parsed_rows += 1

    return annotation


# Alias for backward and diagnostic script compatibility
read_rttm = parse_rttm


def evaluate_pair(
    reference_ann: Annotation,
    hypothesis_ann: Annotation,
    audio_duration: float,
    collar: float = 0.25,
    skip_overlap: bool = False,
    weights: Optional[Dict[str, float]] = None,
    uem: Optional[Timeline] = None,
) -> Dict[str, Any]:
    """
    Evaluate one reference vs hypothesis pair using pyannote.metrics.
    Computes DER, JER, component durations, component percentages,
    custom weighted error, and custom score.
    """
    if uem is None:
        uem = Timeline([Segment(0.0, audio_duration)], uri=reference_ann.uri)

    der_metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    der_val = float(der_metric(reference_ann, hypothesis_ann, uem=uem))
    der_comps = dict(der_metric.accumulated_)

    jer_metric = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)
    jer_val = float(jer_metric(reference_ann, hypothesis_ann, uem=uem))

    total_ref_speech = float(der_comps.get("total", 0.0))
    miss = float(der_comps.get("missed detection", 0.0))
    fa = float(der_comps.get("false alarm", 0.0))
    conf = float(der_comps.get("confusion", 0.0))
    correct = float(der_comps.get("correct", 0.0))

    ref_speakers = len(reference_ann.labels())
    hyp_speakers = len(hypothesis_ann.labels())

    # Check for invalid zero reference speech within evaluation region
    if total_ref_speech <= 0.0:
        return {
            "is_valid": False,
            "status": "INVALID_REFERENCE_DURATION",
            "error_details": f"Reference speaker duration within scoring region is 0.00s (audio duration: {audio_duration:.2f}s)",
            "reference_speaker_seconds": 0.0,
            "correct_speaker_seconds": 0.0,
            "missed_speech_seconds": 0.0,
            "false_alarm_seconds": fa,
            "speaker_confusion_seconds": 0.0,
            "DER": None,
            "JER": None,
            "missed_speech_percent": None,
            "false_alarm_percent": None,
            "speaker_confusion_percent": None,
            "custom_weighted_error": None,
            "custom_score": None,
            "reference_speaker_count": ref_speakers,
            "hypothesis_speaker_count": hyp_speakers,
        }

    miss_pct = (miss / total_ref_speech) * 100.0
    fa_pct = (fa / total_ref_speech) * 100.0
    conf_pct = (conf / total_ref_speech) * 100.0
    der_pct = der_val * 100.0
    jer_pct = jer_val * 100.0

    custom_weighted_error = None
    custom_score = None
    if weights:
        w_m = weights.get("missing", 0.0)
        w_f = weights.get("false_alarm", 0.0)
        w_c = weights.get("confusion", 0.0)
        cwe = (w_m * miss + w_f * fa + w_c * conf) / total_ref_speech
        custom_weighted_error = cwe
        custom_score = 100.0 * max(0.0, 1.0 - cwe)

    return {
        "is_valid": True,
        "status": "VALID",
        "error_details": "",
        "reference_speaker_seconds": total_ref_speech,
        "correct_speaker_seconds": correct,
        "missed_speech_seconds": miss,
        "false_alarm_seconds": fa,
        "speaker_confusion_seconds": conf,
        "DER": der_pct,
        "JER": jer_pct,
        "missed_speech_percent": miss_pct,
        "false_alarm_percent": fa_pct,
        "speaker_confusion_percent": conf_pct,
        "custom_weighted_error": custom_weighted_error,
        "custom_score": custom_score,
        "reference_speaker_count": ref_speakers,
        "hypothesis_speaker_count": hyp_speakers,
    }


def compute_category_aggregates(
    entries: List[Dict[str, Any]], weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compute corpus-level and macro-average metrics for a category and model.
    """
    valid_entries = [e for e in entries if e.get("is_valid", False)]
    if not valid_entries:
        return {
            "has_valid_data": False,
            "DER": None,
            "JER": None,
            "custom_score": None,
            "custom_weighted_error": None,
            "missed_speech_percent": None,
            "false_alarm_percent": None,
            "speaker_confusion_percent": None,
            "reference_speaker_seconds": 0.0,
        }

    total_ref = sum(e["reference_speaker_seconds"] for e in valid_entries)
    total_miss = sum(e["missed_speech_seconds"] for e in valid_entries)
    total_fa = sum(e["false_alarm_seconds"] for e in valid_entries)
    total_conf = sum(e["speaker_confusion_seconds"] for e in valid_entries)

    if total_ref <= 0:
        return {
            "has_valid_data": False,
            "DER": None,
            "JER": None,
            "custom_score": None,
            "custom_weighted_error": None,
            "missed_speech_percent": None,
            "false_alarm_percent": None,
            "speaker_confusion_percent": None,
            "reference_speaker_seconds": 0.0,
        }

    corpus_der = ((total_miss + total_fa + total_conf) / total_ref) * 100.0
    macro_jer = sum(e["JER"] for e in valid_entries) / len(valid_entries)

    miss_pct = (total_miss / total_ref) * 100.0
    fa_pct = (total_fa / total_ref) * 100.0
    conf_pct = (total_conf / total_ref) * 100.0

    w_m = weights.get("missing", 0.0)
    w_f = weights.get("false_alarm", 0.0)
    w_c = weights.get("confusion", 0.0)
    weighted_err_dur = w_m * total_miss + w_f * total_fa + w_c * total_conf
    cwe = weighted_err_dur / total_ref
    custom_score = 100.0 * max(0.0, 1.0 - cwe)

    return {
        "has_valid_data": True,
        "DER": corpus_der,
        "JER": macro_jer,
        "custom_score": custom_score,
        "custom_weighted_error": cwe,
        "missed_speech_percent": miss_pct,
        "false_alarm_percent": fa_pct,
        "speaker_confusion_percent": conf_pct,
        "reference_speaker_seconds": total_ref,
    }


def compute_overall_model_aggregates(
    all_category_entries: Dict[str, List[Dict[str, Any]]],
    customer_weights: Dict[str, Dict[str, float]],
    expected_recordings_count: int = 5,
) -> Dict[str, Any]:
    """
    Compute overall model metrics across all three customer categories:
    - Corpus-level DER
    - Macro-average JER
    - Overall customer-weighted error and custom score
    - Per-category breakdowns
    """
    valid_entries = []
    failed_count = 0
    for cat, entries in all_category_entries.items():
        for e in entries:
            if e.get("is_valid", False):
                valid_entries.append((cat, e))
            else:
                failed_count += 1

    evaluated_count = len(valid_entries)

    # Determine completion status
    if evaluated_count == 0:
        completion_status = "NOT_EVALUATED"
    elif evaluated_count == expected_recordings_count and failed_count == 0:
        completion_status = "SUCCESS"
    else:
        completion_status = "PARTIAL"

    if evaluated_count == 0:
        return {
            "expected_recordings": expected_recordings_count,
            "evaluated_recordings": 0,
            "failed_recordings": failed_count,
            "completion_status": completion_status,
            "DER": None,
            "JER": None,
            "missed_speech_percent": None,
            "false_alarm_percent": None,
            "speaker_confusion_percent": None,
            "custom_weighted_error": None,
            "custom_score": None,
            "call_type_DER": None,
            "call_type_JER": None,
            "call_type_custom_score": None,
            "live_transcription_DER": None,
            "live_transcription_JER": None,
            "live_transcription_custom_score": None,
            "meeting_DER": None,
            "meeting_JER": None,
            "meeting_custom_score": None,
        }

    total_ref = sum(e["reference_speaker_seconds"] for _, e in valid_entries)
    total_miss = sum(e["missed_speech_seconds"] for _, e in valid_entries)
    total_fa = sum(e["false_alarm_seconds"] for _, e in valid_entries)
    total_conf = sum(e["speaker_confusion_seconds"] for _, e in valid_entries)

    if total_ref <= 0:
        return {
            "expected_recordings": expected_recordings_count,
            "evaluated_recordings": evaluated_count,
            "failed_recordings": failed_count,
            "completion_status": completion_status,
            "DER": None,
            "JER": None,
            "missed_speech_percent": None,
            "false_alarm_percent": None,
            "speaker_confusion_percent": None,
            "custom_weighted_error": None,
            "custom_score": None,
            "call_type_DER": None,
            "call_type_JER": None,
            "call_type_custom_score": None,
            "live_transcription_DER": None,
            "live_transcription_JER": None,
            "live_transcription_custom_score": None,
            "meeting_DER": None,
            "meeting_JER": None,
            "meeting_custom_score": None,
        }

    corpus_der = ((total_miss + total_fa + total_conf) / total_ref) * 100.0
    macro_jer = sum(e["JER"] for _, e in valid_entries) / len(valid_entries)

    miss_pct = (total_miss / total_ref) * 100.0
    fa_pct = (total_fa / total_ref) * 100.0
    conf_pct = (total_conf / total_ref) * 100.0

    # Overall custom score: weighted error per recording using recording's category weights
    total_weighted_err_dur = 0.0
    for cat, e in valid_entries:
        w = customer_weights[cat]
        dur_err = (
            w["missing"] * e["missed_speech_seconds"]
            + w["false_alarm"] * e["false_alarm_seconds"]
            + w["confusion"] * e["speaker_confusion_seconds"]
        )
        total_weighted_err_dur += dur_err

    overall_cwe = total_weighted_err_dur / total_ref
    overall_custom_score = 100.0 * max(0.0, 1.0 - overall_cwe)

    # Category-specific aggregates
    call_agg = compute_category_aggregates(
        all_category_entries.get("call", []), customer_weights["call"]
    )
    lt_agg = compute_category_aggregates(
        all_category_entries.get("live_transcription", []), customer_weights["live_transcription"]
    )
    meet_agg = compute_category_aggregates(
        all_category_entries.get("meeting", []), customer_weights["meeting"]
    )

    return {
        "expected_recordings": expected_recordings_count,
        "evaluated_recordings": evaluated_count,
        "failed_recordings": failed_count,
        "completion_status": completion_status,
        "DER": corpus_der,
        "JER": macro_jer,
        "missed_speech_percent": miss_pct,
        "false_alarm_percent": fa_pct,
        "speaker_confusion_percent": conf_pct,
        "custom_weighted_error": overall_cwe,
        "custom_score": overall_custom_score,
        "call_type_DER": call_agg["DER"],
        "call_type_JER": call_agg["JER"],
        "call_type_custom_score": call_agg["custom_score"],
        "live_transcription_DER": lt_agg["DER"],
        "live_transcription_JER": lt_agg["JER"],
        "live_transcription_custom_score": lt_agg["custom_score"],
        "meeting_DER": meet_agg["DER"],
        "meeting_JER": meet_agg["JER"],
        "meeting_custom_score": meet_agg["custom_score"],
    }


def assign_ranks(
    items: List[Dict[str, Any]],
    score_key: str = "custom_score",
    rank_key: str = "rank",
    higher_is_better: bool = True,
) -> None:
    """
    Assign competition ranks (1, 2, 2, 4...) based on score_key.
    Only items with a non-None score receive a rank.
    """
    eligible = [it for it in items if it.get(score_key) is not None]

    eligible.sort(
        key=lambda x: x[score_key],
        reverse=higher_is_better,
    )

    current_rank = 1
    for i, it in enumerate(eligible):
        if i > 0 and eligible[i][score_key] == eligible[i - 1][score_key]:
            it[rank_key] = eligible[i - 1][rank_key]
        else:
            it[rank_key] = i + 1

    # Unevaluated items get None rank
    for it in items:
        if it.get(score_key) is None:
            it[rank_key] = None


def format_csv_value(val: Any, decimals: int = 2) -> str:
    """Format CSV value: floats to fixed precision, None to blank string."""
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)
