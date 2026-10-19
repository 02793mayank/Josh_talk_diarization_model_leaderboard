"""
Scoring Configuration Module for Josh Talks Voice Arena.
Centralizes all project paths, customer categories, model slots,
evaluation parameters, customer-specific weights, and output definitions.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any


def get_default_project_root() -> Path:
    """Resolve project root based on scoring_config.py location:
    Scoring/scoring_config.py -> parent of Scoring/ is the project root."""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_default_project_root()

# Reference dataset folder locations relative to project root
CATEGORY_AUDIO_FOLDERS = {
    "call": PROJECT_ROOT / "Audio and annotations" / "call_inputs_audio",
    "live_transcription": PROJECT_ROOT / "Audio and annotations" / "liveTranscription_inputs_audio",
    "meeting": PROJECT_ROOT / "Audio and annotations" / "meeting_inputs",
}

CATEGORY_RTTM_FOLDERS = {
    "call": PROJECT_ROOT / "Audio and annotations" / "call_inputs_rttm",
    "live_transcription": PROJECT_ROOT / "Audio and annotations" / "liveTranscription_inputs_rttm",
    "meeting": PROJECT_ROOT / "Audio and annotations" / "meeting_inputs_rttm",
}

# Directory for provisionally aligned reference copies (kept strictly outside Audio and annotations)
ALIGNED_REFERENCES_DIR = PROJECT_ROOT / "Scoring" / "aligned_references"

# Explicit recording-specific reference override map (for provisional alignment testing only)
PROVISIONAL_REFERENCE_MAP: Dict[str, Path] = {
    "call_input02": ALIGNED_REFERENCES_DIR / "call_input02_aligned.rttm",
}

# Fixed benchmark recordings by customer category
RECORDINGS_BY_CATEGORY: Dict[str, List[str]] = {
    "call": ["call_input01", "call_input02"],
    "live_transcription": ["liveTranscription_input01", "liveTranscription_input03"],
    "meeting": ["meeting_input"],
}

ALL_EXPECTED_RECORDINGS: List[str] = [
    rec for recs in RECORDINGS_BY_CATEGORY.values() for rec in recs
]

# Supported audio extensions
SUPPORTED_AUDIO_EXTENSIONS = [
    ".wav", ".flac", ".sph", ".ogg", ".aif", ".aiff", ".mp3", ".m4a"
]

# Evaluation parameters
DEFAULT_COLLAR_SECONDS = 0.25
DEFAULT_SKIP_OVERLAP = False
DEFAULT_UEM_POLICY = "FULL_AUDIO_NO_UEM"

# Fixed customer-specific weights (Must sum to 1.0 within tolerance)
CUSTOMER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "call": {
        "missing": 0.40,
        "false_alarm": 0.25,
        "confusion": 0.35,
    },
    "live_transcription": {
        "missing": 0.50,
        "false_alarm": 0.20,
        "confusion": 0.30,
    },
    "meeting": {
        "missing": 0.45,
        "false_alarm": 0.15,
        "confusion": 0.40,
    },
}


def validate_customer_weights(weights: Dict[str, Dict[str, float]] = None) -> None:
    """Validate that customer-specific weights exist and sum to 1.0 for each category."""
    if weights is None:
        weights = CUSTOMER_WEIGHTS

    required_categories = ["call", "live_transcription", "meeting"]
    required_components = ["missing", "false_alarm", "confusion"]

    for cat in required_categories:
        if cat not in weights:
            raise ValueError(f"Missing required customer category in weights: '{cat}'")
        cat_w = weights[cat]
        for comp in required_components:
            if comp not in cat_w:
                raise ValueError(f"Missing component '{comp}' in weights for category '{cat}'")
            if not isinstance(cat_w[comp], (int, float)) or cat_w[comp] < 0:
                raise ValueError(f"Weight for '{cat}.{comp}' must be a non-negative number: {cat_w[comp]}")

        total = sum(cat_w[comp] for comp in required_components)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Customer weights for category '{cat}' must sum to 1.00 (got {total:.6f})"
            )


# 10 Model slots configuration
MODEL_SLOTS = list(range(1, 11))

# Output paths
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scoring_output"
OUTPUT_FILES = {
    "call_type_report": "call_type_report.csv",
    "live_transcription_report": "live_transcription_report.csv",
    "meeting_report": "meeting_report.csv",
    "overall_model_report": "overall_model_report.csv",
    "evaluation_errors": "evaluation_errors.csv",
    "run_summary": "run_summary.json",
}

# CSV schema definitions
CUSTOMER_REPORT_COLUMNS = [
    "category",
    "recording",
    "model",
    "model_number",
    "status",
    "rank",
    "audio_file",
    "reference_file",
    "hypothesis_file",
    "collar_seconds",
    "skip_overlap",
    "uem_source",
    "reference_speaker_seconds",
    "correct_speaker_seconds",
    "missed_speech_seconds",
    "false_alarm_seconds",
    "speaker_confusion_seconds",
    "DER",
    "JER",
    "missed_speech_percent",
    "false_alarm_percent",
    "speaker_confusion_percent",
    "weight_missing",
    "weight_false_alarm",
    "weight_confusion",
    "custom_weighted_error",
    "custom_score",
    "reference_speaker_count",
    "hypothesis_speaker_count",
    "validation_status",
    "error_details",
]

OVERALL_REPORT_COLUMNS = [
    "overall_rank",
    "model",
    "model_number",
    "expected_recordings",
    "evaluated_recordings",
    "failed_recordings",
    "completion_status",
    "DER",
    "JER",
    "missed_speech_percent",
    "false_alarm_percent",
    "speaker_confusion_percent",
    "custom_weighted_error",
    "custom_score",
    "call_type_DER",
    "call_type_JER",
    "call_type_custom_score",
    "live_transcription_DER",
    "live_transcription_JER",
    "live_transcription_custom_score",
    "meeting_DER",
    "meeting_JER",
    "meeting_custom_score",
]

ERROR_REPORT_COLUMNS = [
    "timestamp",
    "severity",
    "category",
    "recording",
    "model",
    "reference_file",
    "hypothesis_file",
    "error_type",
    "error_message",
]
