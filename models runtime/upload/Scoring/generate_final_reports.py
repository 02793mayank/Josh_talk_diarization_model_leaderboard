"""
Script to generate final deliverables:
1. scoring_output/final_comparison_results.csv (50 rows: 10 models x 5 recordings)
2. scoring_output/final_model_summary.csv (10 rows: Models 1 to 10)
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORING_OUT = ROOT / "scoring_output"

# Recordings metadata
RECORDINGS = [
    {
        "id": "call_input01",
        "category": "call",
        "audio_file": "Audio and annotations/call_inputs_audio/call_input01.sph",
        "reference_file": "Audio and annotations/call_inputs_rttm/call_input01.rttm",
        "duration": 120.00,
    },
    {
        "id": "call_input02",
        "category": "call",
        "audio_file": "Audio and annotations/call_inputs_audio/call_input02.flac",
        "reference_file": "Audio and annotations/call_inputs_rttm/call_input02.rttm",
        "duration": 120.00,
    },
    {
        "id": "liveTranscription_input01",
        "category": "live_transcription",
        "audio_file": "Audio and annotations/liveTranscription_inputs_audio/liveTranscription_input01.wav",
        "reference_file": "Audio and annotations/liveTranscription_inputs_rttm/liveTranscription_input01.rttm",
        "duration": 178.69,
    },
    {
        "id": "liveTranscription_input03",
        "category": "live_transcription",
        "audio_file": "Audio and annotations/liveTranscription_inputs_audio/liveTranscription_input03.wav",
        "reference_file": "Audio and annotations/liveTranscription_inputs_rttm/liveTranscription_input03.rttm",
        "duration": 60.00,
    },
    {
        "id": "meeting_input",
        "category": "meeting",
        "audio_file": "Audio and annotations/meeting_inputs/meeting_input.wav",
        "reference_file": "Audio and annotations/meeting_inputs_rttm/meeting_input.rttm",
        "duration": 139.91,
    },
]

MODELS = [
    (1, "Model1(pyannote Community-1)", True),
    (2, "Model2(NVIDIA NeMo Sortformer)", True),
    (3, "Model3(NVIDIA NeMo Clustering Diarizer)", True),
    (4, "Model4(NVIDIA NeMo Streaming Sortformer)", True),
    (5, "Model5(SpeechBrain AMI Diarization)", False),
    (6, "Model6(WeSpeaker Diarization)", False),
    (7, "Model7(ESPnet Diarization)", False),
    (8, "Model8(Kaldi Diarization)", False),
    (9, "Model9(UIS-RNN)", False),
    (10, "Model10(VBx Diarization)", False),
]

# Read official category reports
def load_category_report(filename):
    path = SCORING_OUT / filename
    rows = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (int(r["model_number"]), r["recording"])
                rows[key] = r
    return rows

call_rows = load_category_report("call_type_report.csv")
live_rows = load_category_report("live_transcription_report.csv")
meet_rows = load_category_report("meeting_report.csv")

all_official = {}
all_official.update(call_rows)
all_official.update(live_rows)
all_official.update(meet_rows)

# Provisional metrics for call_input02 (from rigorous offset sweep at 400.0s)
PROVISIONAL_CALL02 = {
    1: {"DER": 73.45, "JER": 45.33, "miss": 6.84, "fa": 66.52, "conf": 0.08, "weighted_error": 0.20, "score": 79.68},
    2: {"DER": 86.60, "JER": 71.10, "miss": 46.22, "fa": 40.23, "conf": 0.15, "weighted_error": 0.29, "score": 70.72},
    3: {"DER": 78.70, "JER": 48.75, "miss": 13.20, "fa": 65.41, "conf": 0.09, "weighted_error": 0.22, "score": 78.12},
    4: {"DER": 77.06, "JER": 67.70, "miss": 44.57, "fa": 32.35, "conf": 0.14, "weighted_error": 0.27, "score": 72.88},
}

# 1. Build final_comparison_results.csv
comparison_csv_path = SCORING_OUT / "final_comparison_results.csv"
comparison_fields = [
    "model_slot",
    "model_name",
    "recording_id",
    "category",
    "audio_duration_sec",
    "evaluation_status",
    "validation_status",
    "DER_percent",
    "JER_percent",
    "missed_speech_percent",
    "false_alarm_percent",
    "speaker_confusion_percent",
    "custom_weighted_error",
    "custom_score",
    "category_rank",
    "provisional_alignment_available",
    "provisional_DER_percent",
    "provisional_JER_percent",
    "provisional_custom_score",
    "hypothesis_file",
    "error_details",
]

comp_rows = []
for slot_num, model_name, is_evaluated in MODELS:
    for rec in RECORDINGS:
        rec_id = rec["id"]
        cat = rec["category"]
        key = (slot_num, rec_id)
        off = all_official.get(key, {})

        if not is_evaluated:
            comp_rows.append({
                "model_slot": slot_num,
                "model_name": model_name,
                "recording_id": rec_id,
                "category": cat,
                "audio_duration_sec": f"{rec['duration']:.2f}",
                "evaluation_status": "NOT_EVALUATED",
                "validation_status": "RESERVED_MODEL_SLOT",
                "DER_percent": "",
                "JER_percent": "",
                "missed_speech_percent": "",
                "false_alarm_percent": "",
                "speaker_confusion_percent": "",
                "custom_weighted_error": "",
                "custom_score": "",
                "category_rank": "",
                "provisional_alignment_available": "FALSE",
                "provisional_DER_percent": "",
                "provisional_JER_percent": "",
                "provisional_custom_score": "",
                "hypothesis_file": "",
                "error_details": "Model slot reserved; no hypothesis RTTM generated.",
            })
        else:
            status = off.get("status", "FAILED")
            val_status = off.get("validation_status", "")
            der = off.get("DER", "")
            jer = off.get("JER", "")
            miss = off.get("missed_speech_percent", "")
            fa = off.get("false_alarm_percent", "")
            conf = off.get("speaker_confusion_percent", "")
            cwe = off.get("custom_weighted_error", "")
            cs = off.get("custom_score", "")
            rank = off.get("rank", "")
            hyp = off.get("hypothesis_file", "")
            err = off.get("error_details", "")

            # Provisional info for call_input02
            prov_avail = "TRUE" if (rec_id == "call_input02" and slot_num in PROVISIONAL_CALL02) else "FALSE"
            prov_der = f"{PROVISIONAL_CALL02[slot_num]['DER']:.2f}" if prov_avail == "TRUE" else ""
            prov_jer = f"{PROVISIONAL_CALL02[slot_num]['JER']:.2f}" if prov_avail == "TRUE" else ""
            prov_cs = f"{PROVISIONAL_CALL02[slot_num]['score']:.2f}" if prov_avail == "TRUE" else ""

            comp_rows.append({
                "model_slot": slot_num,
                "model_name": model_name,
                "recording_id": rec_id,
                "category": cat,
                "audio_duration_sec": f"{rec['duration']:.2f}",
                "evaluation_status": status,
                "validation_status": val_status,
                "DER_percent": der,
                "JER_percent": jer,
                "missed_speech_percent": miss,
                "false_alarm_percent": fa,
                "speaker_confusion_percent": conf,
                "custom_weighted_error": cwe,
                "custom_score": cs,
                "category_rank": rank,
                "provisional_alignment_available": prov_avail,
                "provisional_DER_percent": prov_der,
                "provisional_JER_percent": prov_jer,
                "provisional_custom_score": prov_cs,
                "hypothesis_file": hyp,
                "error_details": err,
            })

with open(comparison_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=comparison_fields)
    writer.writeheader()
    writer.writerows(comp_rows)

print(f"[OK] Generated {comparison_csv_path} with {len(comp_rows)} rows.")

# 2. Build final_model_summary.csv
summary_csv_path = SCORING_OUT / "final_model_summary.csv"
summary_fields = [
    "official_overall_rank",
    "model_slot",
    "model_name",
    "evaluation_status",
    "valid_comparisons_count",
    "invalid_comparisons_count",
    "total_comparisons_count",
    "official_macro_der_percent",
    "official_macro_jer_percent",
    "official_macro_miss_percent",
    "official_macro_fa_percent",
    "official_macro_conf_percent",
    "official_custom_score",
    "call_custom_score",
    "live_transcription_custom_score",
    "meeting_custom_score",
    "provisional_macro_der_percent",
    "provisional_custom_score",
    "provisional_rank",
]

# Read official overall_model_report.csv
overall_map = {}
overall_rep_path = SCORING_OUT / "overall_model_report.csv"
if overall_rep_path.exists():
    with open(overall_rep_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            overall_map[int(r["model_number"])] = r

# Provisional overall metrics
PROVISIONAL_OVERALL = {
    1: {"der": "32.19", "score": "90.58", "rank": "1"},
    4: {"der": "31.56", "score": "90.41", "rank": "2"},
    2: {"der": "31.11", "score": "90.27", "rank": "3"},
    3: {"der": "28.12", "score": "89.97", "rank": "4"},
}

sum_rows = []
for slot_num, model_name, is_evaluated in MODELS:
    off = overall_map.get(slot_num, {})
    if not is_evaluated:
        sum_rows.append({
            "official_overall_rank": "",
            "model_slot": slot_num,
            "model_name": model_name,
            "evaluation_status": "NOT_EVALUATED",
            "valid_comparisons_count": 0,
            "invalid_comparisons_count": 0,
            "total_comparisons_count": 5,
            "official_macro_der_percent": "",
            "official_macro_jer_percent": "",
            "official_macro_miss_percent": "",
            "official_macro_fa_percent": "",
            "official_macro_conf_percent": "",
            "official_custom_score": "",
            "call_custom_score": "",
            "live_transcription_custom_score": "",
            "meeting_custom_score": "",
            "provisional_macro_der_percent": "",
            "provisional_custom_score": "",
            "provisional_rank": "",
        })
    else:
        prov = PROVISIONAL_OVERALL.get(slot_num, {})
        sum_rows.append({
            "official_overall_rank": off.get("overall_rank", ""),
            "model_slot": slot_num,
            "model_name": model_name,
            "evaluation_status": off.get("completion_status", "PARTIAL"),
            "valid_comparisons_count": off.get("evaluated_recordings", "4"),
            "invalid_comparisons_count": off.get("failed_recordings", "1"),
            "total_comparisons_count": off.get("expected_recordings", "5"),
            "official_macro_der_percent": off.get("DER", ""),
            "official_macro_jer_percent": off.get("JER", ""),
            "official_macro_miss_percent": off.get("missed_speech_percent", ""),
            "official_macro_fa_percent": off.get("false_alarm_percent", ""),
            "official_macro_conf_percent": off.get("speaker_confusion_percent", ""),
            "official_custom_score": off.get("custom_score", ""),
            "call_custom_score": off.get("call_type_custom_score", ""),
            "live_transcription_custom_score": off.get("live_transcription_custom_score", ""),
            "meeting_custom_score": off.get("meeting_custom_score", ""),
            "provisional_macro_der_percent": prov.get("der", ""),
            "provisional_custom_score": prov.get("score", ""),
            "provisional_rank": prov.get("rank", ""),
        })

with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(sum_rows)

print(f"[OK] Generated {summary_csv_path} with {len(sum_rows)} rows.")
