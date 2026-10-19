"""
Reproducible Benchmark Evaluation Script for Optional Task - Diarization Scoring.
Uses existing pyannote.metrics installation in Optional Task - Diarization Scoring\\.venv.
Scores pre-generated model hypothesis RTTMs against ground-truth references.
Does NOT run model inference.
"""

import json
from pathlib import Path
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent.parent.parent.parent

RECORDINGS = [
    {
        "id": "liveTranscription_input02",
        "audio_path": PROJECT_ROOT / "Audio and annotations" / "liveTranscription_inputs_audio" / "liveTranscription_input02.wav",
        "ref_path": PROJECT_ROOT / "Audio and annotations" / "liveTranscription_inputs_rttm" / "liveTranscription_input02.rttm",
        "duration": 139.91,
        "source": "VoxConverse",
        "evidence": "Reference URI is 'zvmyn' (matches VoxConverse clip zvmyn.wav).",
        "is_valid": True,
    },
    {
        "id": "call_input01",
        "audio_path": PROJECT_ROOT / "Audio and annotations" / "call_inputs_audio" / "call_input01.sph",
        "ref_path": PROJECT_ROOT / "Audio and annotations" / "call_inputs_rttm" / "call_input01.rttm",
        "duration": 120.00,
        "source": "CALLHOME",
        "evidence": "Reference URI is 'LDC97S42' (matches LDC CALLHOME American English Speech catalog ID).",
        "is_valid": True,
    },
    {
        "id": "call_input02",
        "audio_path": PROJECT_ROOT / "Audio and annotations" / "call_inputs_audio" / "call_input02.flac",
        "ref_path": PROJECT_ROOT / "Audio and annotations" / "call_inputs_rttm" / "call_input02.rttm",
        "duration": 120.00,
        "source": "CALLHOME",
        "evidence": "Reference URI is 'en_5551', but reference timestamps (400.35s-479.77s) are offset by ~400s outside the 120s audio boundary.",
        "is_valid": False,
    },
]

SYSTEMS = [
    {
        "key": "Model01",
        "name": "pyannote Community-1",
        "checkpoint": "pyannote/speaker-diarization-community-1",
        "version": "pyannote.audio 4.0.7",
        "hyp_dir": PROJECT_ROOT / "Model1(pyannote Community-1)" / "hypothesis_rttm",
        "pattern": "{id}_Model01.rttm",
    },
    {
        "key": "Model02",
        "name": "NVIDIA NeMo Sortformer (Offline)",
        "checkpoint": "nvidia/diar_sortformer_4spk-v1",
        "version": "NeMo 3.0.0",
        "hyp_dir": PROJECT_ROOT / "Model2(NVIDIA NeMo Sortformer)" / "hypothesis_rttm",
        "pattern": "{id}_Model02.rttm",
    },
    {
        "key": "Model04",
        "name": "NVIDIA NeMo Streaming Sortformer",
        "checkpoint": "nvidia/diar_streaming_sortformer_4spk-v2.1",
        "version": "NeMo 3.0.0",
        "hyp_dir": PROJECT_ROOT / "Model4(NVIDIA NeMo Streaming Sortformer)" / "hypothesis_rttm",
        "pattern": "{id}_Model04.rttm",
    },
]


def run_benchmark():
    print("=" * 70)
    print("OPTIONAL SPEAKER DIARIZATION BENCHMARK — pyannote.metrics")
    print("=" * 70)
    print("Scoring configuration: collar=0.0s, skip_overlap=False, optimal Hungarian matching\n")

    results = {
        "scoring_config": {
            "library": "pyannote.metrics 4.2.0 (pyannote.core 6.1.0)",
            "collar": 0.0,
            "skip_overlap": False,
            "speaker_mapping": "Optimal Hungarian bipartite matching",
        },
        "recordings": {},
        "matrix": [],
        "valid_aggregates": {},
    }

    csv_rows = [
        "recording_id,system_id,system_name,evaluation_status,der_percent,jer_percent,miss_percent,false_alarm_percent,confusion_percent,ref_duration_sec,ref_speech_sec,hyp_speech_sec,notes"
    ]

    for rec in RECORDINGS:
        rec_id = rec["id"]
        ref_file = rec["ref_path"]
        ref_dict = load_rttm(str(ref_file))
        ref_ann = list(ref_dict.values())[0]
        ref_ann.uri = rec_id

        ref_timeline = ref_ann.get_timeline()
        ref_speech_total = ref_timeline.duration()
        ref_extent = ref_timeline.extent()

        results["recordings"][rec_id] = {
            "source": rec["source"],
            "evidence": rec["evidence"],
            "is_valid": rec["is_valid"],
            "duration_sec": rec["duration"],
            "ref_path": str(ref_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "ref_turns": len(ref_ann),
            "ref_speakers": sorted(list(ref_ann.labels())),
            "ref_speech_sec": round(ref_speech_total, 3),
            "ref_extent_start": round(ref_extent.start, 3),
            "ref_extent_end": round(ref_extent.end, 3),
        }

        print(f"\nRecording: {rec_id} (Source: {rec['source']})")
        print(f"  Duration: {rec['duration']:.2f}s | Reference speech: {ref_speech_total:.2f}s | Status: {'VALID' if rec['is_valid'] else 'INVALID_REFERENCE'}")

        for sys_info in SYSTEMS:
            hyp_path = sys_info["hyp_dir"] / sys_info["pattern"].format(id=rec_id)
            if not hyp_path.is_file():
                print(f"  [{sys_info['key']}] MISSING_HYPOTHESIS at {hyp_path.name}")
                continue

            hyp_dict = load_rttm(str(hyp_path))
            hyp_ann = list(hyp_dict.values())[0]
            hyp_ann.uri = rec_id
            hyp_speech = hyp_ann.get_timeline().duration()

            if not rec["is_valid"]:
                status = "INVALID_REFERENCE"
                note = "Reference timestamps unaligned (+400s offset); excluded from official metrics"
                csv_rows.append(f"{rec_id},{sys_info['key']},{sys_info['name']},{status},,,,,, {rec['duration']:.2f},{ref_speech_total:.2f},{hyp_speech:.2f},{note}")
                results["matrix"].append({
                    "recording_id": rec_id,
                    "system_id": sys_info["key"],
                    "system_name": sys_info["name"],
                    "status": status,
                    "notes": note,
                    "hyp_speech_sec": round(hyp_speech, 3),
                    "der_percent": None,
                    "jer_percent": None,
                })
                print(f"  [{sys_info['key']}] EXCLUDED from official benchmark ({status})")
                continue

            status = "VALID"
            der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
            der_val = float(der_metric(ref_ann, hyp_ann))
            der_comps = dict(der_metric.accumulated_)

            jer_metric = JaccardErrorRate(collar=0.0, skip_overlap=False)
            jer_val = float(jer_metric(ref_ann, hyp_ann))

            tot = der_comps.get("total", 0.0)
            miss = der_comps.get("missed detection", 0.0)
            fa = der_comps.get("false alarm", 0.0)
            conf = der_comps.get("confusion", 0.0)

            der_pct = round(der_val * 100.0, 2)
            jer_pct = round(jer_val * 100.0, 2)
            miss_pct = round(miss / tot * 100.0, 2) if tot > 0 else 0.0
            fa_pct = round(fa / tot * 100.0, 2) if tot > 0 else 0.0
            conf_pct = round(conf / tot * 100.0, 2) if tot > 0 else 0.0

            note = f"Verified {rec['source']} recording"
            csv_rows.append(f"{rec_id},{sys_info['key']},{sys_info['name']},{status},{der_pct},{jer_pct},{miss_pct},{fa_pct},{conf_pct},{rec['duration']:.2f},{ref_speech_total:.2f},{hyp_speech:.2f},{note}")

            results["matrix"].append({
                "recording_id": rec_id,
                "system_id": sys_info["key"],
                "system_name": sys_info["name"],
                "status": status,
                "der_percent": der_pct,
                "jer_percent": jer_pct,
                "components": {
                    "total_ref_speech_sec": round(tot, 3),
                    "miss_sec": round(miss, 3),
                    "fa_sec": round(fa, 3),
                    "confusion_sec": round(conf, 3),
                    "miss_percent": miss_pct,
                    "fa_percent": fa_pct,
                    "confusion_percent": conf_pct,
                },
                "hyp_speech_sec": round(hyp_speech, 3),
            })
            print(f"  [{sys_info['key']}] DER: {der_pct}% (Miss: {miss_pct}%, FA: {fa_pct}%, Conf: {conf_pct}%) | JER: {jer_pct}%")

    # Aggregates across valid pairs only
    for sys_info in SYSTEMS:
        skey = sys_info["key"]
        valid_entries = [e for e in results["matrix"] if e["system_id"] == skey and e["status"] == "VALID"]
        macro_der = round(sum(e["der_percent"] for e in valid_entries) / len(valid_entries), 2)
        macro_jer = round(sum(e["jer_percent"] for e in valid_entries) / len(valid_entries), 2)

        tot_speech = sum(e["components"]["total_ref_speech_sec"] for e in valid_entries)
        tot_err = sum(e["components"]["miss_sec"] + e["components"]["fa_sec"] + e["components"]["confusion_sec"] for e in valid_entries)
        micro_der = round(tot_err / tot_speech * 100.0, 2) if tot_speech > 0 else 0.0

        results["valid_aggregates"][skey] = {
            "system_name": sys_info["name"],
            "valid_pair_count": len(valid_entries),
            "macro_avg_der_percent": macro_der,
            "macro_avg_jer_percent": macro_jer,
            "micro_avg_der_percent": micro_der,
        }

    # Save outputs exclusively to canonical directory
    csv_path = BENCHMARK_DIR / "diarization_scores.csv"
    csv_path.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    print(f"\n[OK] Saved {csv_path}")

    json_path = BENCHMARK_DIR / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved {json_path}")

    print("\n" + "=" * 70)
    print("VALID BENCHMARK AGGREGATE RANKING")
    print("=" * 70)
    for skey, agg in results["valid_aggregates"].items():
        print(f"  {agg['system_name']}: Macro DER = {agg['macro_avg_der_percent']}% | Macro JER = {agg['macro_avg_jer_percent']}% | Micro DER = {agg['micro_avg_der_percent']}%")


if __name__ == "__main__":
    run_benchmark()
