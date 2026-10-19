import json
from pathlib import Path
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

ROOT = Path(__file__).resolve().parent.parent

RECORDINGS = [
    {
        "id": "liveTranscription_input02",
        "audio_path": ROOT / "Audio and annotations" / "liveTranscription_inputs_audio" / "liveTranscription_input02.wav",
        "ref_path": ROOT / "Audio and annotations" / "liveTranscription_inputs_rttm" / "liveTranscription_input02.rttm",
        "duration": 139.91,
        "claimed_source": "VoxConverse",
        "verified_corpus_evidence": "Reference URI is 'zvmyn', which matches VoxConverse clip ID 'zvmyn.wav'. Single speaker dialogue excerpt.",
    },
    {
        "id": "call_input01",
        "audio_path": ROOT / "Audio and annotations" / "call_inputs_audio" / "call_input01.sph",
        "ref_path": ROOT / "Audio and annotations" / "call_inputs_rttm" / "call_input01.rttm",
        "duration": 120.00,
        "claimed_source": "CALLHOME",
        "verified_corpus_evidence": "Reference URI is 'LDC97S42', which is the Linguistic Data Consortium CALLHOME American English Speech catalog number.",
    },
    {
        "id": "call_input02",
        "audio_path": ROOT / "Audio and annotations" / "call_inputs_audio" / "call_input02.flac",
        "ref_path": ROOT / "Audio and annotations" / "call_inputs_rttm" / "call_input02.rttm",
        "duration": 120.00,
        "claimed_source": "CALLHOME",
        "verified_corpus_evidence": "Reference URI is 'en_5551', which is a recognized CALLHOME English conversation identifier.",
    },
]

MODELS = [
    {
        "key": "Model01",
        "display_name": "Model 01: pyannote Community-1",
        "hyp_dir": ROOT / "Model1(pyannote Community-1)" / "hypothesis_rttm",
        "filename_pattern": "{id}_Model01.rttm",
        "provenance": "pyannote/speaker-diarization-community-1",
        "library_version": "pyannote.audio 4.0.7",
        "python_version": "3.11.9",
        "settings": "16 kHz mono resampling via soundfile, pipeline default threshold and clustering",
    },
    {
        "key": "Model02",
        "display_name": "Model 02: NVIDIA NeMo Sortformer (Offline)",
        "hyp_dir": ROOT / "Model2(NVIDIA NeMo Sortformer)" / "hypothesis_rttm",
        "filename_pattern": "{id}_Model02.rttm",
        "provenance": "nvidia/diar_sortformer_4spk-v1",
        "library_version": "NeMo 3.0.0",
        "python_version": "3.11.9",
        "settings": "batch_size=1, Lhotse dataloader, SafeTemporaryDirectory monkeypatch for Windows file lock",
    },
    {
        "key": "Model04",
        "display_name": "Model 04: NVIDIA NeMo Streaming Sortformer",
        "hyp_dir": ROOT / "Model4(NVIDIA NeMo Streaming Sortformer)" / "hypothesis_rttm",
        "filename_pattern": "{id}_Model04.rttm",
        "provenance": "nvidia/diar_streaming_sortformer_4spk-v2.1",
        "library_version": "NeMo 3.0.0",
        "python_version": "3.11.9",
        "settings": "streaming buffer size default, batch_size=1, streaming_sortformer_to_rttm converter",
    },
]

def evaluate():
    results = {
        "scoring_config": {
            "metric_library": "pyannote.metrics 4.2.0 (pyannote.core 6.1.0)",
            "collar": 0.0,
            "skip_overlap": False,
            "speaker_mapping": "Optimal Hungarian bipartite matching (pyannote.metrics default)",
        },
        "recordings": {},
        "matrix": [],
        "model_summaries": {},
    }

    print("================================================================================")
    print("RUNNING BENCHMARK SCORING VIA pyannote.metrics (collar=0.0, skip_overlap=False)")
    print("================================================================================\n")

    for rec_info in RECORDINGS:
        rec_id = rec_info["id"]
        ref_file = rec_info["ref_path"]

        ref_dict = load_rttm(str(ref_file))
        ref_ann = list(ref_dict.values())[0]
        # Normalize URI for metric alignment
        ref_ann.uri = rec_id

        ref_timeline = ref_ann.get_timeline()
        ref_speech_total = ref_timeline.duration()
        ref_extent = ref_timeline.extent()
        ref_speakers = sorted(list(ref_ann.labels()))

        results["recordings"][rec_id] = {
            "claimed_source": rec_info["claimed_source"],
            "verified_corpus_evidence": rec_info["verified_corpus_evidence"],
            "audio_duration_sec": rec_info["duration"],
            "ref_path": str(ref_file.relative_to(ROOT)).replace("\\", "/"),
            "ref_turns": len(ref_ann),
            "ref_speakers": ref_speakers,
            "ref_speech_total_sec": round(ref_speech_total, 3),
            "ref_start_sec": round(ref_extent.start, 3),
            "ref_end_sec": round(ref_extent.end, 3),
        }

        print(f"--- Recording: {rec_id} ---")
        print(f"  Source: {rec_info['claimed_source']} (Evidence: {rec_info['verified_corpus_evidence']})")
        print(f"  Duration: {rec_info['duration']:.2f}s | Reference speech: {ref_speech_total:.2f}s | Speakers ({len(ref_speakers)}): {ref_speakers}")

        for model in MODELS:
            hyp_path = model["hyp_dir"] / model["filename_pattern"].format(id=rec_id)
            pair_key = f"{rec_id}__{model['key']}"

            if not hyp_path.is_file():
                print(f"  [{model['key']}] MISSING_HYPOTHESIS at {hyp_path}")
                results["matrix"].append({
                    "recording_id": rec_id,
                    "model_key": model["key"],
                    "model_name": model["display_name"],
                    "status": "MISSING_HYPOTHESIS",
                    "hyp_path": str(hyp_path),
                })
                continue

            hyp_dict = load_rttm(str(hyp_path))
            if not hyp_dict:
                print(f"  [{model['key']}] EMPTY_HYPOTHESIS at {hyp_path}")
                results["matrix"].append({
                    "recording_id": rec_id,
                    "model_key": model["key"],
                    "model_name": model["display_name"],
                    "status": "EMPTY_HYPOTHESIS",
                    "hyp_path": str(hyp_path),
                })
                continue

            hyp_ann = list(hyp_dict.values())[0]
            hyp_ann.uri = rec_id
            hyp_timeline = hyp_ann.get_timeline()
            hyp_speech_total = hyp_timeline.duration()
            hyp_extent = hyp_timeline.extent()
            hyp_speakers = sorted(list(hyp_ann.labels()))

            # DiarizationErrorRate (collar=0.0, skip_overlap=False)
            der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
            der_val = float(der_metric(ref_ann, hyp_ann))
            der_comps = dict(der_metric.accumulated_)

            # JaccardErrorRate (collar=0.0, skip_overlap=False)
            jer_metric = JaccardErrorRate(collar=0.0, skip_overlap=False)
            jer_val = float(jer_metric(ref_ann, hyp_ann))
            jer_comps = dict(jer_metric.accumulated_)

            total_speech = der_comps.get("total", 0.0)
            miss = der_comps.get("missed detection", 0.0)
            fa = der_comps.get("false alarm", 0.0)
            confusion = der_comps.get("confusion", 0.0)
            correct = der_comps.get("correct", 0.0)

            miss_rate = (miss / total_speech * 100.0) if total_speech > 0 else 0.0
            fa_rate = (fa / total_speech * 100.0) if total_speech > 0 else 0.0
            conf_rate = (confusion / total_speech * 100.0) if total_speech > 0 else 0.0

            entry = {
                "recording_id": rec_id,
                "model_key": model["key"],
                "model_name": model["display_name"],
                "status": "VALID",
                "hyp_path": str(hyp_path.relative_to(ROOT)).replace("\\", "/"),
                "hyp_turns": len(hyp_ann),
                "hyp_speakers": hyp_speakers,
                "hyp_speech_total_sec": round(hyp_speech_total, 3),
                "hyp_start_sec": round(hyp_extent.start, 3),
                "hyp_end_sec": round(hyp_extent.end, 3),
                "der_percent": round(der_val * 100.0, 2),
                "jer_percent": round(jer_val * 100.0, 2),
                "der_raw": der_val,
                "jer_raw": jer_val,
                "components": {
                    "total_ref_speech_sec": round(total_speech, 3),
                    "correct_sec": round(correct, 3),
                    "miss_sec": round(miss, 3),
                    "false_alarm_sec": round(fa, 3),
                    "confusion_sec": round(confusion, 3),
                    "miss_rate_percent": round(miss_rate, 2),
                    "false_alarm_rate_percent": round(fa_rate, 2),
                    "confusion_rate_percent": round(conf_rate, 2),
                },
                "jer_components": {
                    "speaker_count": jer_comps.get("speaker count", None),
                    "speaker_error": jer_comps.get("speaker error", None),
                }
            }
            results["matrix"].append(entry)

            print(f"  [{model['key']}] DER: {entry['der_percent']:.2f}% (Miss: {entry['components']['miss_rate_percent']:.2f}%, FA: {entry['components']['false_alarm_rate_percent']:.2f}%, Conf: {entry['components']['confusion_rate_percent']:.2f}%) | JER: {entry['jer_percent']:.2f}%")

    # Aggregate summaries per model across the 3 recordings
    for model in MODELS:
        mkey = model["key"]
        entries = [e for e in results["matrix"] if e["model_key"] == mkey and e["status"] == "VALID"]
        if not entries:
            results["model_summaries"][mkey] = {"status": "NO_VALID_DATA"}
            continue

        macro_der = sum(e["der_percent"] for e in entries) / len(entries)
        macro_jer = sum(e["jer_percent"] for e in entries) / len(entries)

        total_ref_speech = sum(e["components"]["total_ref_speech_sec"] for e in entries)
        total_miss = sum(e["components"]["miss_sec"] for e in entries)
        total_fa = sum(e["components"]["false_alarm_sec"] for e in entries)
        total_conf = sum(e["components"]["confusion_sec"] for e in entries)
        total_err = total_miss + total_fa + total_conf
        micro_der = (total_err / total_ref_speech * 100.0) if total_ref_speech > 0 else 0.0

        results["model_summaries"][mkey] = {
            "model_name": model["display_name"],
            "evaluated_pairs": len(entries),
            "macro_avg_der_percent": round(macro_der, 2),
            "macro_avg_jer_percent": round(macro_jer, 2),
            "micro_avg_der_percent": round(micro_der, 2),
            "cumulative_ref_speech_sec": round(total_ref_speech, 3),
            "cumulative_miss_sec": round(total_miss, 3),
            "cumulative_fa_sec": round(total_fa, 3),
            "cumulative_conf_sec": round(total_conf, 3),
        }

    # Save results json
    out_dir = Path(__file__).resolve().parent / "experiments" / "results" / "optional_diarization_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {json_path}")

if __name__ == "__main__":
    evaluate()
