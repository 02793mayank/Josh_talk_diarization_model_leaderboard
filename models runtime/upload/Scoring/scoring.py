"""
Main Scoring Orchestrator for Josh Talks Voice Arena.
Executes end-to-end evaluation, input validation, metric computation,
customer-specific weighting, ranking, and report generation.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Ensure parent directory is in path so Scoring can be run from anywhere
SCORING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCORING_DIR.parent
if str(SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_DIR))

import scoring_config as cfg
import scoring_utils as utils

# Re-export key utilities for diagnostic and external script compatibility
parse_rttm = utils.parse_rttm
read_rttm = utils.parse_rttm
evaluate_pair = utils.evaluate_pair
get_audio_duration = utils.get_audio_duration


def run_scoring(
    project_root: Path = cfg.PROJECT_ROOT,
    output_dir: Path = cfg.DEFAULT_OUTPUT_DIR,
    validate_only: bool = False,
    collar: float = cfg.DEFAULT_COLLAR_SECONDS,
    skip_overlap: bool = cfg.DEFAULT_SKIP_OVERLAP,
    enable_provisional_alignment: bool = False,
) -> int:
    """
    Main orchestration routine.
    Returns exit code (0 for success or expected partial run, 1 for fatal configuration error).
    """
    print("=" * 80)
    print("JOSH TALKS VOICE ARENA — SPEAKER DIARIZATION SCORING SYSTEM")
    print("=" * 80)
    print(f"Project Root: {project_root}")
    print(f"Output Directory: {output_dir}")
    align_mode_str = "ENABLED (-400s offset test)" if enable_provisional_alignment else "DISABLED (standard ground truth only)"
    print(f"Collar: {collar}s | Skip Overlap: {skip_overlap} | UEM: {cfg.DEFAULT_UEM_POLICY}")
    print(f"Provisional Reference Alignment: {align_mode_str}\n")

    # 1. Validate customer weights
    try:
        cfg.validate_customer_weights()
        print("[OK] Customer-specific weights validated successfully:")
        for cat, w in cfg.CUSTOMER_WEIGHTS.items():
            print(f"     - {cat:20s}: Missing={w['missing']:.2f}, FA={w['false_alarm']:.2f}, Conf={w['confusion']:.2f} (Sum=1.00)")
    except Exception as e:
        print(f"[FATAL CONFIG ERROR] {e}")
        return 1

    # 2. Discover model slots (1-10)
    model_slots = utils.discover_model_slots(project_root, max_slots=10)
    print(f"\n[OK] Model slot discovery (Slots 1 to 10):")
    for slot in model_slots:
        status_str = "ACTIVE DIRECTORY" if slot["is_present"] else "RESERVED SLOT (FUTURE)"
        hyp_str = "hypothesis_rttm available" if slot["has_hyp_dir"] else "no hypothesis_rttm"
        print(f"     Slot {slot['model_number']:2d}: {slot['model_name']:40s} [{status_str}, {hyp_str}]")

    # 3. Discover recordings & resolve audio + reference RTTMs
    print("\n" + "=" * 80)
    print("PHASE 4: INPUT VALIDATION MATRIX")
    print("=" * 80)

    recording_metadata: Dict[str, Dict[str, Any]] = {}
    validation_rows = []

    for category, rec_ids in cfg.RECORDINGS_BY_CATEGORY.items():
        audio_folder = cfg.CATEGORY_AUDIO_FOLDERS[category]
        rttm_folder = cfg.CATEGORY_RTTM_FOLDERS[category]

        for rec_id in rec_ids:
            audio_path = utils.find_audio_file(audio_folder, rec_id, cfg.SUPPORTED_AUDIO_EXTENSIONS)
            ref_overrides = cfg.PROVISIONAL_REFERENCE_MAP if enable_provisional_alignment else None
            ref_path = utils.find_reference_rttm(rttm_folder, rec_id, reference_overrides=ref_overrides)
            is_provisional = bool(enable_provisional_alignment and rec_id in cfg.PROVISIONAL_REFERENCE_MAP)

            duration = None
            if audio_path:
                try:
                    duration = utils.get_audio_duration(audio_path)
                except Exception as e:
                    utils.logger.log(
                        severity="ERROR",
                        category=category,
                        recording=rec_id,
                        model="",
                        error_type="AUDIO_DURATION_ERROR",
                        error_message=str(e),
                        reference_file=str(ref_path) if ref_path else "",
                    )

            ref_ann = None
            ref_err = None
            if ref_path:
                try:
                    ref_ann = utils.parse_rttm(ref_path, recording_id=rec_id)
                except Exception as e:
                    ref_err = str(e)
                    utils.logger.log(
                        severity="ERROR",
                        category=category,
                        recording=rec_id,
                        model="",
                        error_type="MALFORMED_REFERENCE_RTTM",
                        error_message=str(e),
                        reference_file=str(ref_path),
                    )

            recording_metadata[rec_id] = {
                "category": category,
                "audio_path": audio_path,
                "ref_path": ref_path,
                "duration": duration,
                "ref_annotation": ref_ann,
                "ref_error": ref_err,
                "is_provisional": is_provisional,
            }

            # Check hypothesis availability across slots 1-4 for matrix
            hyp_statuses = []
            for slot in model_slots[:4]:
                hpath, herr = utils.match_hypothesis_rttm(
                    slot["hypothesis_dir"], rec_id, slot["model_number"]
                )
                hyp_statuses.append("OK" if hpath else "MISSING")

            status_summary = "READY" if (audio_path and ref_ann and duration) else "INCOMPLETE_REF"
            prov_tag = " [PROVISIONAL_REF]" if is_provisional else ""
            print(f"[{category[:4].upper()}] {rec_id:28s} | Audio: {'YES' if audio_path else 'NO ':<3s} | Ref: {'YES' if ref_ann else 'NO ':<3s} | M1:{hyp_statuses[0]} M2:{hyp_statuses[1]} M3:{hyp_statuses[2]} M4:{hyp_statuses[3]} | Status: {status_summary}{prov_tag}")

    if validate_only:
        print("\n[VALIDATION-ONLY MODE] Input validation completed. Exiting without scoring.")
        return 0

    # 4. Evaluation Execution
    print("\n" + "=" * 80)
    print("PHASE 5: DIARIZATION EVALUATION & SCORING")
    print("=" * 80)

    category_reports_data: Dict[str, List[Dict[str, Any]]] = {
        "call": [],
        "live_transcription": [],
        "meeting": [],
    }

    # Track slot counters
    expected_active_comparisons = 20  # 5 recordings * 4 active models
    successful_comparisons = 0
    failed_comparisons = 0
    not_evaluated_slots = 30  # 5 recordings * 6 reserved models (Models 5-10)

    for slot in model_slots:
        slot_num = slot["model_number"]
        model_name = slot["model_name"]
        is_active = (slot_num <= 4 and slot["is_present"])

        for category, rec_ids in cfg.RECORDINGS_BY_CATEGORY.items():
            cat_weights = cfg.CUSTOMER_WEIGHTS[category]

            for rec_id in rec_ids:
                rec_meta = recording_metadata[rec_id]
                audio_path = rec_meta["audio_path"]
                ref_path = rec_meta["ref_path"]
                duration = rec_meta["duration"]
                ref_ann = rec_meta["ref_annotation"]

                row = {
                    "category": category,
                    "recording": rec_id,
                    "model": model_name,
                    "model_number": slot_num,
                    "status": "NOT_EVALUATED",
                    "rank": None,
                    "audio_file": str(audio_path.relative_to(project_root)).replace("\\", "/") if audio_path else "",
                    "reference_file": str(ref_path.relative_to(project_root)).replace("\\", "/") if ref_path else "",
                    "hypothesis_file": "",
                    "collar_seconds": collar,
                    "skip_overlap": skip_overlap,
                    "uem_source": ("PROVISIONAL_ALIGNED_SHIFT_400S" if rec_meta.get("is_provisional") else cfg.DEFAULT_UEM_POLICY) if duration else "",
                    "reference_speaker_seconds": None,
                    "correct_speaker_seconds": None,
                    "missed_speech_seconds": None,
                    "false_alarm_seconds": None,
                    "speaker_confusion_seconds": None,
                    "DER": None,
                    "JER": None,
                    "missed_speech_percent": None,
                    "false_alarm_percent": None,
                    "speaker_confusion_percent": None,
                    "weight_missing": cat_weights["missing"],
                    "weight_false_alarm": cat_weights["false_alarm"],
                    "weight_confusion": cat_weights["confusion"],
                    "custom_weighted_error": None,
                    "custom_score": None,
                    "reference_speaker_count": None,
                    "hypothesis_speaker_count": None,
                    "validation_status": "",
                    "error_details": "",
                    "is_valid": False,
                }

                if not is_active:
                    row["status"] = "NOT_EVALUATED"
                    row["validation_status"] = "RESERVED_MODEL_SLOT"
                    category_reports_data[category].append(row)
                    continue

                # For active models (1-4):
                if not audio_path or not ref_ann or duration is None:
                    row["status"] = "FAILED"
                    row["validation_status"] = "INVALID_REFERENCE_OR_AUDIO"
                    row["error_details"] = "Missing audio file, missing reference RTTM, or unknown duration"
                    failed_comparisons += 1
                    utils.logger.log(
                        severity="ERROR",
                        category=category,
                        recording=rec_id,
                        model=model_name,
                        error_type="MISSING_INPUT",
                        error_message=row["error_details"],
                    )
                    category_reports_data[category].append(row)
                    continue

                hyp_path, hyp_err = utils.match_hypothesis_rttm(
                    slot["hypothesis_dir"], rec_id, slot_num
                )
                if not hyp_path:
                    row["status"] = "FAILED"
                    row["validation_status"] = "MISSING_HYPOTHESIS"
                    row["error_details"] = hyp_err or "No hypothesis file"
                    failed_comparisons += 1
                    utils.logger.log(
                        severity="ERROR",
                        category=category,
                        recording=rec_id,
                        model=model_name,
                        error_type="MISSING_HYPOTHESIS",
                        error_message=row["error_details"],
                        reference_file=str(ref_path),
                    )
                    category_reports_data[category].append(row)
                    continue

                row["hypothesis_file"] = str(hyp_path.relative_to(project_root)).replace("\\", "/")

                # Parse hypothesis RTTM
                try:
                    hyp_ann = utils.parse_rttm(hyp_path, recording_id=rec_id)
                except Exception as e:
                    row["status"] = "FAILED"
                    row["validation_status"] = "MALFORMED_HYPOTHESIS"
                    row["error_details"] = str(e)
                    failed_comparisons += 1
                    utils.logger.log(
                        severity="ERROR",
                        category=category,
                        recording=rec_id,
                        model=model_name,
                        error_type="MALFORMED_HYPOTHESIS_RTTM",
                        error_message=str(e),
                        reference_file=str(ref_path),
                        hypothesis_file=str(hyp_path),
                    )
                    category_reports_data[category].append(row)
                    continue

                # Run metric evaluation
                eval_res = utils.evaluate_pair(
                    reference_ann=ref_ann,
                    hypothesis_ann=hyp_ann,
                    audio_duration=duration,
                    collar=collar,
                    skip_overlap=skip_overlap,
                    weights=cat_weights,
                )

                row.update(eval_res)
                if eval_res["is_valid"]:
                    successful_comparisons += 1
                    if rec_meta.get("is_provisional"):
                        row["status"] = "PROVISIONAL_ALIGNED"
                        row["validation_status"] = "PROVISIONAL_ALIGNED_SHIFT_400S"
                        row["error_details"] = "Provisional reference alignment active (-400.00s shift)"
                        print(f"  [PROVISIONAL] {model_name[:15]} x {rec_id:25s} -> DER: {eval_res['DER']:.2f}% | JER: {eval_res['JER']:.2f}% | Custom Score: {eval_res['custom_score']:.2f}")
                    else:
                        row["status"] = "SUCCESS"
                        row["validation_status"] = "VALID"
                        print(f"  [SUCCESS] {model_name[:15]} x {rec_id:25s} -> DER: {eval_res['DER']:.2f}% | JER: {eval_res['JER']:.2f}% | Custom Score: {eval_res['custom_score']:.2f}")
                else:
                    failed_comparisons += 1
                    row["status"] = "INVALID"
                    row["validation_status"] = eval_res["status"]
                    utils.logger.log(
                        severity="WARNING",
                        category=category,
                        recording=rec_id,
                        model=model_name,
                        error_type=eval_res["status"],
                        error_message=eval_res["error_details"],
                        reference_file=str(ref_path),
                        hypothesis_file=str(hyp_path),
                    )
                    print(f"  [INVALID] {model_name[:15]} x {rec_id:25s} -> {eval_res['error_details']}")

                category_reports_data[category].append(row)

    # 5. Assign Category Ranks
    for cat, rows in category_reports_data.items():
        utils.assign_ranks(rows, score_key="custom_score", rank_key="rank", higher_is_better=True)

    # 6. Overall Model Aggregations and Ranking
    overall_rows = []
    for slot in model_slots:
        slot_num = slot["model_number"]
        model_name = slot["model_name"]

        model_cat_entries = {
            cat: [r for r in category_reports_data[cat] if r["model_number"] == slot_num]
            for cat in cfg.RECORDINGS_BY_CATEGORY
        }

        agg = utils.compute_overall_model_aggregates(
            all_category_entries=model_cat_entries,
            customer_weights=cfg.CUSTOMER_WEIGHTS,
            expected_recordings_count=len(cfg.ALL_EXPECTED_RECORDINGS),
        )

        agg["model"] = model_name
        agg["model_number"] = slot_num
        agg["overall_rank"] = None
        overall_rows.append(agg)

    utils.assign_ranks(overall_rows, score_key="custom_score", rank_key="overall_rank", higher_is_better=True)

    # Sort overall table: ranked models first by rank, then unranked by model_number
    overall_rows.sort(
        key=lambda x: (x["overall_rank"] is None, x["overall_rank"] or 0, x["model_number"])
    )

    # 7. Write CSV Reports
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv_report(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for r in rows:
                row_vals = [utils.format_csv_value(r.get(col, "")) for col in fieldnames]
                writer.writerow(row_vals)

    call_report_path = output_dir / cfg.OUTPUT_FILES["call_type_report"]
    live_report_path = output_dir / cfg.OUTPUT_FILES["live_transcription_report"]
    meet_report_path = output_dir / cfg.OUTPUT_FILES["meeting_report"]
    overall_report_path = output_dir / cfg.OUTPUT_FILES["overall_model_report"]
    error_report_path = output_dir / cfg.OUTPUT_FILES["evaluation_errors"]
    summary_path = output_dir / cfg.OUTPUT_FILES["run_summary"]

    write_csv_report(call_report_path, category_reports_data["call"], cfg.CUSTOMER_REPORT_COLUMNS)
    write_csv_report(live_report_path, category_reports_data["live_transcription"], cfg.CUSTOMER_REPORT_COLUMNS)
    write_csv_report(meet_report_path, category_reports_data["meeting"], cfg.CUSTOMER_REPORT_COLUMNS)
    write_csv_report(overall_report_path, overall_rows, cfg.OVERALL_REPORT_COLUMNS)
    utils.logger.write_csv(error_report_path, cfg.ERROR_REPORT_COLUMNS)

    # 8. Write Run Summary JSON
    completion_status = "SUCCESS" if (successful_comparisons == expected_active_comparisons) else "PARTIAL"
    if successful_comparisons == 0:
        completion_status = "FAILED"

    # Check installed package versions
    try:
        import pyannote.core
        core_ver = getattr(pyannote.core, "__version__", "unknown")
    except Exception:
        core_ver = "unknown"

    try:
        import pyannote.metrics
        metrics_ver = getattr(pyannote.metrics, "__version__", "unknown")
    except Exception:
        metrics_ver = "unknown"

    try:
        import soundfile
        sf_ver = getattr(soundfile, "__version__", "unknown")
    except Exception:
        sf_ver = "unknown"

    summary_data = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(project_root.resolve()).replace("\\", "/"),
        "scoring_version": "1.0.0",
        "python_version": sys.version.split()[0],
        "pyannote_core_version": str(core_ver),
        "pyannote_metrics_version": str(metrics_ver),
        "soundfile_version": str(sf_ver),
        "collar_seconds": collar,
        "skip_overlap": skip_overlap,
        "uem_policy": cfg.DEFAULT_UEM_POLICY,
        "provisional_alignment_enabled": enable_provisional_alignment,
        "customer_weights": cfg.CUSTOMER_WEIGHTS,
        "configured_model_count": len(model_slots),
        "configured_models": [s["model_name"] for s in model_slots],
        "expected_active_comparisons": expected_active_comparisons,
        "successful_comparisons": successful_comparisons,
        "failed_comparisons": failed_comparisons,
        "not_evaluated_slots": not_evaluated_slots,
        "completion_status": completion_status,
        "report_paths": {
            "call_type_report": str(call_report_path.relative_to(project_root)).replace("\\", "/"),
            "live_transcription_report": str(live_report_path.relative_to(project_root)).replace("\\", "/"),
            "meeting_report": str(meet_report_path.relative_to(project_root)).replace("\\", "/"),
            "overall_model_report": str(overall_report_path.relative_to(project_root)).replace("\\", "/"),
        },
        "error_report_path": str(error_report_path.relative_to(project_root)).replace("\\", "/"),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 9. Print Final Execution Summary
    print("\n" + "=" * 80)
    print("PHASE 6: FINAL EXECUTION SUMMARY & OVERALL RANKING")
    print("=" * 80)
    print(f"Completion Status: {completion_status}")
    print(f"Active Comparisons: {successful_comparisons} SUCCESS, {failed_comparisons} FAILED / INVALID (Expected: {expected_active_comparisons})")
    print(f"Reserved Model Slots: {not_evaluated_slots} (Slots 5-10)")
    print(f"Reports Directory: {output_dir}\n")

    print(f"{'Rank':<5} | {'Model':<35} | {'DER':<8} | {'JER':<8} | {'Miss %':<8} | {'FA %':<8} | {'Conf %':<8} | {'Custom Score':<12} | {'Status':<10}")
    print("-" * 115)
    for r in overall_rows:
        rnk = str(r["overall_rank"]) if r["overall_rank"] is not None else "-"
        der = f"{r['DER']:.2f}%" if r["DER"] is not None else "-"
        jer = f"{r['JER']:.2f}%" if r["JER"] is not None else "-"
        miss = f"{r['missed_speech_percent']:.2f}%" if r["missed_speech_percent"] is not None else "-"
        fa = f"{r['false_alarm_percent']:.2f}%" if r["false_alarm_percent"] is not None else "-"
        conf = f"{r['speaker_confusion_percent']:.2f}%" if r["speaker_confusion_percent"] is not None else "-"
        cs = f"{r['custom_score']:.2f}" if r["custom_score"] is not None else "-"
        st = r["completion_status"]
        print(f"{rnk:<5} | {r['model']:<35} | {der:<8} | {jer:<8} | {miss:<8} | {fa:<8} | {conf:<8} | {cs:<12} | {st:<10}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Josh Talks Voice Arena — Speaker Diarization Scoring Pipeline"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=cfg.PROJECT_ROOT,
        help="Path to project root (default: auto-detected)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.DEFAULT_OUTPUT_DIR,
        help="Directory to save CSV and JSON reports (default: scoring_output/)",
    )
    parser.add_argument(
        "--collar",
        type=float,
        default=cfg.DEFAULT_COLLAR_SECONDS,
        help="Evaluation collar in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--skip-overlap",
        action="store_true",
        default=cfg.DEFAULT_SKIP_OVERLAP,
        help="Exclude overlapping speech segments from scoring",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run input discovery and validation matrix without executing scoring",
    )
    parser.add_argument(
        "--enable-provisional-alignment",
        action="store_true",
        default=False,
        help="Enable provisional reference alignment for unaligned recordings (e.g. call_input02 shifted by -400s)",
    )

    args = parser.parse_args()
    exit_code = run_scoring(
        project_root=args.root,
        output_dir=args.output_dir,
        validate_only=args.validate_only,
        collar=args.collar,
        skip_overlap=args.skip_overlap,
        enable_provisional_alignment=args.enable_provisional_alignment,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
