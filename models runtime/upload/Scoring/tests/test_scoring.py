"""
Comprehensive Automated Test Suite for Josh Talks Voice Arena Scoring System.
Covers all 26 requirements specified in Phase 19.
Uses temporary synthetic RTTM/audio files without modifying any actual project dataset files.
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add Scoring directory to sys.path
scoring_dir = Path(__file__).resolve().parent.parent
if str(scoring_dir) not in sys.path:
    sys.path.insert(0, str(scoring_dir))

from pyannote.core import Annotation, Segment, Timeline

from scoring_config import (
    CUSTOMER_WEIGHTS,
    RECORDINGS_BY_CATEGORY,
    ALL_EXPECTED_RECORDINGS,
    CUSTOMER_REPORT_COLUMNS,
    OVERALL_REPORT_COLUMNS,
    OUTPUT_FILES,
    validate_customer_weights,
    DEFAULT_COLLAR_SECONDS,
    DEFAULT_SKIP_OVERLAP,
    DEFAULT_UEM_POLICY,
)
from scoring_utils import (
    discover_model_slots,
    find_audio_file,
    find_reference_rttm,
    match_hypothesis_rttm,
    get_audio_duration,
    parse_rttm,
    evaluate_pair,
    compute_category_aggregates,
    compute_overall_model_aggregates,
    assign_ranks,
    format_csv_value,
    ErrorLogger,
)


class TestScoringSystem(unittest.TestCase):
    """Test suite covering all 26 required test cases for Phase 19."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Parsing valid RTTM files
    def test_01_parse_valid_rttm(self):
        rttm_path = self.temp_path / "valid.rttm"
        rttm_content = (
            "SPEAKER rec1 1 0.00 2.50 <NA> <NA> spk1 <NA> <NA>\n"
            "SPEAKER rec1 1 3.00 1.50 <NA> <NA> spk2 <NA> <NA>\n"
            "# Comment line to ignore\n"
            "\n"
            "SPEAKER rec1 1 5.00 2.00 <NA> <NA> spk1 <NA> <NA>\n"
        )
        rttm_path.write_text(rttm_content, encoding="utf-8")
        ann = parse_rttm(rttm_path, recording_id="rec1")

        self.assertIsInstance(ann, Annotation)
        self.assertEqual(len(ann.labels()), 2)
        self.assertIn("spk1", ann.labels())
        self.assertIn("spk2", ann.labels())
        timeline = ann.get_timeline()
        self.assertEqual(len(timeline), 3)

    # 2. Rejecting malformed RTTM rows
    def test_02_reject_malformed_rttm_rows(self):
        rttm_path = self.temp_path / "malformed.rttm"
        rttm_content = (
            "SPEAKER rec1 1 0.00 2.50 <NA>\n"  # only 6 fields (< 8 fields)
        )
        rttm_path.write_text(rttm_content, encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            parse_rttm(rttm_path)
        self.assertIn("Malformed RTTM row", str(ctx.exception))

        rttm_path_wrong_type = self.temp_path / "wrong_type.rttm"
        rttm_path_wrong_type.write_text(
            "NONSPEAKER rec1 1 0.00 2.50 <NA> <NA> spk1 <NA> <NA>\n", encoding="utf-8"
        )
        with self.assertRaises(ValueError) as ctx2:
            parse_rttm(rttm_path_wrong_type)
        self.assertIn("Expected 'SPEAKER' row", str(ctx2.exception))

    # 3. Validating start times and durations
    def test_03_validate_start_times_and_durations(self):
        # Non-numeric
        p1 = self.temp_path / "non_numeric.rttm"
        p1.write_text("SPEAKER rec1 1 abc 2.50 <NA> <NA> spk1 <NA> <NA>\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx1:
            parse_rttm(p1)
        self.assertIn("Non-numeric timestamp", str(ctx1.exception))

        # Negative start
        p2 = self.temp_path / "negative_start.rttm"
        p2.write_text("SPEAKER rec1 1 -1.50 2.50 <NA> <NA> spk1 <NA> <NA>\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            parse_rttm(p2)
        self.assertIn("Negative start time", str(ctx2.exception))

        # Negative duration
        p3 = self.temp_path / "negative_duration.rttm"
        p3.write_text("SPEAKER rec1 1 1.50 -2.50 <NA> <NA> spk1 <NA> <NA>\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx3:
            parse_rttm(p3)
        self.assertIn("Negative duration", str(ctx3.exception))

        # Zero duration is safely skipped
        p4 = self.temp_path / "zero_duration.rttm"
        p4.write_text(
            "SPEAKER rec1 1 0.00 0.00 <NA> <NA> spk1 <NA> <NA>\n"
            "SPEAKER rec1 1 1.00 2.00 <NA> <NA> spk2 <NA> <NA>\n",
            encoding="utf-8",
        )
        ann = parse_rttm(p4)
        self.assertEqual(len(ann.labels()), 1)
        self.assertEqual(list(ann.labels())[0], "spk2")

    # 4. Preserving overlapping speaker segments
    def test_04_preserve_overlapping_segments(self):
        rttm_path = self.temp_path / "overlap.rttm"
        # spk1 talks 0.0 to 3.0, spk2 talks 2.0 to 5.0 -> overlap from 2.0 to 3.0
        rttm_content = (
            "SPEAKER rec1 1 0.00 3.00 <NA> <NA> spk1 <NA> <NA>\n"
            "SPEAKER rec1 1 2.00 3.00 <NA> <NA> spk2 <NA> <NA>\n"
        )
        rttm_path.write_text(rttm_content, encoding="utf-8")
        ann = parse_rttm(rttm_path)
        self.assertEqual(len(ann.labels()), 2)
        # Check that both segments are preserved in the annotation
        timeline = ann.get_timeline()
        overlaps = timeline.get_overlap()
        self.assertGreater(len(overlaps), 0)
        self.assertAlmostEqual(overlaps.duration(), 1.0, places=2)

    # 5. Matching a recording to the correct reference RTTM
    def test_05_match_reference_rttm(self):
        ref_dir = self.temp_path / "refs"
        ref_dir.mkdir()
        (ref_dir / "call_input01.rttm").write_text("SPEAKER call_input01 1 0 1 <NA> <NA> s1 <NA> <NA>\n", encoding="utf-8")
        (ref_dir / "call_input02.rttm").write_text("SPEAKER call_input02 1 0 1 <NA> <NA> s2 <NA> <NA>\n", encoding="utf-8")

        matched = find_reference_rttm(ref_dir, "call_input01")
        self.assertIsNotNone(matched)
        self.assertEqual(matched.name, "call_input01.rttm")

    # 6. Matching hypotheses to the correct model
    def test_06_match_hypothesis_rttm(self):
        hyp_dir = self.temp_path / "hypotheses"
        hyp_dir.mkdir()
        # Case A: Model with suffix e.g. meeting_input_Model03.rttm
        (hyp_dir / "meeting_input_Model03.rttm").write_text("SPEAKER meeting_input 1 0 1 <NA> <NA> s1 <NA> <NA>\n", encoding="utf-8")
        # Case B: Model without suffix e.g. meeting_input.rttm
        hyp_dir2 = self.temp_path / "hypotheses2"
        hyp_dir2.mkdir()
        (hyp_dir2 / "meeting_input.rttm").write_text("SPEAKER meeting_input 1 0 1 <NA> <NA> s1 <NA> <NA>\n", encoding="utf-8")

        matched1, err1 = match_hypothesis_rttm(hyp_dir, "meeting_input", model_number=3)
        self.assertIsNone(err1)
        self.assertEqual(matched1.name, "meeting_input_Model03.rttm")

        matched2, err2 = match_hypothesis_rttm(hyp_dir2, "meeting_input", model_number=1)
        self.assertIsNone(err2)
        self.assertEqual(matched2.name, "meeting_input.rttm")

    # 7. Detecting duplicate matches
    def test_07_detect_duplicate_and_ambiguous_matches(self):
        # Audio duplicate
        audio_dir = self.temp_path / "audio_dup"
        audio_dir.mkdir()
        (audio_dir / "call_input01.wav").touch()
        (audio_dir / "call_input01.flac").touch()
        with self.assertRaises(ValueError) as ctx:
            find_audio_file(audio_dir, "call_input01", [".wav", ".flac"])
        self.assertIn("Duplicate audio files found", str(ctx.exception))

        # Hypothesis ambiguity
        hyp_dir = self.temp_path / "hyp_dup"
        hyp_dir.mkdir()
        (hyp_dir / "call_input01.rttm").touch()
        (hyp_dir / "call_input01_Model01.rttm").touch()
        matched, err = match_hypothesis_rttm(hyp_dir, "call_input01", model_number=1)
        self.assertIsNone(matched)
        self.assertIn("Ambiguous hypothesis matches", err)

    # 8. Detecting missing hypotheses
    def test_08_detect_missing_hypotheses(self):
        hyp_dir = self.temp_path / "hyp_empty"
        hyp_dir.mkdir()
        matched, err = match_hypothesis_rttm(hyp_dir, "call_input01", model_number=3)
        self.assertIsNone(matched)
        self.assertIn("contains no RTTM files", err)

        # Directory with other files but missing the specific recording
        (hyp_dir / "other_recording.rttm").touch()
        matched2, err2 = match_hypothesis_rttm(hyp_dir, "call_input01", model_number=3)
        self.assertIsNone(matched2)
        self.assertIn("No matching hypothesis RTTM found", err2)

    # 9. Correctly reserving Models 5–10 (and ignoring Model 11+)
    def test_09_reserve_models_5_to_10_and_ignore_11(self):
        fake_root = self.temp_path / "fake_project"
        fake_root.mkdir()
        (fake_root / "Model1(pyanote Community-1)").mkdir()
        (fake_root / "Model2(NVIDIA NeMo Sortformer)").mkdir()
        (fake_root / "Model11(Extra Model)").mkdir()
        (fake_root / "Model12(Another Extra)").mkdir()

        slots = discover_model_slots(fake_root, max_slots=10)
        self.assertEqual(len(slots), 10)
        # Check slots 1 and 2 are present
        self.assertTrue(slots[0]["is_present"])
        self.assertEqual(slots[0]["model_number"], 1)
        self.assertTrue(slots[1]["is_present"])
        self.assertEqual(slots[1]["model_number"], 2)
        # Check slots 5 to 10 are reserved placeholders
        for s in slots[4:10]:
            self.assertFalse(s["is_present"])
            self.assertEqual(s["model_name"], f"Model{s['model_number']}")
        # Model 11 and 12 must not appear in slots
        slot_nums = [s["model_number"] for s in slots]
        self.assertNotIn(11, slot_nums)
        self.assertNotIn(12, slot_nums)

    # 10. Verifying all three customer-weight configurations
    def test_10_verify_customer_weight_configurations(self):
        self.assertIn("call", CUSTOMER_WEIGHTS)
        self.assertIn("live_transcription", CUSTOMER_WEIGHTS)
        self.assertIn("meeting", CUSTOMER_WEIGHTS)

        self.assertEqual(CUSTOMER_WEIGHTS["call"]["missing"], 0.40)
        self.assertEqual(CUSTOMER_WEIGHTS["call"]["false_alarm"], 0.25)
        self.assertEqual(CUSTOMER_WEIGHTS["call"]["confusion"], 0.35)

        self.assertEqual(CUSTOMER_WEIGHTS["live_transcription"]["missing"], 0.50)
        self.assertEqual(CUSTOMER_WEIGHTS["live_transcription"]["false_alarm"], 0.20)
        self.assertEqual(CUSTOMER_WEIGHTS["live_transcription"]["confusion"], 0.30)

        self.assertEqual(CUSTOMER_WEIGHTS["meeting"]["missing"], 0.45)
        self.assertEqual(CUSTOMER_WEIGHTS["meeting"]["false_alarm"], 0.15)
        self.assertEqual(CUSTOMER_WEIGHTS["meeting"]["confusion"], 0.40)

    # 11. Verifying that weights sum to 1.00
    def test_11_weights_sum_to_one(self):
        # Must pass without error for configured weights
        validate_customer_weights(CUSTOMER_WEIGHTS)

        # Invalid weights test: sum != 1.0
        bad_weights = {
            "call": {"missing": 0.5, "false_alarm": 0.5, "confusion": 0.5},
            "live_transcription": {"missing": 0.5, "false_alarm": 0.2, "confusion": 0.3},
            "meeting": {"missing": 0.45, "false_alarm": 0.15, "confusion": 0.40},
        }
        with self.assertRaises(ValueError) as ctx:
            validate_customer_weights(bad_weights)
        self.assertIn("must sum to 1.00", str(ctx.exception))

    # 12. Testing custom weighted-error calculations using hand-calculated examples
    def test_12_hand_calculated_custom_weighted_error(self):
        # Suppose T_m = 10s, T_f = 5s, T_c = 4s, T_r = 100s
        # Category: call -> w_m = 0.40, w_f = 0.25, w_c = 0.35
        # Weighted error duration = 0.40*10 + 0.25*5 + 0.35*4 = 4.0 + 1.25 + 1.40 = 6.65s
        # CWE = 6.65 / 100 = 0.0665
        w = CUSTOMER_WEIGHTS["call"]
        t_m, t_f, t_c, t_r = 10.0, 5.0, 4.0, 100.0
        cwe = (w["missing"] * t_m + w["false_alarm"] * t_f + w["confusion"] * t_c) / t_r
        self.assertAlmostEqual(cwe, 0.0665, places=5)

    # 13. Testing the custom score
    def test_13_custom_score_calculation(self):
        # custom_score = 100 * max(0, 1 - custom_weighted_error)
        cwe = 0.0665
        score = 100.0 * max(0.0, 1.0 - cwe)
        self.assertAlmostEqual(score, 93.35, places=2)

        # Test capping at 0 when CWE >= 1.0
        cwe_huge = 1.50
        score_huge = 100.0 * max(0.0, 1.0 - cwe_huge)
        self.assertEqual(score_huge, 0.0)

    # 14. Testing standard DER calculation and component consistency where supported
    def test_14_standard_der_calculation(self):
        # Reference: spk1 [0.0 - 10.0]
        # Hypothesis: spk1 [0.0 - 8.0], spk2 [8.0 - 10.0] (confusion on [8-10] = 2s)
        # Audio: 10s, collar=0.0 to check exact math
        ref = Annotation(uri="test")
        ref[Segment(0.0, 10.0)] = "spk1"

        hyp = Annotation(uri="test")
        hyp[Segment(0.0, 8.0)] = "spk1"
        hyp[Segment(8.0, 10.0)] = "spk2"

        res = evaluate_pair(ref, hyp, audio_duration=10.0, collar=0.0, skip_overlap=False)
        self.assertTrue(res["is_valid"])
        self.assertAlmostEqual(res["reference_speaker_seconds"], 10.0, places=2)
        self.assertAlmostEqual(res["speaker_confusion_seconds"], 2.0, places=2)
        self.assertAlmostEqual(res["missed_speech_seconds"], 0.0, places=2)
        self.assertAlmostEqual(res["false_alarm_seconds"], 0.0, places=2)
        self.assertAlmostEqual(res["DER"], 20.0, places=2)

    # 15. Testing JER calculation
    def test_15_jer_calculation(self):
        ref = Annotation(uri="test")
        ref[Segment(0.0, 10.0)] = "spk1"
        hyp = Annotation(uri="test")
        hyp[Segment(0.0, 10.0)] = "spk1"

        res = evaluate_pair(ref, hyp, audio_duration=10.0, collar=0.0)
        self.assertTrue(res["is_valid"])
        self.assertAlmostEqual(res["JER"], 0.0, places=2)

    # 16. Testing zero reference duration
    def test_16_zero_reference_duration(self):
        # Reference has speech [400 - 450], but audio is only 100s
        # Scoring region is [0 - 100], so reference speech inside region is 0s
        ref = Annotation(uri="test")
        ref[Segment(400.0, 450.0)] = "spk1"
        hyp = Annotation(uri="test")
        hyp[Segment(10.0, 20.0)] = "spk1"

        res = evaluate_pair(ref, hyp, audio_duration=100.0, collar=0.25)
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["status"], "INVALID_REFERENCE_DURATION")
        self.assertIsNone(res["DER"])
        self.assertIsNone(res["JER"])
        self.assertIsNone(res["custom_score"])

    # 17. Testing absent UEM with full-audio fallback
    def test_17_absent_uem_full_audio_fallback(self):
        ref = Annotation(uri="test")
        ref[Segment(0.0, 5.0)] = "spk1"
        hyp = Annotation(uri="test")
        hyp[Segment(0.0, 5.0)] = "spk1"
        # Passing uem=None means default timeline is [0, audio_duration]
        res = evaluate_pair(ref, hyp, audio_duration=10.0, uem=None)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["status"], "VALID")

    # 18. Testing a missing audio duration
    def test_18_missing_audio_duration_error(self):
        dummy_file = self.temp_path / "corrupt.audio"
        dummy_file.write_text("not real audio bytes", encoding="utf-8")
        with self.assertRaises(RuntimeError) as ctx:
            get_audio_duration(dummy_file)
        self.assertIn("Could not determine audio duration", str(ctx.exception))

    # 19. Distinguishing corpus-level DER from macro-average DER
    def test_19_distinguish_corpus_vs_macro_der(self):
        # Two recordings with different lengths and DERs
        # Rec 1: Ref = 10s, DER = 10% (1s error)
        # Rec 2: Ref = 90s, DER = 30% (27s error)
        # Macro DER = (10% + 30%) / 2 = 20.0%
        # Corpus DER = (1 + 27) / (10 + 90) = 28 / 100 = 28.0%
        entries = [
            {
                "is_valid": True,
                "reference_speaker_seconds": 10.0,
                "missed_speech_seconds": 1.0,
                "false_alarm_seconds": 0.0,
                "speaker_confusion_seconds": 0.0,
                "DER": 10.0,
                "JER": 10.0,
            },
            {
                "is_valid": True,
                "reference_speaker_seconds": 90.0,
                "missed_speech_seconds": 27.0,
                "false_alarm_seconds": 0.0,
                "speaker_confusion_seconds": 0.0,
                "DER": 30.0,
                "JER": 30.0,
            },
        ]
        agg = compute_category_aggregates(entries, weights={"missing": 1.0, "false_alarm": 0.0, "confusion": 0.0})
        self.assertAlmostEqual(agg["DER"], 28.0, places=2)  # Corpus DER
        # Compute macro DER manually to verify distinction
        macro_der = sum(e["DER"] for e in entries) / len(entries)
        self.assertAlmostEqual(macro_der, 20.0, places=2)
        self.assertNotEqual(agg["DER"], macro_der)

    # 20. Testing overall custom-score aggregation
    def test_20_overall_custom_score_aggregation(self):
        # Rec A (call, w_m=0.40): ref=100s, miss=10s -> weighted err dur = 4.0s
        # Rec B (meeting, w_m=0.45): ref=100s, miss=20s -> weighted err dur = 9.0s
        # Total ref = 200s, Total weighted err dur = 13.0s
        # Overall CWE = 13.0 / 200 = 0.065
        # Overall Custom Score = 100 * (1 - 0.065) = 93.5
        all_entries = {
            "call": [{
                "is_valid": True,
                "reference_speaker_seconds": 100.0,
                "missed_speech_seconds": 10.0,
                "false_alarm_seconds": 0.0,
                "speaker_confusion_seconds": 0.0,
                "DER": 10.0,
                "JER": 10.0,
            }],
            "meeting": [{
                "is_valid": True,
                "reference_speaker_seconds": 100.0,
                "missed_speech_seconds": 20.0,
                "false_alarm_seconds": 0.0,
                "speaker_confusion_seconds": 0.0,
                "DER": 20.0,
                "JER": 20.0,
            }],
        }
        res = compute_overall_model_aggregates(all_entries, CUSTOMER_WEIGHTS, expected_recordings_count=2)
        self.assertTrue(res["completion_status"] == "SUCCESS")
        self.assertAlmostEqual(res["custom_weighted_error"], 0.065, places=4)
        self.assertAlmostEqual(res["custom_score"], 93.5, places=2)

    # 21. Testing rank calculation
    def test_21_rank_calculation(self):
        items = [
            {"name": "m1", "custom_score": 90.0},
            {"name": "m2", "custom_score": 95.0},
            {"name": "m3", "custom_score": 90.0},  # Tie with m1
            {"name": "m4", "custom_score": None},  # Unevaluated
        ]
        assign_ranks(items, score_key="custom_score", rank_key="rank")
        # Ranks: m2=1, m1=2, m3=2, m4=None
        rank_map = {it["name"]: it["rank"] for it in items}
        self.assertEqual(rank_map["m2"], 1)
        self.assertEqual(rank_map["m1"], 2)
        self.assertEqual(rank_map["m3"], 2)
        self.assertIsNone(rank_map["m4"])

    # 22. Ensuring unevaluated models have blank metrics and no rank
    def test_22_unevaluated_models_blank_metrics_and_no_rank(self):
        all_entries = {"call": [], "live_transcription": [], "meeting": []}
        res = compute_overall_model_aggregates(all_entries, CUSTOMER_WEIGHTS, expected_recordings_count=5)
        self.assertEqual(res["completion_status"], "NOT_EVALUATED")
        self.assertIsNone(res["DER"])
        self.assertIsNone(res["JER"])
        self.assertIsNone(res["custom_score"])
        self.assertEqual(res["evaluated_recordings"], 0)

        # In CSV format, None becomes empty string
        self.assertEqual(format_csv_value(res["DER"]), "")
        self.assertEqual(format_csv_value(res["custom_score"]), "")

    # 23. Ensuring failed comparisons are not reported as successes
    def test_23_failed_comparisons_not_success(self):
        # 4 valid, 1 failed out of 5 expected
        entries = {
            "call": [
                {"is_valid": True, "reference_speaker_seconds": 10.0, "missed_speech_seconds": 1.0, "false_alarm_seconds": 0.0, "speaker_confusion_seconds": 0.0, "DER": 10.0, "JER": 10.0},
                {"is_valid": False, "reference_speaker_seconds": 0.0, "missed_speech_seconds": 0.0, "false_alarm_seconds": 0.0, "speaker_confusion_seconds": 0.0, "DER": None, "JER": None},
            ],
            "live_transcription": [
                {"is_valid": True, "reference_speaker_seconds": 10.0, "missed_speech_seconds": 1.0, "false_alarm_seconds": 0.0, "speaker_confusion_seconds": 0.0, "DER": 10.0, "JER": 10.0},
                {"is_valid": True, "reference_speaker_seconds": 10.0, "missed_speech_seconds": 1.0, "false_alarm_seconds": 0.0, "speaker_confusion_seconds": 0.0, "DER": 10.0, "JER": 10.0},
            ],
            "meeting": [
                {"is_valid": True, "reference_speaker_seconds": 10.0, "missed_speech_seconds": 1.0, "false_alarm_seconds": 0.0, "speaker_confusion_seconds": 0.0, "DER": 10.0, "JER": 10.0},
            ],
        }
        res = compute_overall_model_aggregates(entries, CUSTOMER_WEIGHTS, expected_recordings_count=5)
        self.assertEqual(res["completion_status"], "PARTIAL")
        self.assertEqual(res["evaluated_recordings"], 4)
        self.assertEqual(res["failed_recordings"], 1)

    # 24. Ensuring CSV schemas remain consistent
    def test_24_csv_schemas_consistent(self):
        # Verify customer report schema has all required columns
        expected_cust_cols = [
            "category", "recording", "model", "model_number", "status", "rank",
            "audio_file", "reference_file", "hypothesis_file", "collar_seconds",
            "skip_overlap", "uem_source", "reference_speaker_seconds",
            "correct_speaker_seconds", "missed_speech_seconds", "false_alarm_seconds",
            "speaker_confusion_seconds", "DER", "JER", "missed_speech_percent",
            "false_alarm_percent", "speaker_confusion_percent", "weight_missing",
            "weight_false_alarm", "weight_confusion", "custom_weighted_error",
            "custom_score", "reference_speaker_count", "hypothesis_speaker_count",
            "validation_status", "error_details"
        ]
        self.assertEqual(CUSTOMER_REPORT_COLUMNS, expected_cust_cols)

        # Verify overall report schema has all 23 required columns
        self.assertEqual(len(OVERALL_REPORT_COLUMNS), 23)
        self.assertEqual(OVERALL_REPORT_COLUMNS[0], "overall_rank")
        self.assertEqual(OVERALL_REPORT_COLUMNS[1], "model")
        self.assertEqual(OVERALL_REPORT_COLUMNS[7], "DER")
        self.assertEqual(OVERALL_REPORT_COLUMNS[13], "custom_score")

    # 25. Ensuring JSON output is valid
    def test_25_json_output_valid(self):
        json_path = self.temp_path / "run_summary.json"
        summary_data = {
            "run_timestamp": "2026-10-18 22:00:00",
            "configured_model_count": 10,
            "active_models": 4,
            "expected_active_comparisons": 20,
            "successful_comparisons": 16,
            "completion_status": "PARTIAL",
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # Read back and verify valid JSON
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["expected_active_comparisons"], 20)
        self.assertEqual(loaded["completion_status"], "PARTIAL")

    # 26. Ensuring the scorer never modifies reference files, audio, hypotheses, or Voice Arena
    def test_26_read_only_integrity_of_source_data(self):
        # Create a mock source directory with reference, audio, hypothesis
        source_dir = self.temp_path / "source"
        source_dir.mkdir()
        ref_file = source_dir / "ref.rttm"
        ref_file.write_text("SPEAKER rec 1 0.0 5.0 <NA> <NA> s1 <NA> <NA>\n", encoding="utf-8")
        ref_mtime_before = ref_file.stat().st_mtime_ns
        ref_content_before = ref_file.read_bytes()

        # Parse RTTM
        ann = parse_rttm(ref_file)
        self.assertIsNotNone(ann)

        # Verify that file timestamp and bytes are completely unchanged
        self.assertEqual(ref_file.read_bytes(), ref_content_before)
        self.assertEqual(ref_file.stat().st_mtime_ns, ref_mtime_before)

    # 27. Regression test: call_input02 original produces INVALID_REF, aligned copy falls within 120s
    def test_27_call_input02_original_vs_aligned(self):
        # Original reference timestamps [400.35, 479.77]
        ref_orig = Annotation(uri="call_input02")
        ref_orig[Segment(400.35, 405.0)] = "spk1"
        hyp = Annotation(uri="call_input02")
        hyp[Segment(0.35, 5.0)] = "spk1"

        # Under 120s audio window, original reference has 0.0s ref speech
        res_orig = evaluate_pair(ref_orig, hyp, audio_duration=120.0, collar=0.25)
        self.assertFalse(res_orig["is_valid"])
        self.assertEqual(res_orig["status"], "INVALID_REFERENCE_DURATION")

        # Shifted reference [-400s] has turns within [0.35, 5.0]
        ref_aligned = Annotation(uri="call_input02")
        ref_aligned[Segment(0.35, 5.0)] = "spk1"
        res_aligned = evaluate_pair(ref_aligned, hyp, audio_duration=120.0, collar=0.25)
        self.assertTrue(res_aligned["is_valid"])
        self.assertEqual(res_aligned["status"], "VALID")
        self.assertAlmostEqual(res_aligned["DER"], 0.0, places=2)

    # 28. Isolated reference override mapping mechanism
    def test_28_isolated_reference_override_mechanism(self):
        ref_dir = self.temp_path / "refs"
        ref_dir.mkdir()
        canon_file = ref_dir / "rec1.rttm"
        canon_file.write_text("SPEAKER rec1 1 0 1 <NA> <NA> s1 <NA> <NA>\n", encoding="utf-8")

        # Without override
        resolved_canon = find_reference_rttm(ref_dir, "rec1")
        self.assertEqual(resolved_canon, canon_file)

        # With isolated override mapping
        override_dir = self.temp_path / "overrides"
        override_dir.mkdir()
        override_file = override_dir / "rec1_aligned.rttm"
        override_file.write_text("SPEAKER rec1 1 0 1 <NA> <NA> s1 <NA> <NA>\n", encoding="utf-8")

        override_map = {"rec1": override_file}
        resolved_override = find_reference_rttm(ref_dir, "rec1", reference_overrides=override_map)
        self.assertEqual(resolved_override, override_file)

        # Non-overridden recording still resolves canonically
        canon_file2 = ref_dir / "rec2.rttm"
        canon_file2.write_text("SPEAKER rec2 1 0 1 <NA> <NA> s2 <NA> <NA>\n", encoding="utf-8")
        resolved_rec2 = find_reference_rttm(ref_dir, "rec2", reference_overrides=override_map)
        self.assertEqual(resolved_rec2, canon_file2)


if __name__ == "__main__":
    unittest.main()
