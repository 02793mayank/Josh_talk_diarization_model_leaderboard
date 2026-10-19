"""
Diagnostic script for investigating the call_input02 reference alignment issue.
Compares original reference, shifted reference (-400.0s), and all 4 model hypotheses.
Runs an offset sweep across models to understand temporal alignment.
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf

# Setup paths
root = Path(__file__).resolve().parent.parent.parent
scoring_dir = root / "Scoring"
if str(scoring_dir) not in sys.path:
    sys.path.insert(0, str(scoring_dir))

from scoring import read_rttm, evaluate_pair
from pyannote.core import Annotation, Segment, Timeline

ref_orig_path = root / "Audio and annotations" / "call_inputs_rttm" / "call_input02.rttm"
ref_shift_path = root / "Scoring" / "diagnostics" / "call_input02_offset_test.rttm"
audio_path = root / "Audio and annotations" / "call_inputs_audio" / "call_input02.flac"

duration = float(sf.info(str(audio_path)).duration)
uem = Timeline([Segment(0.0, duration)])

models = {
    "Model 1 (pyannote)": root / "Model1(pyannote Community-1)" / "hypothesis_rttm" / "call_input02_Model01.rttm",
    "Model 2 (Sortformer)": root / "Model2(NVIDIA NeMo Sortformer)" / "hypothesis_rttm" / "call_input02_Model02.rttm",
    "Model 3 (Clustering)": root / "Model3(NVIDIA NeMo Clustering Diarizer)" / "hypothesis_rttm" / "call_input02_Model03.rttm",
    "Model 4 (Streaming)": root / "Model4(NVIDIA NeMo Streaming Sortformer)" / "hypothesis_rttm" / "call_input02_Model04.rttm",
}

ref_orig = read_rttm(ref_orig_path)
ref_shift = read_rttm(ref_shift_path)

print(f"Audio File: {audio_path.name}")
print(f"Audio Duration: {duration:.2f}s")
print(f"Original Reference File: {ref_orig_path.name}")
print(f"Original Reference Turns: {len(ref_orig)}")
orig_segments = list(ref_orig.itersegments())
print(f"Original Reference Bounds: [{orig_segments[0].start:.3f}s, {orig_segments[-1].end:.3f}s]")

print("\n" + "=" * 80)
print("1. EVALUATION AGAINST ORIGINAL UNALTERED REFERENCE (Scoring Window: [0, 120s])")
print("=" * 80)
for mname, mpath in models.items():
    if mpath.is_file():
        hyp = read_rttm(mpath)
        res = evaluate_pair(ref_orig, hyp, audio_duration=duration, collar=0.25, skip_overlap=False)
        print(f"{mname:<25} | Status: {res['status']:<25} | RefSpeechInWindow: {res['reference_speaker_seconds']:.2f}s | Error: {res['error_details']}")

print("\n" + "=" * 80)
print("2. EVALUATION AGAINST SHIFTED REFERENCE (-400.0s Shift)")
print("=" * 80)
shifted_segments = list(ref_shift.itersegments())
print(f"Shifted Reference Bounds: [{shifted_segments[0].start:.3f}s, {shifted_segments[-1].end:.3f}s]")
print(f"{'Model':<25} | {'DER':<8} | {'JER':<8} | {'Miss %':<8} | {'FA %':<8} | {'Conf %':<8} | {'Ref Speech':<10}")
print("-" * 80)
for mname, mpath in models.items():
    if mpath.is_file():
        hyp = read_rttm(mpath)
        res = evaluate_pair(ref_shift, hyp, audio_duration=duration, collar=0.25, skip_overlap=False)
        if res["is_valid"]:
            print(f"{mname:<25} | {res['DER']:>6.2f}% | {res['JER']:>6.2f}% | {res['missed_speech_percent']:>6.2f}% | {res['false_alarm_percent']:>6.2f}% | {res['speaker_confusion_percent']:>6.2f}% | {res['reference_speaker_seconds']:>8.2f}s")

print("\n" + "=" * 80)
print("3. OFFSET SWEEP ANALYSIS ACROSS ALL 4 MODELS (Offsets: 395.0s to 405.0s)")
print("=" * 80)

def evaluate_offset(offset_sec: float, hyp_ann: Annotation) -> dict:
    shifted_ann = Annotation(uri="call_input02")
    for segment, track, speaker in ref_orig.itertracks(yield_label=True):
        new_start = segment.start - offset_sec
        new_end = segment.end - offset_sec
        if new_end > 0 and new_start < duration:
            shifted_ann[Segment(max(0.0, new_start), min(duration, new_end))] = speaker
    return evaluate_pair(shifted_ann, hyp_ann, audio_duration=duration, collar=0.25, skip_overlap=False)

offsets = np.arange(398.0, 402.1, 0.2)
print(f"{'Offset':<8} | {'Model 1 DER':<12} | {'Model 2 DER':<12} | {'Model 3 DER':<12} | {'Model 4 DER':<12}")
print("-" * 65)
for off in offsets:
    row = [f"{off:6.2f}s"]
    for mname, mpath in models.items():
        hyp = read_rttm(mpath)
        r = evaluate_offset(off, hyp)
        row.append(f"{r['DER']:>10.2f}%" if r["is_valid"] else "       N/A")
    print(" | ".join(row))

# Best offset per model
print("\n" + "-" * 65)
print("OPTIMAL OFFSETS (MINIMUM DER) BY MODEL:")
for mname, mpath in models.items():
    hyp = read_rttm(mpath)
    best_off = None
    best_der = 999.0
    for off in np.arange(395.0, 405.1, 0.1):
        r = evaluate_offset(off, hyp)
        if r["is_valid"] and r["DER"] < best_der:
            best_der = r["DER"]
            best_off = off
    print(f"{mname:<25} -> Optimal Offset: {best_off:6.2f}s (Min DER: {best_der:6.2f}%)")
