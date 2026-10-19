# Optional Speaker Diarization Benchmark

This folder contains the complete artifacts, metrics, and reproducible scoring scripts for the optional speaker diarization benchmark across **Models 01, 02, and 04** on three representative audio recordings.

---

## 1. Directory Structure

```
Optional Task - Diarization Scoring/experiments/results/optional_diarization_benchmark/
├── README.md                          # Reproduction and interpretation guide (this document)
├── diarization_evaluation_report.md   # Comprehensive benchmark narrative and component analysis
├── diarization_scores.csv             # Official tabular metric sheet (9 recording-system pairs)
├── evaluation_manifest.json           # File mapping of audio, reference, and hypothesis RTTMs
├── evaluation_results.json            # Machine-readable evaluation results and component metadata
└── run_evaluation.py                  # Standalone reproducible scoring script (pyannote.metrics)
```

---

## 2. Evaluation Scope & Setup

- **Canonical Workspace:** `Optional Task - Diarization Scoring\`
- **Scoring Engine:** `pyannote.metrics 4.2.0` (`pyannote.core 6.1.0`)
- **Python Environment:** `Optional Task - Diarization Scoring\.venv` (Python 3.14.7)
- **Primary Metric Configuration:**
  - `collar = 0.0s` (strict, no tolerance collar around boundaries)
  - `skip_overlap = False` (overlapping speech regions fully scored)
  - Optimal Hungarian bipartite matching for speaker labels

### Evaluated Systems
1. **Model 01 — pyannote Community-1:** `pyannote/speaker-diarization-community-1` (pyannote.audio 4.0.7)
2. **Model 02 — NVIDIA NeMo Sortformer (Offline):** `nvidia/diar_sortformer_4spk-v1` (NeMo 3.0.0)
3. **Model 04 — NVIDIA NeMo Streaming Sortformer:** `nvidia/diar_streaming_sortformer_4spk-v2.1` (NeMo 3.0.0)

### Selected Audio Inputs
1. `liveTranscription_input02.wav` (139.91s, VoxConverse presentation, 1 reference speaker)
2. `call_input01.sph` (120.00s, CALLHOME LDC97S42 8 kHz telephony, 3 reference speakers)
3. `call_input02.flac` (120.00s, CALLHOME en_5551 8 kHz telephony excerpt) — **EXCLUDED from official aggregates** due to an unaligned +400s reference timestamp offset.

---

## 3. How to Reproduce Scoring

To rerun the evaluation without invoking model inference:

```powershell
# From the project root:
cd "C:\Users\mrdag\OneDrive\Desktop\Task1 Josh talks"

# Execute using the canonical virtual environment:
& "Optional Task - Diarization Scoring\.venv\Scripts\python.exe" "Optional Task - Diarization Scoring\experiments\results\optional_diarization_benchmark\run_evaluation.py"
```

The script will:
1. Load and validate all reference RTTMs and pre-generated hypothesis RTTMs.
2. Evaluate all 9 pairs under `collar=0.0s`, `skip_overlap=False`.
3. Exclude `call_input02` from official ranking due to the +400s unaligned ground truth.
4. Output the updated `diarization_scores.csv` and `evaluation_results.json` directly into this directory.

---

## 4. Benchmark Summary Results

### Valid-Set Ranking (Average across `liveTranscription_input02` and `call_input01`)

| Rank | System | Macro-Avg DER | Macro-Avg JER | Micro-Avg DER |
|:---:|---|:---:|:---:|:---:|
| **1** | **Model 02 (NeMo Sortformer Offline)** | **39.14%** | **32.60%** | **32.55%** |
| **2** | **Model 04 (NeMo Streaming Sortformer)** | **40.45%** | **27.39%** | **33.68%** |
| **3** | **Model 01 (pyannote Community-1)** | **45.72%** | **40.90%** | **38.15%** |

### Per-Recording Breakdown

| Recording ID | Source | Model 01 DER (JER) | Model 02 DER (JER) | Model 04 DER (JER) |
|---|---|:---:|:---:|:---:|
| `liveTranscription_input02` | VoxConverse (Clean) | 4.99% (4.81%) | **3.67%** (3.57%) | 4.03% (3.90%) |
| `call_input01` | CALLHOME (Telephony) | 86.45% (76.99%) | **74.61%** (61.63%) | 76.86% (50.89%) |
| `call_input02` | CALLHOME (Unaligned) | *INVALID_REF* | *INVALID_REF* | *INVALID_REF* |

---

## 5. Interpreting Results

- **Broadcast / Clean Audio (`liveTranscription_input02`):** All three models deliver production-grade diarization ($3.67\%$ to $4.99\%$ DER), with NeMo Sortformer achieving zero speaker confusion.
- **Narrowband Telephony (`call_input01`):** In 8 kHz conversational audio with multiple overlapping speakers, pyannote has lower missed speech ($9.07\%$) but higher false alarms ($51.99\%$). Sortformer models demonstrate lower false alarm and lower confusion.
- **Sample Size Limitation:** A benchmark of 2 valid recordings provides insight into domain adaptability, but does not represent generalized performance across large-scale speech corpuses.
