# Model 3 Setup Report: NVIDIA NeMo Clustering Diarizer

## 1. Existing Files Inspected
The initial project repository contained the following files for Model 3:
- `config/diar_infer.yaml`: NeMo offline speaker diarization configuration file specifying VAD (`vad_multilingual_marblenet`), speaker embedding extractor (`titanet_large`), clustering parameters, and ASR decoder options.
- `run_nemo_clustering.py`: Runner script implementing input discovery, temporary manifest generation, configuration parameter overrides, ClusteringDiarizer invocation, and automatic hypothesis RTTM conversion.
- `nemo_clustering_to_rttm.py`: Conversion and validation function reading NeMo's raw RTTM outputs, asserting 10-column RTTM schema, checking start/duration bounds, updating file/recording ID, and outputting to hypothesis directory.
- `manifests/`: Directory for input manifests.
- `hypothesis_rttm/`: Destination directory for generated hypothesis RTTMs.

---

## 2. Environment Path and Python Version
- **Virtual Environment Path**: `Model3(NVIDIA NeMo Clustering Diarizer)/.venv`
- **Python Version**: Python 3.11.9 (`[MSC v.1938 64 bit (AMD64)]`)
- **PyTorch Device**: CPU (`torch.cuda.is_available() == False`)

---

## 3. Dependency Versions
The isolated virtual environment was configured and verified clean with `pip check` (0 broken requirements). Key package versions include:
- `python`: 3.11.9
- `nemo-toolkit`: 3.0.0
- `torch`: 2.14.1+cpu
- `torchaudio`: 2.11.0+cpu
- `omegaconf`: 2.3.0
- `hydra-core`: 1.3.2
- `lightning`: 2.4.0
- `pytorch-lightning`: 2.6.6
- `transformers`: 5.20.1
- `soundfile`: 0.14.0
- `librosa`: 0.11.0
- `lhotse`: 1.33.0
- `einops`: 0.8.2
- `kaldialign`: 0.12.0
- `pandas`: 3.0.6
- `pyarrow`: 26.0.0
- `datasets`: 5.1.0
- `webdataset`: 1.0.2
- `nv-one-logger-core`: 2.3.1

---

## 4. Changes Made and Rationale
1. **Virtual Environment Creation**:
   - Initialized dedicated `.venv` using Python 3.11 (`py -3.11 -m venv .venv`).
   - Installed CPU-compatible PyTorch/TorchAudio from official PyTorch CPU wheel index (`--extra-index-url https://download.pytorch.org/whl/cpu`).
   - Installed NeMo 3.0.0 and its core execution dependencies (`lightning`, `lhotse`, `kaldialign`, `transformers`, `einops`, `pandas`, `nv-one-logger`, `webdataset`).
2. **Device Fallback in `run_nemo_clustering.py`**:
   - Added explicit device resolution check before initializing `ClusteringDiarizer`:
     ```python
     import torch
     if config.device is None:
         config.device = "cuda" if torch.cuda.is_available() else "cpu"
     ```
   - **Rationale**: In `diar_infer.yaml`, `device` is set to `null`. Without explicit CPU fallback on non-CUDA systems, PyTorch `map_location` can default to CUDA if pretrained weights contain CUDA metadata, causing a runtime crash on CPU machines.

---

## 5. Commands and Tests Executed

| Command / Test | Purpose | Result |
| :--- | :--- | :--- |
| `py -3.11 -m venv .venv` | Create isolated Model 3 venv | Passed |
| `pip install torch==2.14.1+cpu torchaudio==2.11.0+cpu ...` | Install CPU PyTorch and dependencies | Passed |
| `pip check` | Verify dependency tree integrity | Passed (No broken requirements found) |
| `python -m py_compile run_nemo_clustering.py nemo_clustering_to_rttm.py` | Verify syntax compilation | Passed |
| `from nemo.collections.asr.models import ClusteringDiarizer` | Verify NeMo import and module integrity | Passed |
| Synthetic RTTM Conversion Suite | Test `nemo_clustering_to_rttm` against valid format, negative start, 0 duration, empty output, bad column count | Passed (All 5 edge cases handled as expected) |

---

## 6. Verification Status

- **Imports**: Verified. `ClusteringDiarizer`, `OmegaConf`, `torch`, `soundfile`, and `nemo_clustering_to_rttm` import without errors.
- **Converter**: Verified. Synthetic 10-column RTTM conversion confirmed valid start/duration calculation, speaker preservation, and strict exception raising on malformed/empty records.
- **Pretrained Checkpoint IDs**: Verified.
  - VAD model: `vad_multilingual_marblenet` is validated against NeMo's catalog (`EncDecClassificationModel.list_available_models()`).
  - Speaker Embedding model: `titanet_large` is validated against NeMo's catalog (`EncDecSpeakerLabelModel.list_available_models()`).
  - Ungated NGC models: Both models are hosted on NVIDIA NGC and do not require `HF_TOKEN`.
- **Checkpoint Access & Loading**: Unverified (Checkpoint files not downloaded during audit, as downloading heavy models was strictly prohibited).
- **CPU Inference Support**: Partially verified. Code paths and libraries run on CPU without CUDA bindings errors; full inference latency and memory requirements on CPU remain unverified until real inference.
- **Audio Inference**: Not verified (Intentionally out of scope; no real audio scanned or processed).

---

## 7. Known Windows Compatibility Considerations
- NeMo's `ClusteringDiarizer` utilizes Python-level orchestration with PyTorch, Lhotse, and Kaldi-align. All compiled extensions (`kaldialign`, `soundfile`, `numba`) are successfully installed as native Windows amd64 wheels.
- Windows path length limitations: NeMo creates intermediate folders under the temporary directory (`pred_rttms`, `speaker_outputs`). `run_nemo_clustering.py` uses `tempfile.TemporaryDirectory(prefix="nemo_diarization_")`, which is safe on Windows.
- On Windows CPU, TitaNet feature extraction and MarbleNet VAD execution can be slow compared to Linux/CUDA environments.

---

## 8. Exact Remaining Blockers Before Inference
1. **Pretrained Checkpoint Download**: At initial inference execution, `ClusteringDiarizer` will attempt to fetch `vad_multilingual_marblenet` (~10MB) and `titanet_large` (~80MB) from NVIDIA NGC. An active internet connection (or pre-downloaded local `.nemo` checkpoint paths specified in `config/diar_infer.yaml`) is required.
2. **Audio Input Provision**: Valid input audio paths must be supplied.

---

## 9. Eventual Inference Command

> [!NOTE]
> **STATUS: NOT EXECUTED** (Preparation and validation only; inference not run)

To run inference on an audio file or directory once checkpoints are available:

```powershell
& ".\Model3(NVIDIA NeMo Clustering Diarizer)\.venv\Scripts\python.exe" ".\Model3(NVIDIA NeMo Clustering Diarizer)\run_nemo_clustering.py" "<PATH_TO_AUDIO_FILE_OR_FOLDER>"
```
