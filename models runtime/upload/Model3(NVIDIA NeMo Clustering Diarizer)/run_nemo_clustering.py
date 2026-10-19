
import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer

from nemo_clustering_to_rttm import nemo_clustering_to_rttm


MODEL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODEL_DIR / "config" / "diar_infer.yaml"
OUTPUT_DIR = MODEL_DIR / "hypothesis_rttm"

AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".sph"
}


def find_audio_files(input_path):
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {path.suffix}")
        return [path.resolve()]

    files = sorted(
        file.resolve()
        for file in path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not files:
        raise ValueError(f"No audio files found in {path}")

    stems = [file.stem for file in files]
    if len(stems) != len(set(stems)):
        raise ValueError(
            "Duplicate audio filename stems found. "
            "Process these files separately to avoid overwriting RTTMs."
        )

    return files


def preprocess_audio_for_nemo(audio_path: Path, temp_dir: Path, target_sr: int = 16000) -> Path:
    """
    Ensure audio is mono and target_sr (16 kHz) PCM WAV for NeMo VAD & clustering.
    Averages stereo/multichannel audio to mono.
    Resamples using scipy.signal.resample_poly.
    Preserves original source audio file completely untouched.
    """
    recording_id = audio_path.stem
    temp_wav = temp_dir / f"{recording_id}.wav"

    data, sr = sf.read(str(audio_path), dtype="float32")

    # Downmix to mono if multi-channel (e.g. 2-D stereo array)
    if data.ndim > 1:
        data = np.mean(data, axis=-1)

    # Resample to target_sr (16000 Hz) if needed
    if sr != target_sr:
        gcd = math.gcd(target_sr, sr)
        up = target_sr // gcd
        down = sr // gcd
        data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
        sr = target_sr

    # Write as standard 16-bit PCM WAV
    sf.write(str(temp_wav), data, sr, subtype="PCM_16")
    return temp_wav


def create_manifest(audio_files, manifest_path, temp_dir):
    """Create a manifest without reference labels or oracle speaker counts.
    Converts audio to temporary mono, 16 kHz WAV files to ensure NeMo VAD compatibility."""

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for audio_path in audio_files:
            temp_wav = preprocess_audio_for_nemo(audio_path, Path(temp_dir))

            entry = {
                "audio_filepath": str(temp_wav),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": None,
                "uem_filepath": None,
            }
            manifest.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run NeMo clustering diarization and export RTTMs."
    )
    parser.add_argument(
        "input",
        help="Path to one audio file or a folder of audio files.",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        default=str(OUTPUT_DIR),
        help="Directory for generated RTTM files.",
    )
    args = parser.parse_args()

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    audio_files = find_audio_files(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out files that already have valid hypothesis RTTM
    pending_files = []
    already_done = 0
    for audio_path in audio_files:
        recording_id = audio_path.stem
        target_rttm = output_dir / f"{recording_id}_Model03.rttm"
        if target_rttm.is_file() and target_rttm.stat().st_size > 0:
            print(f"Output already exists and is non-empty: {target_rttm.name} (skipping)")
            already_done += 1
        else:
            pending_files.append(audio_path)

    if not pending_files:
        print(f"\nAll {len(audio_files)} file(s) already processed for Model 3.")
        return

    print(f"\nProcessing {len(pending_files)} audio file(s) ({already_done} already completed).")

    # Create a temporary manifest; reference annotations are not used.
    with tempfile.TemporaryDirectory(prefix="nemo_diarization_") as temp_dir:
        temp_dir = Path(temp_dir)
        manifest_path = temp_dir / "manifest.json"
        nemo_output_dir = temp_dir / "nemo_output"

        create_manifest(pending_files, manifest_path, temp_dir)

        # Load the downloaded YAML and replace its placeholders.
        config = OmegaConf.load(CONFIG_PATH)
        config.diarizer.manifest_filepath = str(manifest_path)
        config.diarizer.out_dir = str(nemo_output_dir)

        # Do not use reference speech regions or speaker counts.
        config.diarizer.oracle_vad = False
        config.diarizer.clustering.parameters.oracle_num_speakers = False

        import torch
        if config.device is None:
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        config.num_workers = 0

        print("Initializing NeMo ClusteringDiarizer...")

        diarizer = ClusteringDiarizer(cfg=config)
        diarizer.diarize()

        # NeMo normally writes predicted RTTMs under pred_rttms.
        predicted_dir = nemo_output_dir / "pred_rttms"

        if not predicted_dir.is_dir():
            raise FileNotFoundError(
                "NeMo did not create the expected pred_rttms directory. "
                f"Inspect its output directory: {nemo_output_dir}"
            )

        successful = already_done
        failed = []

        for audio_path in pending_files:
            recording_id = audio_path.stem
            source_rttm = predicted_dir / f"{recording_id}.rttm"
            destination = output_dir / f"{recording_id}_Model03.rttm"

            try:
                nemo_clustering_to_rttm(
                    source_rttm=source_rttm,
                    recording_id=recording_id,
                    output_path=destination,
                )
                successful += 1

            except Exception as error:
                failed.append((audio_path, str(error)))
                print(f"FAILED: {audio_path.name}: {error}")

        print("\n========== SUMMARY ==========")
        print(f"Successful: {successful}")
        print(f"Failed: {len(failed)}")
        print(f"Hypothesis RTTMs: {output_dir.resolve()}")

        for audio_path, error in failed:
            print(f"{audio_path.name}: {error}")


if __name__ == "__main__":
    main()
