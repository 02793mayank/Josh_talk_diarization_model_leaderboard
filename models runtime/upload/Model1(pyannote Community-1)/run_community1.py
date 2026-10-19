
import os
import argparse
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from pyannote.audio import Pipeline
from community1_to_rttm import community1_to_rttm


def load_audio(audio_path, target_sr=16000):
    """
    Load audio using soundfile, downmix multichannel to mono,
    resample to target_sr if needed, and return pyannote waveform dict.
    """
    data, sr = sf.read(str(audio_path))
    waveform = torch.from_numpy(data).float()

    if waveform.ndim == 2:
        # soundfile yields (time, channels) -> transpose to (channels, time)
        waveform = waveform.t()
        # Downmix to mono: average across channels -> (1, time)
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected audio shape: {waveform.shape}")

    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=target_sr
        )
        sr = target_sr

    return {"waveform": waveform, "sample_rate": sr}


MODEL_ID = "pyannote/speaker-diarization-community-1"

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = MODEL_DIR / "hypothesis_rttm"

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".sph"
}


def find_audio_files(input_path):
    """Accept one audio file or recursively find audio files in a folder."""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {input_path.suffix}")
        return [input_path]

    files = sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not files:
        raise ValueError(f"No supported audio files found in: {input_path}")

    # Prevent two files with the same stem from overwriting the same RTTM.
    stems = [p.stem for p in files]
    if len(stems) != len(set(stems)):
        raise ValueError(
            "Duplicate audio filenames (same name without extension) found. "
            "Process those files separately or assign unique recording IDs."
        )

    return files


def main():
    parser = argparse.ArgumentParser(
        description="Run Community-1 and automatically save hypothesis RTTMs."
    )
    parser.add_argument(
        "input",
        help="Path to one audio file or a folder containing audio files."
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder for hypothesis RTTMs."
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if token:
        print("Using HF_TOKEN from environment.")
    else:
        print("HF_TOKEN not set in process; attempting to load from local Hugging Face cache...")

    audio_files = find_audio_files(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Audio files found: {len(audio_files)}")
    print("Loading Community-1 model...")

    if token:
        pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    else:
        pipeline = Pipeline.from_pretrained(MODEL_ID)

    successful = 0
    failed = []

    for index, audio_path in enumerate(audio_files, start=1):
        recording_id = audio_path.stem
        rttm_path = output_dir / f"{recording_id}_Model01.rttm"

        print(f"\n[{index}/{len(audio_files)}] {audio_path.name}")

        if rttm_path.is_file() and rttm_path.stat().st_size > 0:
            print(f"Output already exists and is non-empty: {rttm_path.name} (skipping)")
            successful += 1
            continue

        try:
            # 1. Load audio with soundfile, convert to mono, resample to 16 kHz
            audio_input = load_audio(audio_path, target_sr=16000)

            # 2. Run the model on the waveform dict.
            model_output = pipeline(audio_input)

            # 2. Automatically convert its output to RTTM.
            community1_to_rttm(
                output=model_output,
                recording_id=recording_id,
                output_path=rttm_path,
            )

            successful += 1

        except Exception as error:
            failed.append((audio_path, str(error)))
            print(f"FAILED: {error}")

    print("\n========== RUN SUMMARY ==========")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")
    print(f"Hypothesis RTTMs: {output_dir.resolve()}")

    if failed:
        print("\nFailed files:")
        for path, error in failed:
            print(f"- {path}: {error}")


if __name__ == "__main__":
    main()
