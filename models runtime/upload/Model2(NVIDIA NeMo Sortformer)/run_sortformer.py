
import argparse
import gc
import tempfile
from pathlib import Path

# Windows fix: prevent WinError 32 on Windows when dataloaders/generators keep open file handles
_orig_tempdir = tempfile.TemporaryDirectory
class SafeTemporaryDirectory(_orig_tempdir):
    def cleanup(self):
        gc.collect()
        try:
            super().cleanup()
        except Exception:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        try:
            super().__exit__(exc_type, exc_val, exc_tb)
        except Exception:
            pass

tempfile.TemporaryDirectory = SafeTemporaryDirectory

import soundfile as sf
from nemo.collections.asr.models import SortformerEncLabelModel
from sortformer_to_rttm import sortformer_to_rttm


MODEL_ID = "nvidia/diar_sortformer_4spk-v1"

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = MODEL_DIR / "hypothesis_rttm"

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".sph"
}


def find_audio_files(input_path):
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {path.suffix}")
        return [path]

    files = sorted(
        file for file in path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not files:
        raise ValueError(f"No audio files found in: {path}")

    # Avoid overwriting files with the same filename stem.
    stems = [file.stem for file in files]
    if len(stems) != len(set(stems)):
        raise ValueError(
            "Duplicate audio filename stems found. "
            "Process these recordings separately."
        )

    return files


def main():
    parser = argparse.ArgumentParser(
        description="Run NeMo Sortformer and generate hypothesis RTTMs."
    )

    parser.add_argument(
        "input",
        help="Path to one audio file or a folder containing audio files.",
    )

    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated RTTM files.",
    )

    args = parser.parse_args()

    audio_files = find_audio_files(args.input)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(audio_files)} audio file(s).")
    print(f"Loading model: {MODEL_ID}")

    model = SortformerEncLabelModel.from_pretrained(
        model_name=MODEL_ID
    )
    model.eval()

    successful = 0
    failed = []

    for index, audio_path in enumerate(audio_files, start=1):
        recording_id = audio_path.stem
        rttm_path = output_dir / f"{recording_id}_Model02.rttm"

        print(f"\n[{index}/{len(audio_files)}] {audio_path.name}")

        if rttm_path.is_file() and rttm_path.stat().st_size > 0:
            print(f"Output already exists and is non-empty: {rttm_path.name} (skipping)")
            successful += 1
            continue

        try:
            # Handle .sph files via temporary wav to avoid missing Windows sph2pipe
            if audio_path.suffix.lower() == ".sph":
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_wav = Path(tmp_dir) / f"{recording_id}.wav"
                    data, sr = sf.read(str(audio_path))
                    sf.write(str(tmp_wav), data, sr)

                    batch_segments = model.diarize(
                        audio=[str(tmp_wav)],
                        batch_size=1,
                    )
            else:
                batch_segments = model.diarize(
                    audio=[str(audio_path)],
                    batch_size=1,
                )

            predicted_segments = batch_segments[0]

            # Automatically convert predictions to RTTM.
            sortformer_to_rttm(
                predicted_segments=predicted_segments,
                recording_id=recording_id,
                output_path=rttm_path,
            )

            successful += 1

        except Exception as error:
            failed.append((audio_path, str(error)))
            print(f"FAILED: {error}")

    print("\n========== SUMMARY ==========")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")
    print(f"Output directory: {output_dir.resolve()}")

    for audio_path, error in failed:
        print(f"\nFailed: {audio_path}\nReason: {error}")


if __name__ == "__main__":
    main()
