import os
import re
from pathlib import Path

# This script and LDC97S42.txt should be in the same folder.
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "LDC97S42.txt"
OUTPUT_FILE = BASE_DIR / "reference_ldc97s42.rttm"

AUDIO_EXTENSIONS = {".wav", ".flac", ".sph", ".mp3", ".m4a", ".ogg"}

# Matches transcript lines such as:
# 7.80 8.77 A: How you doing?
# 55.88 57.41 A1: #and the apples# on the floor
LINE_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s+"
    r"(\d+(?:\.\d+)?)\s+"
    r"([A-Za-z][A-Za-z0-9_-]*):"
)


def find_audio_id():
    """Use the only audio file in this folder if there is exactly one."""
    audio_files = [
        p for p in BASE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if len(audio_files) == 1:
        return audio_files[0].stem
    return None


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Transcript file not found:\n{INPUT_FILE}")
        return

    recording_id = find_audio_id()

    if recording_id is None:
        recording_id = input(
            "Enter the matching audio filename WITHOUT extension "
            "(for example, en_5551): "
        ).strip()

    if not recording_id:
        print("ERROR: Recording ID cannot be empty.")
        return

    converted = 0
    skipped = 0

    with INPUT_FILE.open("r", encoding="utf-8-sig", errors="replace") as source, \
         OUTPUT_FILE.open("w", encoding="utf-8", newline="") as output:

        for line_number, line in enumerate(source, start=1):
            match = LINE_PATTERN.match(line)

            # Ignore blank lines, headings, and other non-annotation lines.
            if not match:
                if line.strip():
                    skipped += 1
                continue

            try:
                start = float(match.group(1))
                end = float(match.group(2))
                speaker = match.group(3)

                if start < 0 or end <= start:
                    raise ValueError("Invalid start/end timestamps")

                duration = end - start

                # RTTM format:
                # SPEAKER <recording-id> <channel> <start> <duration>
                #        <NA> <NA> <speaker-id> <NA> <NA>
                output.write(
                    f"SPEAKER {recording_id} 1 "
                    f"{start:.3f} {duration:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>\n"
                )
                converted += 1

            except ValueError:
                print(f"Skipping invalid timestamp at line {line_number}")
                skipped += 1

    print("\nConversion completed.")
    print(f"Speaker segments converted: {converted}")
    print(f"Other/invalid lines skipped: {skipped}")
    print(f"RTTM saved to: {OUTPUT_FILE}")
    print(f"Recording ID used: {recording_id}")
    print(
        "\nIMPORTANT: Ensure this recording ID matches the corresponding "
        "audio filename (without its extension)."
    )


if __name__ == "__main__":
    main()
