
import csv
import os

# Folder containing this Python script and the TSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "LDC2026S08.tsv")
OUTPUT_FILE = os.path.join(BASE_DIR, "reference.rttm")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: TSV file not found:\n{INPUT_FILE}")
        return

    count = 0
    skipped = 0

    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as tsv, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as rttm:

        reader = csv.DictReader(tsv, delimiter="\t")

        required = {"Audio", "Beg", "End", "Speaker"}

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            print("ERROR: Required columns not found.")
            print("Columns detected:", reader.fieldnames)
            return

        for line_no, row in enumerate(reader, start=2):
            try:
                audio = row["Audio"].strip()
                start = float(row["Beg"])
                end = float(row["End"])
                speaker = row["Speaker"].strip()

                if not audio or not speaker or start < 0 or end <= start:
                    raise ValueError("Invalid audio, speaker, or timestamps")

                # Remove audio file extension for RTTM recording ID
                recording_id = os.path.splitext(
                    os.path.basename(audio)
                )[0]

                duration = end - start

                # RTTM format:
                # SPEAKER file channel start duration
                # <NA> <NA> speaker_id <NA> <NA>
                rttm.write(
                    f"SPEAKER {recording_id} 1 "
                    f"{start:.3f} {duration:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>\n"
                )

                count += 1

            except (ValueError, TypeError, AttributeError, KeyError):
                print(f"Skipping invalid row: {line_no}")
                skipped += 1

    print("\nConversion completed!")
    print(f"Annotations converted: {count}")
    print(f"Rows skipped: {skipped}")
    print(f"Output saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
