
from pathlib import Path


def community1_to_rttm(output, recording_id, output_path):
    """
    Convert pyannote Community-1 output to RTTM format.

    Args:
        output: Output object returned by the diarization pipeline.
        recording_id: Recording ID matching the reference RTTM.
        output_path: Destination path for the RTTM file.
    """

    if hasattr(output, "speaker_diarization"):
        diarization = output.speaker_diarization
    else:
        diarization = output

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    if hasattr(diarization, "itertracks"):
        iterator = ((turn, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True))
    else:
        iterator = diarization

    for turn, speaker in iterator:
        start = float(turn.start)
        end = float(turn.end)
        duration = end - start

        if start < 0 or duration <= 0:
            raise ValueError(
                f"Invalid segment: start={start}, end={end}"
            )

        row = (
            f"SPEAKER {recording_id} 1 "
            f"{start:.3f} {duration:.3f} "
            f"<NA> <NA> {speaker} <NA> <NA>"
        )
        rows.append(row)

    if not rows:
        raise ValueError("No speaker turns found in model output.")

    output_path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8"
    )

    print(f"Converted {len(rows)} speaker turns to RTTM.")
    print(f"Saved file: {output_path}")


if __name__ == "__main__":
    print(
        "Converter ready. Import community1_to_rttm() "
        "into your model inference script."
    )
