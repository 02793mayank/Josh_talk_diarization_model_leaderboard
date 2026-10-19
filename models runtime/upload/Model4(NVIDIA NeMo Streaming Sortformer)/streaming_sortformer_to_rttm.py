from pathlib import Path


def streaming_sortformer_to_rttm(
    predicted_segments,
    recording_id,
    output_path,
):
    """
    Convert NeMo Streaming Sortformer segments to RTTM.

    Accepts segments in string format: "start end speaker"
    or list/tuple format: [start, end, speaker]
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for segment in predicted_segments:
        if isinstance(segment, str):
            parts = segment.strip().split()
            if len(parts) != 3:
                raise ValueError(f"Unexpected segment format: {segment}")
            start_val, end_val, spk_val = parts
        elif hasattr(segment, '__len__') and len(segment) == 3:
            start_val, end_val, spk_val = segment
        else:
            raise ValueError(f"Unexpected segment format: {segment}")

        start = float(start_val)
        end = float(end_val)
        duration = end - start

        if start < 0 or duration <= 0:
            raise ValueError(
                f"Invalid segment: start={start}, end={end}"
            )

        if str(spk_val).startswith("speaker_"):
            speaker_id = str(spk_val)
        else:
            try:
                speaker_id = f"speaker_{int(spk_val)}"
            except ValueError:
                speaker_id = f"speaker_{spk_val}"

        rows.append(
            f"SPEAKER {recording_id} 1 "
            f"{start:.3f} {duration:.3f} "
            f"<NA> <NA> {speaker_id} <NA> <NA>"
        )

    if not rows:
        raise ValueError("No speaker segments found.")

    output_path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    print(f"Converted {len(rows)} segments.")
    print(f"RTTM saved: {output_path}")
