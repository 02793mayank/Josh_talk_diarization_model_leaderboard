
from pathlib import Path


def nemo_clustering_to_rttm(
    source_rttm,
    recording_id,
    output_path,
):
    """Validate NeMo's RTTM and save the hypothesis RTTM."""

    source_rttm = Path(source_rttm)
    output_path = Path(output_path)

    if not source_rttm.is_file():
        raise FileNotFoundError(
            f"NeMo output RTTM not found: {source_rttm}"
        )

    rows = []

    for line in source_rttm.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.split()

        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise ValueError(f"Invalid RTTM row: {line}")

        start = float(fields[3])
        duration = float(fields[4])

        if start < 0 or duration <= 0:
            raise ValueError(f"Invalid timing: {line}")

        # Replace the file ID with the requested recording ID.
        fields[1] = recording_id
        rows.append(" ".join(fields))

    if not rows:
        raise ValueError("NeMo produced no speaker turns.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {len(rows)} turns to {output_path}")
