
from pathlib import Path
import subprocess
import re
import csv
import sys

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "reference"
HYPOTHESIS_DIR = BASE_DIR / "hypothese"
DSCORE_SCRIPT = BASE_DIR / "score.py"

MODEL_NUMBERS = [1, 2]

RECORDINGS = [
    "call_input01",
    "liveTranscription_input01",
    "meeting_input01",
]


def score_pair(reference_file, hypothesis_file):
    """Run the existing dscore CLI and extract DER and JER."""

    command = [
        sys.executable,
        str(DSCORE_SCRIPT),
        "-r", str(reference_file),
        "-s", str(hypothesis_file),
    ]

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )

    output = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        raise RuntimeError(output.strip())

    # Find DER/JER values in dscore's output.
    # If the output format differs, print it to help adjust parsing.
    der = extract_metric(output, "DER")
    jer = extract_metric(output, "JER")

    return der, jer


def extract_metric(output, metric_name):
    """
    Extract DER or JER from dscore output.
    Supports labeled metrics and the OVERALL summary table.
    """
    patterns = [
        rf"(?im)^\s*{metric_name}\s*[:=]\s*([0-9]*\.?[0-9]+)",
        rf"(?im)^\s*{metric_name}\s+([0-9]*\.?[0-9]+)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))

    # Parse the dscore table:
    # *** OVERALL ***   DER   JER   B3-Precision ...
    match = re.search(
        r"(?m)^\s*\*{3}\s*OVERALL\s*\*{3}\s+"
        r"([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)",
        output
    )

    if match:
        if metric_name.upper() == "DER":
            return float(match.group(1))
        if metric_name.upper() == "JER":
            return float(match.group(2))

    raise ValueError(
        f"Could not find {metric_name} in dscore output.\n"
        f"Actual output was:\n{output}"
    )

def main():
    if not DSCORE_SCRIPT.is_file():
        raise FileNotFoundError(f"Could not find {DSCORE_SCRIPT}")

    if not REFERENCE_DIR.is_dir():
        raise FileNotFoundError(f"Could not find {REFERENCE_DIR}")

    if not HYPOTHESIS_DIR.is_dir():
        raise FileNotFoundError(f"Could not find {HYPOTHESIS_DIR}")

    results = []
    errors = []

    for recording in RECORDINGS:
        reference_file = REFERENCE_DIR / f"{recording}.rttm"

        if not reference_file.is_file():
            errors.append(f"Missing reference: {reference_file}")
            continue

        for model_number in MODEL_NUMBERS:
            hypothesis_file = (
                HYPOTHESIS_DIR
                / f"{recording}_model{model_number:02d}.rttm"
            )

            if not hypothesis_file.is_file():
                errors.append(f"Missing hypothesis: {hypothesis_file}")
                continue

            try:
                der, jer = score_pair(reference_file, hypothesis_file)

                results.append({
                    "recording": recording,
                    "model": f"model {model_number}",
                    "DER": der,
                    "JER": jer,
                })

                print(
                    f"{recording} | model {model_number} | "
                    f"DER={der:.4f} | JER={jer:.4f}"
                )

            except Exception as exc:
                errors.append(
                    f"{recording}, model {model_number}: {exc}"
                )

    # Save detailed results.
    csv_path = BASE_DIR / "model_ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["recording", "model", "DER", "JER"],
        )
        writer.writeheader()
        writer.writerows(results)

    # Macro-average across recordings successfully scored for each model.
    print("\n" + "=" * 55)
    print(f"{'Recording':<30} {'DER':>10} {'JER':>10}")
    print("=" * 55)

    for recording in RECORDINGS:
        print(f"\n{recording}")
        for model_number in MODEL_NUMBERS:
            model_name = f"model {model_number}"
            row = next(
                (
                    item for item in results
                    if item["recording"] == recording
                    and item["model"] == model_name
                ),
                None,
            )

            if row:
                print(
                    f"  {model_name:<20} "
                    f"{row['DER']:>10.4f} {row['JER']:>10.4f}"
                )

    print("\nAVERAGE (macro-average across available recordings)")
    for model_number in MODEL_NUMBERS:
        model_name = f"model {model_number}"
        model_rows = [r for r in results if r["model"] == model_name]

        if model_rows:
            avg_der = sum(r["DER"] for r in model_rows) / len(model_rows)
            avg_jer = sum(r["JER"] for r in model_rows) / len(model_rows)
            print(
                f"  {model_name:<20} {avg_der:>10.4f} {avg_jer:>10.4f}"
            )
        else:
            print(f"  {model_name:<20} No results")

    print(f"\nDetailed CSV saved to: {csv_path}")

    if errors:
        print("\nProblems found:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
