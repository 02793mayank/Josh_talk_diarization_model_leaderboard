
from pathlib import Path
import xml.etree.ElementTree as ET

# Project folders
base_dir = Path(__file__).parent
segments_dir = base_dir / "annotations" / "ICSI" / "Segments"
output_file = base_dir / "Bed004_reference.rttm"

meeting_id = "Bed004"
annotations = set()

# Read all Bed004 segment annotation files
for xml_file in sorted(segments_dir.glob("Bed004.*.segs.xml")):
    root = ET.parse(xml_file).getroot()

    for seg in root.iter("segment"):
        start = seg.get("starttime")
        end = seg.get("endtime")
        speaker = seg.get("participant")

        if start is None or end is None or not speaker:
            continue

        start = float(start)
        end = float(end)

        if end <= start:
            continue

        # Remove exact duplicate annotations
        annotations.add((start, end, speaker))

# Write standard RTTM records
with output_file.open("w", encoding="utf-8") as f:
    for start, end, speaker in sorted(annotations):
        duration = end - start
        f.write(
            f"SPEAKER {meeting_id} 1 "
            f"{start:.3f} {duration:.3f} "
            f"<NA> <NA> {speaker} <NA> <NA>\n"
        )

print(f"Created: {output_file}")
print(f"Number of segments: {len(annotations)}")
