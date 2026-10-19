
import xml.etree.ElementTree as ET
from pathlib import Path

# Folder containing the AMI annotation XML files
segments_dir = Path("segments")
output_file = Path("reference.rttm")
meeting_id = "ES2002a"

speaker_files = sorted(
    segments_dir.glob(f"{meeting_id}.*.segments.xml")
)

if not speaker_files:
    raise FileNotFoundError(
        f"No annotation files found in {segments_dir.resolve()}"
    )

rttm_lines = []

for xml_file in speaker_files:
    # ES2002a.A.segments.xml -> speaker A
    speaker_id = xml_file.name.split(".")[1]

    root = ET.parse(xml_file).getroot()

    for segment in root.iter("segment"):
        start = float(segment.attrib["transcriber_start"])
        end = float(segment.attrib["transcriber_end"])
        duration = end - start

        if duration <= 0:
            continue

        rttm_lines.append(
            f"SPEAKER {meeting_id} 1 "
            f"{start:.3f} {duration:.3f} "
            f"<NA> <NA> {speaker_id} <NA> <NA>"
        )

# Sort segments by start time
rttm_lines.sort(key=lambda line: float(line.split()[3]))

output_file.write_text(
    "\n".join(rttm_lines) + "\n",
    encoding="utf-8"
)

print("RTTM file created:", output_file.resolve())
print("Annotation files:", len(speaker_files))
print("Total segments:", len(rttm_lines))
print("Speakers:", ", ".join(
    path.name.split(".")[1] for path in speaker_files
))
