"""
HTML/text parsing and alignment: separate prepared remarks from Q&A, speaker diarisation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from bs4 import BeautifulSoup


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_html_transcript(html: str) -> list[dict[str, Any]]:
    """
    Parse transcript HTML into speaker turns.
    Returns list of dicts: speaker, text, section (prepared_remarks | qa), turn_index.
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    return parse_text_transcript(text)


def parse_text_transcript(text: str) -> list[dict[str, Any]]:
    """Parse plain text transcript into speaker turns using common patterns."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    turns = []
    current_speaker = "Unknown"
    current_section = "prepared_remarks"
    buffer: list[str] = []
    turn_index = 0

    # Patterns for speaker lines (e.g. "John Smith, CEO:", "Operator:", "Q&A Session")
    speaker_pattern = re.compile(
        r"^([A-Za-z\s\.\-]+(?:CEO|CFO|COO|CIO|President|Operator|Analyst|Question|Answer|Q&A))[\s]*[:\.]?\s*",
        re.IGNORECASE,
    )
    qa_markers = ("q&a", "question and answer", "questions and answers", "operator")

    for line in lines:
        if not line:
            continue
        line_lower = line.lower()
        if any(m in line_lower for m in qa_markers) and len(line) < 80:
            if buffer:
                turns.append({
                    "speaker": current_speaker,
                    "text": " ".join(buffer),
                    "section": current_section,
                    "turn_index": turn_index,
                })
                turn_index += 1
                buffer = []
            current_section = "qa"
            continue
        match = speaker_pattern.match(line)
        if match and len(line) < 120:
            if buffer:
                turns.append({
                    "speaker": current_speaker,
                    "text": " ".join(buffer),
                    "section": current_section,
                    "turn_index": turn_index,
                })
                turn_index += 1
                buffer = []
            current_speaker = match.group(1).strip()
            rest = line[match.end():].strip()
            if rest:
                buffer.append(rest)
            continue
        buffer.append(line)

    if buffer:
        turns.append({
            "speaker": current_speaker,
            "text": " ".join(buffer),
            "section": current_section,
            "turn_index": turn_index,
        })
    return turns


def classify_speaker_role(speaker: str) -> str:
    """Map speaker label to CEO, CFO, Analyst, Operator, Other."""
    s = speaker.upper()
    if "CEO" in s or "CHIEF EXECUTIVE" in s:
        return "CEO"
    if "CFO" in s or "CHIEF FINANCIAL" in s:
        return "CFO"
    if "ANALYST" in s or "QUESTION" in s:
        return "Analyst"
    if "OPERATOR" in s:
        return "Operator"
    return "Other"


def process_file(input_path: Path, output_dir: Path) -> pd.DataFrame | None:
    """Parse one transcript file and save parsed turns; return DataFrame of turns."""
    content = input_path.read_text(encoding="utf-8", errors="replace")
    if "<?xml" in content[:200] or "<html" in content[:200].lower():
        turns = parse_html_transcript(content)
    else:
        turns = parse_text_transcript(content)
    if not turns:
        return None
    for t in turns:
        t["role"] = classify_speaker_role(t["speaker"])
    df = pd.DataFrame(turns)
    df["source_file"] = input_path.name
    out_path = output_dir / f"{input_path.stem}_parsed.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    raw_dir = Path(paths.get("raw_transcripts", "data/raw/transcripts"))
    out_dir = Path(paths.get("processed_transcripts", "data/processed/transcripts_parsed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not raw_dir.exists():
        print(f"Raw transcript dir not found: {raw_dir}. Run edgar_scraper first.")
        return
    count = 0
    for f in raw_dir.glob("*"):
        if f.name.startswith("_") or f.suffix in (".csv", ".parquet"):
            continue
        try:
            process_file(f, out_dir)
            count += 1
        except Exception as e:
            print(f"Skip {f.name}: {e}")
    print(f"Parsed {count} transcripts to {out_dir}")


if __name__ == "__main__":
    main()
