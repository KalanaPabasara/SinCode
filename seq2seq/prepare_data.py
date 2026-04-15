"""
Parse WSD.txt into a CSV training dataset for ByT5 fine-tuning.

Input format (WSD.txt):
    Word: <romanized>, Sinhala Words: ['<s1>', '<s2>', ...]

Output (wsd_pairs.csv):
    romanized,sinhala
    wadi,වෑඩි
    wadi,වාඩි
    ...

One row per (romanized, sinhala) pair. Duplicate sinhala entries per
word are kept since ByT5 learns from all valid transliterations.
"""

import ast
import csv
import re
import sys
from pathlib import Path

WSD_PATH   = Path(r"C:\Y5_Docs\FYP\WSD.txt")
OUT_PATH   = Path(__file__).parent / "wsd_pairs.csv"

LINE_RE = re.compile(r"^Word:\s*(.+?),\s*Sinhala Words:\s*(\[.+\])\s*$")

MIN_ROMAN_LEN = 2   # skip single-char romanized entries
MAX_ROMAN_LEN = 40  # skip obviously malformed long entries


def parse_wsd(wsd_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    skipped = 0

    with wsd_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            m = LINE_RE.match(line)
            if not m:
                skipped += 1
                continue

            roman = m.group(1).strip().lower()
            if not (MIN_ROMAN_LEN <= len(roman) <= MAX_ROMAN_LEN):
                skipped += 1
                continue

            try:
                sinhala_list = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                skipped += 1
                continue

            for sinhala in sinhala_list:
                sinhala = sinhala.strip()
                if sinhala:
                    pairs.append((roman, sinhala))

            if lineno % 100_000 == 0:
                print(f"  processed {lineno:,} lines, {len(pairs):,} pairs so far…")

    print(f"  skipped {skipped:,} malformed lines")
    return pairs


def write_csv(pairs: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["romanized", "sinhala"])
        writer.writerows(pairs)


def main() -> None:
    print(f"Parsing {WSD_PATH} …")
    pairs = parse_wsd(WSD_PATH)
    print(f"\nTotal pairs: {len(pairs):,}")

    print(f"Writing to {OUT_PATH} …")
    write_csv(pairs, OUT_PATH)
    print("Done.")

    # Quick sanity check
    print("\nSample rows:")
    for roman, sinhala in pairs[:5]:
        print(f"  {roman!r:20s} → {sinhala}")


if __name__ == "__main__":
    main()
