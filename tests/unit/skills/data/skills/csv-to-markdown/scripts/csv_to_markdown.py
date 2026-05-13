#!/usr/bin/env python3
"""
Convert a CSV file to a Markdown table.

Usage:
    python csv_to_markdown.py data.csv
    python csv_to_markdown.py data.csv --max-rows 20
    python csv_to_markdown.py data.csv --columns name,age,email
"""

import argparse
import csv
import sys
from pathlib import Path


def _cell(value: str, width: int) -> str:
    return value.ljust(width)


def csv_to_markdown(
    rows: list[list[str]],
    columns: list[str] | None = None,
    max_rows: int | None = None,
) -> str:
    if not rows:
        return "(empty table)"

    header = rows[0]

    # Select and reorder columns if requested
    if columns:
        indices = []
        for col in columns:
            try:
                indices.append(header.index(col))
            except ValueError:
                print(
                    f"Warning: column '{col}' not found. Available: {header}",
                    file=sys.stderr,
                )
        if not indices:
            return "(no matching columns)"
        header = [header[i] for i in indices]
        data = [[row[i] if i < len(row) else "" for i in indices] for row in rows[1:]]
    else:
        data = rows[1:]

    # Truncate rows if requested
    truncated = False
    if max_rows is not None and len(data) > max_rows:
        data = data[:max_rows]
        truncated = True

    # Compute column widths
    all_rows = [header] + data
    widths = [max(len(str(row[c])) for row in all_rows) for c in range(len(header))]

    def fmt_row(row: list[str]) -> str:
        cells = [
            _cell(str(row[c]) if c < len(row) else "", widths[c])
            for c in range(len(header))
        ]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * w for w in widths) + " |"

    lines = [fmt_row(header), separator] + [fmt_row(row) for row in data]
    if truncated:
        lines.append(f"\n_(showing {max_rows} of {len(rows) - 1} rows)_")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", help="Path to the CSV file.")
    parser.add_argument(
        "--max-rows",
        type=int,
        metavar="N",
        help="Maximum number of data rows to include.",
    )
    parser.add_argument(
        "--columns",
        metavar="COL1,COL2,...",
        help="Comma-separated list of column names to include (in order).",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
    print(csv_to_markdown(rows, columns=columns, max_rows=args.max_rows))


if __name__ == "__main__":
    main()
