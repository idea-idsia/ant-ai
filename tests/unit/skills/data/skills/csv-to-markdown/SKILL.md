---
name: csv-to-markdown
description: Convert a CSV file to a Markdown table. Use when the user shares a CSV file and wants to display, inspect, or discuss its contents as a formatted table.
compatibility: Python 3.8+ standard library only (no extra packages required)
metadata:
  author: ant-ai
  version: "1.0"
---

# CSV to Markdown

Use this skill when the user provides a path to a CSV file and wants to view or work with its contents as a Markdown table.

## Steps

1. Run the conversion script, passing the CSV file path:

   ```
   python scripts/csv_to_markdown.py path/to/data.csv
   ```

2. To limit the output to a specific number of rows:

   ```
   python scripts/csv_to_markdown.py path/to/data.csv --max-rows 20
   ```

3. To include only certain columns (in a given order):

   ```
   python scripts/csv_to_markdown.py path/to/data.csv --columns name,age,email
   ```

4. Present the Markdown table output to the user. If the file was truncated, mention the total row count.

## Script reference

`scripts/csv_to_markdown.py` — pure Python (stdlib only). Reads a CSV, aligns columns, and writes a Markdown table to stdout.

```
usage: csv_to_markdown.py [-h] [--max-rows N] [--columns COL1,COL2,...] file

positional arguments:
  file                   Path to the CSV file.

options:
  --max-rows N           Maximum number of data rows to include.
  --columns COL1,COL2,...  Comma-separated list of columns to include (in order).
```

## Example output

Given `users.csv`:

```
name,age,city
Alice,30,Zurich
Bob,25,Milan
Carol,35,Paris
```

Running `python scripts/csv_to_markdown.py users.csv` produces:

```markdown
| name  | age | city   |
| ----- | --- | ------ |
| Alice | 30  | Zurich |
| Bob   | 25  | Milan  |
| Carol | 35  | Paris  |
```
