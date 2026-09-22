"""Small input and output helpers used by the analysis scripts.

The project stores most experiment records as JSON or JSONL files. Keeping the
file handling here avoids repeating the same code in every analysis script.
"""

from pathlib import Path
import csv
import json


def read_json(path):
    """Read one JSON file and return the decoded Python object."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path):
    """Read a JSONL file into a list of dictionaries."""
    records = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def write_json(path, data):
    """Write a Python object to JSON with readable indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def write_rows_csv(path, rows, fieldnames):
    """Write a list of dictionaries to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
