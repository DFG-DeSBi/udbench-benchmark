from __future__ import annotations

import csv
from pathlib import Path


def _write_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    """Rewrite the entire CSV from scratch (safe re-entry; use for final saves)."""
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_row(
    csv_path: Path,
    row: dict[str, object],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    """Append one row to CSV; writes the header automatically on first call."""
    if fieldnames is None:
        fieldnames = list(row.keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing_fieldnames: list[str] = []
    existing_rows: list[dict[str, object]] = []
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    all_fieldnames = list(existing_fieldnames or fieldnames)
    new_cols = [key for key in fieldnames if key not in all_fieldnames]
    all_fieldnames.extend(new_cols)

    if new_cols and existing_fieldnames:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
        return

    write_header = not existing_fieldnames
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
