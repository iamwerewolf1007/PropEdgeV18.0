"""
PropEdge V18.0 — audit.py
Append-only CSV audit trail. Never truncated or rotated automatically.
"""

import csv
from datetime import datetime
from pathlib import Path
from config import FILE_AUDIT, get_uk


def log_event(
    batch: str,
    event: str,
    file: str = "",
    rows_before: int = 0,
    rows_after: int = 0,
    detail: str = "",
) -> None:
    """Append one audit row. Thread-safe via file append mode."""
    ts = datetime.now(get_uk()).strftime("%Y-%m-%d %H:%M:%S UK")
    row = [ts, batch, event, file, rows_before, rows_after, detail]
    write_header = not FILE_AUDIT.exists() or FILE_AUDIT.stat().st_size == 0
    with open(FILE_AUDIT, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["ts", "batch", "event", "file", "rows_before", "rows_after", "detail"])
        w.writerow(row)


def verify_no_deletion(
    path: Path,
    rows_before: int,
    batch: str,
) -> bool:
    """
    Check that CSV row count did not decrease after an append.
    Logs DELETION_ALERT if rows were lost.
    Returns True if OK, False if deletion detected.
    """
    import pandas as pd
    try:
        rows_after = len(pd.read_csv(path))
        if rows_after < rows_before:
            log_event(batch, "DELETION_ALERT", str(path.name),
                      rows_before, rows_after,
                      f"LOST {rows_before - rows_after} rows — investigate immediately")
            print(f"  ⚠ DELETION ALERT: {path.name} lost {rows_before - rows_after} rows!")
            return False
        log_event(batch, "INTEGRITY_OK", str(path.name), rows_before, rows_after)
        return True
    except Exception as e:
        log_event(batch, "INTEGRITY_CHECK_ERROR", str(path.name), rows_before, 0, str(e))
        return False
