"""Sequential LPO numbers: LPO-YYYYMMDD-NNN, resetting each day.

The counter lives in `<outputs_dir>/.lpo_counter.json` as {"YYYYMMDD": last_n}.
Simple and good enough for a single-user demo (not safe for concurrent writers).
Directories starting with "_" (fixtures, in-progress runs) are ignored.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

COUNTER_FILE = ".lpo_counter.json"


def _ignored(p: Path, root: Path) -> bool:
    """True if any directory between root and p starts with '_'."""
    return any(part.startswith("_") for part in p.relative_to(root).parts[:-1]) or p.name.startswith("_")


def new_lpo_number(date: _dt.date | None = None, outputs_dir: str | Path = "outputs") -> str:
    date = date or _dt.date.today()
    day = date.strftime("%Y%m%d")
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    counter_path = out / COUNTER_FILE

    counts: dict[str, int] = {}
    if counter_path.exists():
        try:
            counts = {k: int(v) for k, v in json.loads(counter_path.read_text()).items()}
        except (ValueError, TypeError, AttributeError):
            counts = {}

    # Also respect any LPO files already on disk, in case the counter was lost.
    prefix = f"LPO-{day}-"
    for p in out.rglob(f"{prefix}*"):
        if _ignored(p, out):
            continue
        tail = p.stem[len(prefix):len(prefix) + 3]
        if tail.isdigit():
            counts[day] = max(counts.get(day, 0), int(tail))

    n = counts.get(day, 0) + 1
    counts[day] = n
    counter_path.write_text(json.dumps(counts, indent=2, sort_keys=True))
    return f"{prefix}{n:03d}"
