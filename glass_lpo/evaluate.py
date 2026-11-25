"""Score an extraction against hand-written ground truth.

Ground truth JSON shape:
{"overall_width_mm": 900, "overall_height_mm": 2100,
 "panels": [{"panel_id": "P1", "width_mm": 900, "height_mm": 400, "quantity": 1,
             "thickness_mm": null, "glass_type": null, "hole_count": 2,
             "hole_diameters_mm": [25, 25], "notches_offcuts": 0,
             "notch_sizes": [], "hole_positions": ["50 from left, 60 from top", null]}]}
"hole_positions" is optional (only scored where given). Extra (hallucinated)
panels each count as one confident-wrong field; a value reported where the
truth is null counts as wrong if clear, flagged if CHECK.

Scoring is per field. A value marked CHECK (unclear) is counted separately from a
wrong value: CHECK on a hard value is honest; a confidently wrong value is the
failure mode that cuts unusable glass.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .schema import Drawing, Field, Panel

TOL_MM = 2.0


@dataclass
class Score:
    correct: int = 0
    flagged: int = 0          # marked CHECK (whether or not value right)
    wrong_confident: int = 0  # clear=True but wrong -> the dangerous case
    missing: int = 0          # in truth, not extracted at all
    details: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.correct + self.flagged + self.wrong_confident + self.missing


def _num_eq(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= TOL_MM


def _check(s: Score, label: str, f: Field, truth, eq=_num_eq) -> None:
    """Score one field. CHECK (clear=False) is always 'flagged', never right/wrong."""
    if truth is None:
        if f.value is None:
            return  # not on drawing, not extracted: nothing to score
        if f.clear:  # reported confidently, but the drawing doesn't have it
            s.wrong_confident += 1
            s.details.append(f"WRONG   {label}: got {f.value}, expected none (not on drawing)")
        else:
            s.flagged += 1
            s.details.append(f"CHECK   {label}: got {f.value}, expected none")
        return
    if f.value is None and not f.shown:
        s.missing += 1
        s.details.append(f"MISSING {label}: expected {truth}")
    elif not f.clear or f.value is None:
        s.flagged += 1
        ok = "right" if f.value is not None and eq(f.value, truth) else "wrong"
        s.details.append(f"CHECK   {label}: got {f.value} ({ok}), expected {truth}")
    elif eq(f.value, truth):
        s.correct += 1
    else:
        s.wrong_confident += 1
        s.details.append(f"WRONG   {label}: got {f.value}, expected {truth}")


def _size_eq(a, b) -> bool:
    """'150x100' vs '150 x 100mm' -> compare the numbers."""
    na, nb = _nums(a), _nums(b)
    return bool(na) and len(na) == len(nb) and all(abs(x - y) <= TOL_MM for x, y in zip(na, nb))


def _nums(x) -> list[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(x or ""))]


def _pos_eq(a, b) -> bool:
    """Hole positions: same numbers (any order) - '50 from left, 100 from top'."""
    na, nb = sorted(_nums(a)), sorted(_nums(b))
    return bool(na) and len(na) == len(nb) and all(abs(x - y) <= TOL_MM for x, y in zip(na, nb))


def _text_eq(a, b) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def _simple(s: Score, label: str, got, truth) -> None:
    if got == truth:
        s.correct += 1
    else:
        s.wrong_confident += 1
        s.details.append(f"WRONG   {label}: got {got}, expected {truth}")


def _score_diameters(s: Score, pid: str, p: Panel, want: list[float]) -> None:
    """Per hole: clear & matching -> correct; clear & no match -> wrong;
    unclear -> flagged (consumes one truth hole); unmatched truth -> missing."""
    remaining = [float(x) for x in want]

    def take(v) -> bool:
        for k, w in enumerate(remaining):
            if v is not None and abs(float(v) - w) <= TOL_MM:
                del remaining[k]
                return True
        return False

    unclear = []
    for j, h in enumerate(p.holes, start=1):
        d = h.diameter_mm
        if d.value is None or not d.clear:
            unclear.append((j, d.value))
            continue
        if take(d.value):
            s.correct += 1
        else:
            s.wrong_confident += 1
            s.details.append(f"WRONG   {pid} hole {j} dia: got {d.value}, not in {want}")
    for j, v in unclear:
        s.flagged += 1
        if not take(v) and remaining:
            remaining.pop(0)
        s.details.append(f"CHECK   {pid} hole {j} dia: got {v}")
    for w in remaining:
        s.missing += 1
        s.details.append(f"MISSING {pid} hole dia {w}")


def _same_size(p: Panel, tp: dict) -> bool:
    tw, th = tp.get("width_mm"), tp.get("height_mm")
    if tw is None or th is None or p.width_mm.value is None or p.height_mm.value is None:
        return False
    return abs(float(p.width_mm.value) - tw) <= TOL_MM and abs(float(p.height_mm.value) - th) <= TOL_MM


def _match_panels(drawing: Drawing, truth_panels: list[dict]) -> list[Optional[Panel]]:
    """Match each truth panel to at most one extracted panel:
    1. same width AND height (most panel IDs are not written on hand drawings,
       so P2 in the output need not be P2 in the truth),
    2. then by panel ID, 3. then by position among the still-unused panels."""
    got = drawing.panels
    used: set[int] = set()
    out: list[Optional[Panel]] = [None] * len(truth_panels)
    for i, tp in enumerate(truth_panels):
        for k, p in enumerate(got):
            if k not in used and _same_size(p, tp):
                out[i] = p
                used.add(k)
                break
    by_id: dict[str, int] = {}
    for k, p in enumerate(got):
        by_id.setdefault(p.panel_id.strip().upper(), k)
    for i, tp in enumerate(truth_panels):
        if out[i] is not None:
            continue
        k = by_id.get(str(tp["panel_id"]).strip().upper())
        if k is not None and k not in used:
            out[i] = got[k]
            used.add(k)
    for i, tp in enumerate(truth_panels):
        if out[i] is not None:
            continue
        cands = [k for k in range(len(got)) if k not in used]
        if not cands:
            break
        k = i if i in cands else cands[0]
        out[i] = got[k]
        used.add(k)
    return out


def score(drawing: Drawing, truth_path: Path | str) -> Score:
    t = json.loads(Path(truth_path).read_text())
    s = Score()
    _check(s, "overall width", drawing.overall_width_mm, t.get("overall_width_mm"))
    _check(s, "overall height", drawing.overall_height_mm, t.get("overall_height_mm"))

    truth_panels = t["panels"]
    matched = _match_panels(drawing, truth_panels)
    used_ids = {id(p) for p in matched if p is not None}
    for p in drawing.panels:
        if id(p) not in used_ids:  # hallucinated panel: a whole wrong cutting line
            s.wrong_confident += 1
            s.details.append(f"EXTRA   panel {p.panel_id} not on drawing")

    for tp, p in zip(truth_panels, matched):
        pid = str(tp["panel_id"]).upper()
        if p is None:
            n = 1 + sum(1 for k in ("width_mm", "height_mm") if tp.get(k) is not None)
            s.missing += n
            s.details.append(f"MISSING panel {pid}")
            continue
        _check(s, f"{pid} width", p.width_mm, tp.get("width_mm"))
        _check(s, f"{pid} height", p.height_mm, tp.get("height_mm"))
        tq = tp.get("quantity", 1)
        # A qty that wasn't written earns nothing: only score it when it matters.
        if (tq is not None and tq != 1) or p.quantity.shown:
            _check(s, f"{pid} qty", p.quantity, tq, eq=lambda a, b: a == b)
        _check(s, f"{pid} thickness", p.thickness_mm, tp.get("thickness_mm"))
        _check(s, f"{pid} glass type", p.glass_type, tp.get("glass_type"), eq=_text_eq)
        _simple(s, f"{pid} hole count", len(p.holes), tp.get("hole_count", 0))
        want_d = tp.get("hole_diameters_mm", [])
        if want_d or p.holes:
            _score_diameters(s, pid, p, want_d)
        for j, want in enumerate(tp.get("hole_positions") or []):
            if want is None:
                continue
            if j < len(p.holes):
                _check(s, f"{pid} hole {j + 1} position", p.holes[j].position, want, eq=_pos_eq)
            else:
                s.missing += 1
                s.details.append(f"MISSING {pid} hole {j + 1} position")
        _simple(s, f"{pid} notches/offcuts", len(p.notches_offcuts), tp.get("notches_offcuts", 0))
        for j, want in enumerate(tp.get("notch_sizes") or []):
            if j < len(p.notches_offcuts):
                _check(s, f"{pid} notch {j + 1} size", p.notches_offcuts[j].size, want, eq=_size_eq)
            else:
                s.missing += 1
                s.details.append(f"MISSING {pid} notch {j + 1} size")
    return s


def format_score(s: Score) -> str:
    if not s.total:
        return "No fields to score."
    pct = lambda n: f"{100 * n / s.total:.0f}%"
    lines = [
        f"Accuracy vs ground truth ({s.total} fields):",
        f"  correct & confident : {s.correct:>3} ({pct(s.correct)})",
        f"  flagged CHECK       : {s.flagged:>3} ({pct(s.flagged)})",
        f"  WRONG but confident : {s.wrong_confident:>3} ({pct(s.wrong_confident)})",
        f"  missing             : {s.missing:>3} ({pct(s.missing)})",
    ]
    lines += ["  " + d for d in s.details]
    return "\n".join(lines)
