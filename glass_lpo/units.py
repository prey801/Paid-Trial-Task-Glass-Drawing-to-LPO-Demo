"""Unit handling: everything downstream (Excel, PDF, rules) is in millimetres.

Client decision: when NO unit is written on a drawing, assume centimetres
(local fabricators write "90 x 210" for a 900 x 2100 mm door) and convert to
mm - never silently: the drawing gets a flag saying it was assumed.

  written mm            -> unchanged
  written cm / m        -> converted (x10 / x1000), noted
  nothing written       -> assumed cm, converted x10, flagged (soft)
                           SAFETY RULE: if x10 would push any size past the
                           6000 mm sheet limit while the numbers as written are
                           sensible mm (50-6000), keep them as mm instead -
                           also flagged, so a human confirms the unit.
  inch / unrecognised   -> NOT converted, hard CHECK (validate.py)

Thickness is never converted: glass thickness is always quoted in mm
("10mm toughened") whatever unit the sizes use.
Hole diameters are converted only when the written number is too small to be
mm (e.g. "Ø2.5" on a cm drawing = 25 mm); "Ø25" stays 25 mm. That rule is
flagged too, because it is a judgement.
"""
from __future__ import annotations

import re

from .schema import Drawing, Field

FACTORS = {"cm": 10.0, "m": 1000.0, "none": 10.0}  # "none" = not written -> assume cm
HOLE_MIN_MM = 4.0
MIN_DIM_MM, MAX_DIM_MM = 50.0, 6000.0
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _scale(f: Field, k: float) -> None:
    if f.value is not None:
        f.value = round(float(f.value) * k, 1)


def _scale_text(f: Field, k: float) -> None:
    """Scale every number inside free text such as '50 from left' or '15x10'."""
    if isinstance(f.value, str) and f.value:
        f.value = _NUM.sub(lambda m: f"{float(m.group()) * k:g}", f.value)


def convert_to_mm(d: Drawing) -> None:
    """Convert `d` in place to mm. Idempotent: sets units_written='mm' and
    records what it did in `units_converted_from`."""
    unit = (d.units_written or "").strip().lower()
    k = FACTORS.get(unit)
    if k is None or d.units_converted_from:
        return

    if unit == "none":
        sizes = [f.value for f in (d.overall_width_mm, d.overall_height_mm) if f.value is not None]
        sizes += [f.value for p in d.panels for f in (p.width_mm, p.height_mm) if f.value is not None]
        if sizes and max(sizes) * k > MAX_DIM_MM and all(MIN_DIM_MM <= v <= MAX_DIM_MM for v in sizes):
            d.units_written = "mm"
            d.units_converted_from = ("no unit written - as cm the sizes would exceed "
                                      f"{MAX_DIM_MM:g} mm, so read as mm (not converted)")
            return

    _scale(d.overall_width_mm, k)
    _scale(d.overall_height_mm, k)
    hole_rule_used = False
    for p in d.panels:
        _scale(p.width_mm, k)
        _scale(p.height_mm, k)
        for h in p.holes:
            v = h.diameter_mm.value
            if v is not None and v < HOLE_MIN_MM:   # too small to be mm -> it's in the drawing unit
                h.diameter_mm.value = round(v * k, 1)
                hole_rule_used = True
            _scale_text(h.position, k)
        for feat in p.notches_offcuts + p.grooves_edgework:
            _scale_text(feat.size, k)
            _scale_text(feat.position, k)

    d.units_converted_from = (
        "no unit written - assumed cm, sizes converted to mm (x10)" if unit == "none"
        else f"written in {unit} - sizes converted to mm (x{k:g})")
    d.units_written = "mm"
    if hole_rule_used:
        d.extraction_issues.append(
            f"hole diameter(s) under {HOLE_MIN_MM:g} read as {unit if unit != 'none' else 'cm'} and converted - confirm")
