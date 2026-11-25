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
  written inch / ft     -> converted (x25.4 / x304.8), noted - fractions such
                           as "107 7/8" or "3/8" are read from the text
  mixed / unrecognised  -> NOT converted, hard CHECK (validate.py)

Mixed units (e.g. "2.95m" next to "75cm"): extract.py reads the unit written
next to EACH number (`parse_length`) and converts that value on its own, so a
drawing does not need a single unit to be usable.

Thickness is never converted: glass thickness is always quoted in mm
("10mm toughened") whatever unit the sizes use.
Hole diameters are converted only when the written number is too small to be
mm (e.g. "Ø2.5" on a cm drawing = 25 mm); "Ø25" stays 25 mm. That rule is
flagged too, because it is a judgement.
"""
from __future__ import annotations

import re

from .schema import Drawing, Field

FACTORS = {"cm": 10.0, "m": 1000.0, "inch": 25.4, "ft": 304.8,
           "none": 10.0}  # "none" = not written -> assume cm
UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4, "ft": 304.8}
HOLE_MIN_MM = 4.0
MIN_DIM_MM, MAX_DIM_MM = 50.0, 6000.0
_NUM = re.compile(r"\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Reading a length (number + unit) from the text exactly as written
# ---------------------------------------------------------------------------
_UNICODE_FRAC = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅛": 0.125, "⅜": 0.375,
                 "⅝": 0.625, "⅞": 0.875, "⅓": 1 / 3, "⅔": 2 / 3}
_UNIT_WORD = {
    "mm": "mm", "millimetre": "mm", "millimetres": "mm", "millimeter": "mm", "millimeters": "mm",
    "cm": "cm", "centimetre": "cm", "centimetres": "cm", "centimeter": "cm", "centimeters": "cm",
    "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m", "mtr": "m", "mtrs": "m",
    "ft": "ft", "feet": "ft", "feets": "ft", "foot": "ft", "fts": "ft", "'": "ft", "′": "ft", "’": "ft",
    "in": "inch", "inch": "inch", "inches": "inch", "ins": "inch", '"': "inch", "″": "inch",
    "''": "inch", "”": "inch",
}
_FR = "".join(_UNICODE_FRAC)
# A number: "3/8" | "107 7/8" / "107-7/8" | "3⅜" / "⅜" | "2.95"
_NUMBER = re.compile(
    rf"(?P<frac>\d+)\s*/\s*(?P<den>\d+)"
    rf"|(?P<whole>\d+(?:\.\d+)?)(?:\s*[- ]\s*(?P<fn>\d+)\s*/\s*(?P<fd>\d+)|\s*(?P<uf>[{_FR}]))?"
    rf"|(?P<lone>[{_FR}])")
_UNIT_AFTER = re.compile(r"\s*(''|[a-z]+|[\"'″′’”])", re.IGNORECASE)
# Annotations that contain a slash but are not fractions: "O/A" (overall), "c/c".
_NOT_FRACTION = re.compile(r"\b[a-z]\s*/\s*[a-z]\b", re.IGNORECASE)


def _number_value(m: re.Match) -> float:
    if m.group("frac"):
        return float(m.group("frac")) / float(m.group("den"))
    if m.group("lone"):
        return _UNICODE_FRAC[m.group("lone")]
    v = float(m.group("whole"))
    if m.group("fn"):
        v += float(m.group("fn")) / float(m.group("fd"))
    elif m.group("uf"):
        v += _UNICODE_FRAC[m.group("uf")]
    return v


def parse_length(text: str) -> tuple[float | None, str | None]:
    """Read a length as written: returns (number, unit) with the number in the
    WRITTEN unit, or (number, None) if no unit is written next to it.

      "2.95m" -> (2.95, "m")       "75cm" -> (75, "cm")     "1200" -> (1200, None)
      "107 7/8" -> (107.875, None) "3/8\"" -> (0.375, "inch")
      "4 feet" -> (4, "ft")        "4'6\"" -> (54, "inch")  "2050 O/A" -> (2050, None)
    """
    t = _NOT_FRACTION.sub(" ", text or "")
    found: list[tuple[float, str | None]] = []
    for m in _NUMBER.finditer(t):
        unit = None
        u = _UNIT_AFTER.match(t, m.end())
        if u:
            unit = _UNIT_WORD.get(u.group(1).lower())
        found.append((_number_value(m), unit))
    if not found:
        return None, None
    if len(found) >= 2 and found[0][1] == "ft" and found[1][1] in ("inch", None):
        return found[0][0] * 12 + found[1][0], "inch"      # 4' 6" -> 54 inches
    return found[0]


def has_fraction(text: str) -> bool:
    t = _NOT_FRACTION.sub(" ", text or "")
    return bool(re.search(rf"\d\s*/\s*\d|[{_FR}]", t))


def units_in_text(text: str) -> set[str]:
    """Every unit word in free text such as "m and cm" -> {"m", "cm"}."""
    out = set()
    for tok in re.findall(r"''|[a-z]+|[\"'″′’”]", (text or "").lower()):
        if tok in _UNIT_WORD:
            out.add(_UNIT_WORD[tok])
    return out


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
