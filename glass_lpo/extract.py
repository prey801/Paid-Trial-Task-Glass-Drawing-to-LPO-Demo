"""Vision-model extraction: phone photo of a hand-drawn glass drawing -> `Drawing`.

Two stages, so everything except the live API call is testable offline:

1. Gemini returns JSON matching a flat "wire" schema (`WireDrawing`). Every
   number comes back as {raw_text, other_possible_readings, value, legible}:
   the model transcribes first, lists any other way the characters could be
   read, then interprets - so it can say "I see something here but can't read
   it" instead of guessing.
2. `wire_to_drawing()` converts that into the shared `schema.Drawing` types,
   mapping legibility onto `Field.clear` / `Field.shown`:
     written & legible     -> value, clear=True,  shown=True
     written but unclear   -> value/None, clear=False, shown=True
     not written at all    -> None,  clear=False, shown=False
   Anything derived rather than read is clear=False (or not_shown).

Core rule: never guess silently. Anything the model did not read clearly
ends up with clear=False and is rendered as CHECK downstream.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError
from pydantic import Field as PydField

from .schema import Drawing, Feature, Field, Hole, Panel
from .units import FEET_ASSUMED, UNIT_TO_MM, has_fraction, only_feet_plausible, parse_length, units_in_text

# Fallback order, first model tried first. 3.5-flash leads because it has been
# the most reliable (3.8/3.7 are often 503-busy or rate-limited).
DEFAULT_MODEL = "gemini-3.5-flash,gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash-lite"
MIN_CALL_TIMEOUT_S = 10.0
DEFAULT_DEADLINE_S = 150.0   # whole extraction, all models (env GEMINI_DEADLINE_S)
DEFAULT_CALL_TIMEOUT_S = 75.0  # one API call (env GEMINI_CALL_TIMEOUT_S)

Legibility = Literal["clear", "unclear", "not_shown"]
DimSource = Literal["own_dimension", "overall_dimension", "inferred", "not_shown"]


# --------------------------------------------------------------------------
# Wire schema (what Gemini returns). Flat, no generics, no recursion.
# Field order matters: transcribe (raw_text) before interpreting (value).
# --------------------------------------------------------------------------
class WireNumber(BaseModel):
    raw_text: str = ""
    other_possible_readings: list[str] = PydField(default_factory=list)
    value: Optional[float] = None
    legible: Legibility = "not_shown"


class WireText(BaseModel):
    raw_text: str = ""
    other_possible_readings: list[str] = PydField(default_factory=list)
    value: Optional[str] = None
    legible: Legibility = "not_shown"


class WireHole(BaseModel):
    count: int = 1  # number of identical holes this entry describes (e.g. "4 holes Ø12")
    diameter_mm: WireNumber = PydField(default_factory=WireNumber)
    from_horizontal_edge: Literal["left", "right", "unknown"] = "unknown"
    horizontal_offset_mm: WireNumber = PydField(default_factory=WireNumber)
    from_vertical_edge: Literal["top", "bottom", "unknown"] = "unknown"
    vertical_offset_mm: WireNumber = PydField(default_factory=WireNumber)
    position_text: str = ""  # free description as drawn, e.g. "centre of panel"


class WireFeature(BaseModel):
    kind: str = ""  # notch | offcut | cutout | groove | polish | bevel | edge work | ...
    width_mm: WireNumber = PydField(default_factory=WireNumber)
    height_mm: WireNumber = PydField(default_factory=WireNumber)
    size: WireText = PydField(default_factory=WireText)  # only if not a width x height pair
    position: WireText = PydField(default_factory=WireText)
    note: str = ""


class WirePanel(BaseModel):
    panel_id: str = ""
    id_written_on_drawing: bool = False
    width_mm: WireNumber = PydField(default_factory=WireNumber)
    width_source: DimSource = "not_shown"
    height_mm: WireNumber = PydField(default_factory=WireNumber)
    height_source: DimSource = "not_shown"
    quantity: WireNumber = PydField(default_factory=WireNumber)
    thickness_mm: WireNumber = PydField(default_factory=WireNumber)
    glass_type: WireText = PydField(default_factory=WireText)
    holes: list[WireHole] = PydField(default_factory=list)
    notches_offcuts: list[WireFeature] = PydField(default_factory=list)
    grooves_edgework: list[WireFeature] = PydField(default_factory=list)
    notes: str = ""


class WireDrawing(BaseModel):
    units_written: str = ""  # e.g. "mm", "cm", "none written"
    overall_width_mm: WireNumber = PydField(default_factory=WireNumber)
    overall_height_mm: WireNumber = PydField(default_factory=WireNumber)
    panels: list[WirePanel] = PydField(default_factory=list)
    general_notes: str = ""


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
TWO_IMAGES_NOTE = """\
Image 1 is the ORIGINAL phone photo. Image 2 is an ENHANCED copy of the same
photo (contrast/sharpening). Enhancement can distort digits. Read every value
on BOTH images. If the two images disagree about a character, mark the value
"unclear" and list both readings (one in value, the other in
other_possible_readings).
"""

PROMPT = """\
You are reading a phone photo of a HAND-DRAWN glass cutting drawing sent to a
glass processing workshop in Nairobi. Your output is used to cut and toughen
real glass, so a wrong number wastes material. A human checker reviews every
value you mark as unclear. Accuracy and honesty matter more than completeness.

GOLDEN RULE - REPORT ONLY WHAT IS WRITTEN
- For every value: FIRST copy the characters exactly as written into raw_text
  (including units/symbols, e.g. "1200", "Ø12", "12mm toughened", "x2"),
  THEN list in other_possible_readings every other plausible reading of those
  same characters (e.g. "1700" for a "1200" whose 2 might be a 7), THEN give
  your best reading in value, THEN set legible.
- Never compute, infer, estimate or "fix" a value and present it as read.
- legible = "clear" ONLY if the characters are unambiguous and
  other_possible_readings is empty.
- legible = "unclear" if the number is smudged, cut off, overwritten, faint,
  or could be read more than one way (1 vs 7, 0 vs 6, 5 vs 6, 3 vs 8, 4 vs 9,
  decimal point vs mark). Put your best reading in value (or null if you have
  none) and the alternatives in other_possible_readings.
- legible = "not_shown" (value null, raw_text "") if the drawing does not
  show it at all.

GLASS DRAWING CONVENTIONS
- units_written: the unit written on the drawing for SIZES ("mm", "cm", "m",
  "inch", "ft"), "mixed" if different sizes use different units (e.g. "2.95m"
  and "75cm"), or "none written" if no unit appears anywhere. Do not assume a
  unit. Report every number exactly as written - never convert (e.g. "90"
  stays 90, "2.95m" stays 2.95, "4 feet" stays 4).
- raw_text MUST include the unit written next to that number, if any
  ("2.95m", "75cm", "4 feet", "3/8\""), and fractions exactly as written
  ("107 7/8") - the unit is read from raw_text.
  Glass thickness is always mm. A hole diameter is reported as written
  (e.g. "Ø25" -> 25, "Ø2.5" -> 2.5).
- Width = horizontal dimension, height = vertical dimension as drawn.
- Panel width/height and width_source/height_source:
  "own_dimension" = a dimension line or number drawn for THIS panel alone.
  "overall_dimension" = the only number is the overall dimension spanning
  several panels / the whole assembly. "inferred" = you worked it out.
  "not_shown" = nothing. NEVER copy the overall dimension into a panel and
  never subtract/divide to get a panel size: if the panel has no own
  dimension, leave its value null with legible "not_shown" and set the source
  to "overall_dimension", "inferred" or "not_shown".
- Overall width/height: a dimension line spanning the whole assembly. Report
  it in overall_width_mm / overall_height_mm.
- Reading dimensions on hand sketches: a dimension may be a line with arrows
  or ticks, or just a number written beside an edge. A number written along
  the LEFT or RIGHT side of the whole outline is a HEIGHT; along the TOP or
  BOTTOM it is a WIDTH. Side dimensions are often written vertically or
  rotated, and arrows may be missing - read them anyway. Check every side of
  every outline for a number before deciding a size is "not_shown".
- Several separate items on one sheet (e.g. windows and a door drawn side by
  side): list the glass panels of every item, and say in each panel's notes
  which item it belongs to. Report overall_width_mm / overall_height_mm for
  ONE item only (never mix one item's width with another's height), and
  write each other item's overall size in general_notes.
- A number written in a circle, or "(2)", beside an item or its name (e.g.
  "Windows (2)") is the quantity of that item: it applies to each glass
  panel of that item.
- Words written inside an area that name what fills it ("glass", "alumin",
  "panel", "wood") tell you whether the area is glass. Only glass areas are
  panels. The word "glass" alone is not a glass_type.
- When an item is divided into areas (e.g. a door with glass at the top and
  aluminium at the bottom), the item's outer dimensions are NOT the glass
  size. Unless a dimension line is drawn for the glass area itself, its
  width/height are null with source "overall_dimension" (or "not_shown").
- "Ø", "dia", "D" before/after a number means a hole DIAMETER.
- A thickness like "12mm", "10mm", "6mm" written next to glass words is the
  GLASS THICKNESS, not a panel dimension.
- glass_type: ONLY descriptive glass words such as clear, frosted, tinted
  (bronze/grey/green/blue), toughened/tempered, laminated, reflective,
  mirror, low-e, float, wired, patterned - written next to a panel or in a
  specification note. Titles, headings, company/client names and labels
  (e.g. "HARD - GLASS", "GLASS ORDER", "Shower") are NOT a glass type. If no
  such words are written, glass_type is "not_shown".
- If thickness or glass type is written once for the whole drawing, apply it
  to every panel it clearly covers; if unsure which panels it covers, mark
  those panels' value "unclear".
- Panels: use the label written on the drawing (e.g. "A", "P1", "Door") and
  set id_written_on_drawing=true. If panels are not labelled, assign P1, P2,
  P3... left-to-right, then top-to-bottom, and set id_written_on_drawing=false.
- Quantity markers such as "x2", "2 No.", "2 pcs", "QTY 2" belong to the
  panel they are next to. If no quantity is written, quantity legible =
  "not_shown".
- Holes: one entry per distinct hole (or one entry with count > 1 when the
  drawing says e.g. "4 holes Ø12" with the same position description).
  Positions come from dimension lines running from the hole centre to the
  nearest edges: give the offset and which edge it is measured from
  (left/right, top/bottom). If an offset is not dimensioned, leave it
  not_shown (do not estimate it) and describe the position in position_text.
- notches_offcuts: notches, corner cut-outs, offcuts, slots, cut-outs for
  hinges/locks/handles (kind = "notch", "offcut", "cutout", ...). Put the
  two numbers of a size like "150x100" in width_mm and height_mm (as written,
  first number in width_mm). Use size only for a size that is not a width x
  height pair (e.g. "R50"); otherwise size is "not_shown".
- grooves_edgework: grooves and edge processing such as polish, flat polish,
  pencil polish, bevel, arris/seaming (kind = "groove", "polish", "bevel",
  "edge work", ...). Put which edges in position (e.g. "all edges", "top").
- Anything else useful to the cutter (handwritten instructions, client name,
  arrows you could not interpret) goes in notes / general_notes. Mention
  anything you could not read in notes so a human can look.

Return JSON only, matching the response schema.
"""


# --------------------------------------------------------------------------
# Wire -> Drawing
# --------------------------------------------------------------------------
def _fmt_num(v: Optional[float]) -> str:
    if v is None:
        return "?"
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _value_str(w: WireNumber | WireText) -> str:
    if isinstance(w, WireNumber):
        return _fmt_num(w.value)
    return (w.value or "").strip() or "?"


def _alternatives(w: WireNumber | WireText) -> list[str]:
    """Other readings the model listed, minus blanks and the chosen value itself."""
    own = _value_str(w)
    out: list[str] = []
    for a in w.other_possible_readings:
        a = str(a).strip()
        if a and a != own and a not in out:
            out.append(a)
    return out


def _is_absent(w: WireNumber | WireText) -> bool:
    """Legibly not on the drawing: nothing written, no value, no alternatives."""
    has_value = w.value is not None and (not isinstance(w.value, str) or w.value.strip())
    return (w.legible == "not_shown" and not has_value
            and not w.raw_text.strip() and not _alternatives(w))


def _num_field(w: WireNumber, *, as_int: bool = False) -> Field:
    """Map a wire number onto a schema Field (see module docstring for the states).

    Not written at all -> shown=False even for required values (width, height,
    hole dia); exporters use `display(required=True)` so those still show CHECK.
    """
    if _is_absent(w):
        return Field(value=None, clear=False, shown=False)
    value = w.value
    if value is not None and as_int:
        if float(value).is_integer():
            value = int(value)
        else:  # "2.5 pieces" makes no sense - keep the reading but force a check
            return Field(value=int(round(value)), clear=False, shown=True)
    clear = w.legible == "clear" and value is not None and not _alternatives(w)
    return Field(value=value, clear=clear, shown=True)


def _text_field(w: WireText) -> Field:
    if _is_absent(w):
        return Field(value=None, clear=False, shown=False)
    value = (w.value or "").strip() or None
    clear = w.legible == "clear" and value is not None and not _alternatives(w)
    return Field(value=value, clear=clear, shown=True)


def _doubt_note(label: str, w: WireNumber | WireText) -> Optional[str]:
    """Human-readable reason a value is not clear (None if it is clear/absent)."""
    if _is_absent(w):
        return None
    alts = _alternatives(w)
    raw = w.raw_text.strip()
    if alts:
        return f"{label} read {_value_str(w)}, could be {' or '.join(alts)}"
    if w.legible != "clear" or w.value is None:
        return f"{label} written '{raw}'" if raw else f"{label} unclear"
    return None


def _panel_dim(label: str, w: WireNumber, source: str, notes: list[str]) -> Field:
    """Panel width/height: only a dimension drawn for this panel counts as read."""
    if source == "own_dimension":
        n = _doubt_note(label, w)
        if n:
            notes.append(n)
        return _num_field(w)
    if not _is_absent(w):  # the model put a number here that wasn't this panel's own
        how = source.replace("_", " ") if source != "not_shown" else "no own dimension"
        notes.append(f"{label} not written for this panel ({how} {_value_str(w)} not used)")
    return Field(value=None, clear=False, shown=False)


def _hole_position(h: WireHole, label: str, notes: list[str]) -> Field:
    parts: list[str] = []
    clear = True
    axes = 0
    for edge, off in (
        (h.from_horizontal_edge, h.horizontal_offset_mm),
        (h.from_vertical_edge, h.vertical_offset_mm),
    ):
        if _is_absent(off):
            continue
        axes += 1
        n = _doubt_note(f"{label} offset", off)
        if n:
            notes.append(n)
        if not _num_field(off).clear or edge == "unknown":
            clear = False
        edge_txt = f"from {edge}" if edge != "unknown" else "from ? edge"
        parts.append(f"{_fmt_num(off.value)} {edge_txt}")
    text = h.position_text.strip()
    if text:
        parts.append(text)
    if axes == 0:
        if text:  # drawn but not dimensioned -> needs a human
            return Field(value=text, clear=False, shown=True)
        return Field(value=None, clear=False, shown=False)
    if axes == 1:  # a point needs two offsets; the other one would be a guess
        clear = False
        notes.append(f"{label} position dimensioned on one axis only")
    return Field(value=", ".join(parts), clear=clear, shown=True)


SIZED_FEATURES = {"notch", "offcut", "cutout", "cut-out", "cut out", "slot"}


def _feature(w: WireFeature) -> Feature:
    kind = (w.kind or "other").strip().lower()
    notes: list[str] = [w.note.strip()] if w.note.strip() else []
    if not (_is_absent(w.width_mm) and _is_absent(w.height_mm)):
        fw, fh = _num_field(w.width_mm), _num_field(w.height_mm)
        for label, n in (("width", w.width_mm), ("height", w.height_mm)):
            d = _doubt_note(label, n)
            if d:
                notes.append(d)
        size = Field(value=f"{_fmt_num(fw.value)}x{_fmt_num(fh.value)}",
                     clear=fw.clear and fh.clear, shown=True)
    else:
        size = _text_field(w.size)
        d = _doubt_note("size", w.size)
        if d:
            notes.append(d)
        if not size.shown and kind in SIZED_FEATURES:  # a notch must have a size
            size = Field(value=None, clear=False, shown=True)
            notes.append("size not written")
    d = _doubt_note("position", w.position)
    if d:
        notes.append(d)
    return Feature(kind=kind, size=size, position=_text_field(w.position), note="; ".join(notes))


MAX_HOLES_PER_ENTRY = 50

_NO_UNITS = {"", "none", "none written", "not written", "not shown", "n/a", "na"}


def normalise_units(text: str) -> str:
    """'MM' -> 'mm', 'Inches' -> 'inch', 'feet' -> 'ft', '' / 'none written' ->
    'none' (units.py then assumes cm), two different units ('m and cm') ->
    'mixed', anything unrecognised -> 'unknown'.
    Values are returned as written; conversion happens later."""
    t = (text or "").strip().lower()
    if t in _NO_UNITS:
        return "none"
    found = units_in_text(t)
    if len(found) == 1:
        return found.pop()
    return "mixed" if found else "unknown"


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.5, 0.005 * abs(b))


def _to_mm(w: WireNumber, num: float, unit: str) -> None:
    """Set w.value to `num` (in `unit`) as mm. If the model's own value matches
    neither the written number nor its mm conversion, it disagrees with its own
    transcription -> unclear."""
    mm = round(num * UNIT_TO_MM[unit], 1)
    v = w.value
    if v is not None and not (_close(v, num) or _close(v, mm)):
        w.legible = "unclear"
    w.value = mm
    alts = []
    for a in w.other_possible_readings:  # keep alternatives in the same unit as value
        an, au = parse_length(str(a))
        alts.append(_fmt_num(round(an * UNIT_TO_MM[au or unit], 1)) if an is not None else a)
    w.other_possible_readings = alts


def _apply_written_units(wire: WireDrawing, notes_by_panel: dict[int, list[str]]) -> tuple[str, str, list[str]]:
    """Convert sizes using the unit written NEXT TO each number (raw_text).

    Handles drawings that mix units ("2.95m" beside "75cm") and imperial
    sizes ("4 feet", "107 7/8"). Returns (units_written, units_converted_from,
    drawing issues). When no number carries its own unit, nothing is changed
    and the drawing-level unit is converted later by units.convert_to_mm.
    """
    unit = normalise_units(wire.units_written)
    dims: list[tuple[str, WireNumber, int | None]] = [
        ("overall width", wire.overall_width_mm, None), ("overall height", wire.overall_height_mm, None)]
    for i, wp in enumerate(wire.panels):
        dims += [("width", wp.width_mm, i), ("height", wp.height_mm, i)]

    own: set[str] = set()
    unitless: list[tuple[str, WireNumber, int | None, float | None]] = []
    for label, w, i in dims:
        if _is_absent(w) or (w.value is None and not w.raw_text.strip()):
            continue
        num, u = parse_length(w.raw_text)
        if u is not None and num is not None:
            _to_mm(w, num, u)
            own.add(u)
        elif w.value is not None or num is not None:
            unitless.append((label, w, i, num))

    if not own:
        if unit == "none":
            fr = [w.raw_text.strip() for _, w, _ in dims if has_fraction(w.raw_text)]
            if fr:  # metric sizes are not written as fractions: feet or inches, but which?
                written = [(w, num if num is not None else w.value) for _, w, _, num in unitless]
                if only_feet_plausible([v for _, v in written]):
                    for w, v in written:
                        _to_mm(w, v, "ft")
                        w.legible = "unclear"
                    return "mm", FEET_ASSUMED, []
                for _, w, _ in dims:
                    if w.value is not None:
                        w.legible = "unclear"
                return "unknown", "", [
                    f"sizes written with fractions (e.g. '{fr[0]}') but no unit - fractions mean "
                    "feet or inches, not cm; sizes left as written (CHECK)"]
        return unit, "", []

    issues: list[str] = []
    used = sorted(own)
    if unitless and unit in UNIT_TO_MM:
        for label, w, i, num in unitless:
            _to_mm(w, num if num is not None else w.value, unit)
        used = sorted(own | {unit})
    elif unitless:  # no unit beside these numbers and none for the drawing - can't convert safely
        for label, w, i, num in unitless:
            w.legible = "unclear"
            msg = f"{label}: no unit written next to it - left as written (CHECK)"
            if i is None:
                issues.append(msg)
            else:
                notes_by_panel.setdefault(i, []).append(msg)
    conv = ("sizes written in " + " and ".join(used)
            + " - each converted to mm using the unit written beside it")
    return "mm", conv, issues


def _thickness_from_inches(w: WireNumber, notes: list[str]) -> None:
    """Glass thickness is quoted in mm - but a "3/8" or 0.375 is inches.
    Convert to mm and force a CHECK rather than report a 0.375 mm glass."""
    if _is_absent(w):
        return
    num, u = parse_length(w.raw_text)
    v = num if num is not None else w.value
    if v is None:
        return
    if u in ("inch", "ft") or (u is None and (has_fraction(w.raw_text) or v < 1)):
        mm = round(v * UNIT_TO_MM["ft" if u == "ft" else "inch"], 1)
        w.value, w.legible = mm, "unclear"
        w.other_possible_readings = []
        notes.append(f'thickness written "{w.raw_text.strip() or _fmt_num(v)}" - read as inches = {mm:g} mm (CHECK)')


def wire_to_drawing(wire: WireDrawing | dict | str) -> Drawing:
    """Convert the model's wire JSON (object, dict or JSON string) to a Drawing."""
    if isinstance(wire, str):
        wire = parse_wire_json(wire)
    elif isinstance(wire, dict):
        wire = WireDrawing.model_validate(wire)
    else:
        wire = wire.model_copy(deep=True)  # unit handling below edits the numbers

    unit_notes: dict[int, list[str]] = {}
    units_written, units_conv, unit_issues = _apply_written_units(wire, unit_notes)

    panels: list[Panel] = []
    for i, wp in enumerate(wire.panels, start=1):
        notes: list[str] = list(unit_notes.get(i - 1, []))
        _thickness_from_inches(wp.thickness_mm, notes)
        issues: list[str] = []
        pid = wp.panel_id.strip() or f"P{i}"
        if not wp.id_written_on_drawing:
            notes.append("ID assigned (not written)")

        width = _panel_dim("width", wp.width_mm, wp.width_source, notes)
        height = _panel_dim("height", wp.height_mm, wp.height_source, notes)
        for label, w in (("qty", wp.quantity), ("thickness", wp.thickness_mm), ("glass type", wp.glass_type)):
            n = _doubt_note(label, w)
            if n:
                notes.append(n)

        holes: list[Hole] = []
        for j, wh in enumerate(wp.holes, start=1):
            label = f"hole {j}"
            n = _doubt_note(f"{label} dia", wh.diameter_mm)
            if n:
                notes.append(n)
            count = wh.count
            if count < 1 or count > MAX_HOLES_PER_ENTRY:
                issues.append(f"hole count '{wh.count}' implausible (CHECK)")
                count = 1
            pos = _hole_position(wh, label, notes)
            dia = _num_field(wh.diameter_mm)
            holes.extend(Hole(diameter_mm=dia.model_copy(), position=pos.model_copy()) for _ in range(count))

        if wp.notes.strip():
            notes.append(wp.notes.strip())

        panels.append(
            Panel(
                panel_id=pid,
                width_mm=width,
                height_mm=height,
                quantity=(
                    Field(value=1, clear=True, shown=False)  # no qty written = 1 piece
                    if _is_absent(wp.quantity)
                    else _num_field(wp.quantity, as_int=True)
                ),
                thickness_mm=_num_field(wp.thickness_mm),
                glass_type=_text_field(wp.glass_type),
                holes=holes,
                notches_offcuts=[_feature(f) for f in wp.notches_offcuts],
                grooves_edgework=[_feature(f) for f in wp.grooves_edgework],
                notes="; ".join(notes),
                extraction_issues=issues,
            )
        )

    general: list[str] = []
    for label, w in (("overall width", wire.overall_width_mm), ("overall height", wire.overall_height_mm)):
        n = _doubt_note(label, w)
        if n:
            general.append(n)
    if wire.general_notes.strip():
        general.append(wire.general_notes.strip())

    return Drawing(
        overall_width_mm=_num_field(wire.overall_width_mm),
        overall_height_mm=_num_field(wire.overall_height_mm),
        panels=panels,
        general_notes="; ".join(general),
        units_written=units_written,
        units_converted_from=units_conv,
        extraction_issues=unit_issues,
    )


# --------------------------------------------------------------------------
# Gemini call
# --------------------------------------------------------------------------
class ExtractionError(RuntimeError):
    """Extraction failed (no API key, API error, or unusable model output)."""


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_wire_json(text: str) -> WireDrawing:
    """Parse the model's JSON text (tolerates ```json fences)."""
    cleaned = _FENCE_RE.sub("", text or "").strip()
    if not cleaned:
        raise ValueError("empty model response")
    return WireDrawing.model_validate(json.loads(cleaned))


def _mime_type(path: Path) -> str:
    mt, _ = mimetypes.guess_type(path.name)
    return mt if mt and mt.startswith("image/") else "image/jpeg"


def _prepare_image(path: Path, max_side: int = 2048, quality: int = 90) -> tuple[bytes, str]:
    """Re-encode as a compact JPEG for upload.

    Enhanced PNGs are 1-5 MB; on a slow connection the upload alone times out,
    and Gemini downsamples large images internally anyway. ~2000px JPEG q90
    keeps hand-written digits sharp at ~10% of the size.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return path.read_bytes(), _mime_type(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return path.read_bytes(), _mime_type(path)
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return (buf.tobytes(), "image/jpeg") if ok else (path.read_bytes(), _mime_type(path))


def _require_all(schema) -> None:
    """Mark every property required (recursively) so Gemini never omits a key
    - a missing key would otherwise silently become "not shown"."""
    if schema is None:
        return
    # Defaults/titles are for our pydantic parsing only; keep the schema lean.
    schema.default = None
    schema.title = None
    if schema.properties:
        schema.required = list(schema.properties.keys())
        for sub in schema.properties.values():
            _require_all(sub)
    _require_all(schema.items)


def gemini_response_schema(client=None):
    """Gemini `types.Schema` for `WireDrawing`, with all fields required.

    Uses the SDK's private `_transformers.t_schema` (hence google-genai is
    pinned to a minor version in pyproject.toml).
    """
    from google.genai import _transformers

    schema = _transformers.t_schema(getattr(client, "_api_client", None), WireDrawing)
    _require_all(schema)
    return schema


# finish_reason / block_reason values that mean "the model refused or was cut off".
_REFUSED = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION",
            "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "MODEL_ARMOR", "JAILBREAK"}


def _enum_name(x) -> str:
    if x is None:
        return ""
    return getattr(x, "name", None) or str(x).split(".")[-1]


def _check_finished(response, model: str) -> None:
    """Raise a clear ExtractionError if the reply was blocked or truncated."""
    feedback = getattr(response, "prompt_feedback", None)
    blocked = _enum_name(getattr(feedback, "block_reason", None))
    if blocked and blocked != "BLOCKED_REASON_UNSPECIFIED":
        raise ExtractionError(f"{model} blocked the request ({blocked}); the photo could not be read.")
    candidates = getattr(response, "candidates", None) or []
    reason = _enum_name(getattr(candidates[0], "finish_reason", None)) if candidates else ""
    if reason == "MAX_TOKENS":
        raise ExtractionError(f"{model} reply was cut off (MAX_TOKENS) - drawing too big for one read.")
    if reason in _REFUSED:
        raise ExtractionError(f"{model} stopped for safety reasons ({reason}); the photo could not be read.")


def _generate(client, model: str, contents: list, timeout_s: float) -> WireDrawing:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=gemini_response_schema(client),
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        ),
    )
    _check_finished(response, model)
    # With a types.Schema response_schema the SDK does not build our pydantic
    # object, so always parse the text ourselves.
    return parse_wire_json(response.text or "")


def _classify(exc: Exception) -> str:
    """'fatal' (stop now), 'retry' (same model once more) or 'next' (next model)."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        if code in (400, 401, 403):
            return "fatal"   # bad key / bad request - another model won't help
        if code == 408 or code >= 500:
            return "retry"   # busy / transient
        return "next"        # 404 model not found, 429 quota, ...
    if isinstance(exc, ValueError):  # JSON / pydantic validation failure
        return "next"
    if isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__module__.startswith(("httpx", "httpcore")):
        return "retry"
    return "next"


def _short(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    text = " ".join(str(exc).split())[:140]
    return f"{code} {text}" if isinstance(code, int) else f"{type(exc).__name__}: {text}"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


# Indirection so tests can fake time.
_clock = time.monotonic
_sleep = time.sleep


def _image_part(path: Path):
    from google.genai import types

    data, mime = _prepare_image(path)
    return types.Part.from_bytes(
        data=data, mime_type=mime,
        media_resolution=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
    )


def extract_drawing(
    image_path: str | Path,
    *,
    original_path: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    client=None,
    second_read: bool | None = None,
) -> Drawing:
    """Read the drawing with Gemini and return an unvalidated Drawing.

    second_read: read the photo a second time (original only) and mark every
      value the two reads disagree on as CHECK - see `cross_check`. Default
      from GLASS_LPO_SECOND_READ (off). Doubles the API calls.

    image_path: the enhanced image (or the only image).
    original_path: the original photo; if given, both images go in ONE call
      (image 1 = original, image 2 = enhanced) so enhancement artefacts can be
      caught.

    Fallback, kept deliberately simple:
      for each model (GEMINI_MODEL, comma-separated, first = preferred):
        up to 2 attempts
      - 400/401/403 (bad key / bad request): raise at once, no other models
      - 404/429 (no such model / quota) or unparseable JSON: next model
      - 5xx / timeout / connection error: retry once, then next model
      - safety block / MAX_TOKENS: raise a clear error
      All within GEMINI_DEADLINE_S (default 150 s); each call is capped at
      GEMINI_CALL_TIMEOUT_S (default 75 s).
    Run `validate.validate()` on the result to set flags and confidence.
    """
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:  # pragma: no cover
        pass

    models = [m.strip() for m in (model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).split(",") if m.strip()]
    deadline_s = _env_float("GEMINI_DEADLINE_S", DEFAULT_DEADLINE_S)
    call_timeout_s = _env_float("GEMINI_CALL_TIMEOUT_S", DEFAULT_CALL_TIMEOUT_S)

    enhanced = Path(image_path)
    original = Path(original_path) if original_path else None
    for p in (enhanced, original):
        if p is not None and not p.is_file():
            raise ExtractionError(f"Image not found: {p}")

    if client is None:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ExtractionError(
                "No Gemini API key: set GEMINI_API_KEY in the environment or a .env file "
                "(get one at https://aistudio.google.com/apikey)."
            )
        from google import genai

        client = genai.Client(api_key=key)

    if original is not None:
        contents = ["Image 1 - original photo:", _image_part(original),
                    "Image 2 - enhanced copy:", _image_part(enhanced),
                    TWO_IMAGES_NOTE + "\n" + PROMPT]
    else:
        contents = [_image_part(enhanced), PROMPT]

    drawing = _read_with_fallback(client, models, contents, deadline_s, call_timeout_s)

    if second_read is None:
        second_read = (os.getenv("GLASS_LPO_SECOND_READ") or "").strip().lower() in ("1", "true", "yes", "on")
    if second_read:
        # A second, independent read of the ORIGINAL photo only (a different
        # input, so it does not just repeat the first answer).
        print("[extract] second read (original photo only) to cross-check", flush=True)
        try:
            other = _read_with_fallback(client, models, [_image_part(original or enhanced), PROMPT],
                                        deadline_s, call_timeout_s)
        except ExtractionError as exc:
            note = "second read failed - values not cross-checked"
            print(f"[extract] {note}: {str(exc).splitlines()[0]}", flush=True)
            drawing.general_notes = f"{drawing.general_notes}; {note}" if drawing.general_notes else note
        else:
            drawing = cross_check(drawing, other)
    return drawing


def _read_with_fallback(client, models: list[str], contents: list,
                        deadline_s: float, call_timeout_s: float) -> Drawing:
    """One read of the drawing, with the retry / model fallback policy."""
    start = _clock()
    errors: list[str] = []
    for m in models:
        for attempt in (1, 2):
            left = deadline_s - (_clock() - start)
            if left < MIN_CALL_TIMEOUT_S:  # Gemini rejects deadlines under 10 s (400)
                print(f"[extract] deadline of {deadline_s:.0f}s reached, giving up", flush=True)
                raise ExtractionError(
                    f"Gemini extraction gave up after {deadline_s:.0f}s:\n  " + "\n  ".join(errors or ["(no attempts)"])
                )
            timeout_s = min(call_timeout_s, left)
            print(f"[extract] {m}: attempt {attempt}/2 (timeout {timeout_s:.0f}s)", flush=True)
            t0 = _clock()
            try:
                wire = _generate(client, m, contents, timeout_s)
                drawing = wire_to_drawing(wire)
            except ExtractionError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                kind = _classify(exc)
                errors.append(f"{m}: {_short(exc)}")
                if kind == "fatal":
                    print(f"[extract] {m}: request rejected ({_short(exc)}), stopping", flush=True)
                    raise ExtractionError(
                        f"Gemini rejected the request ({_short(exc)}). Check GEMINI_API_KEY "
                        "and the model name; other models were not tried."
                    ) from exc
                if kind == "retry" and attempt == 1:
                    print(f"[extract] {m}: {_short(exc)[:60]} - retrying once", flush=True)
                    _sleep(3)
                    continue
                print(f"[extract] {m}: {_short(exc)[:60]} - trying next model", flush=True)
                break
            drawing.model_used = m
            print(f"[extract] {m}: read OK in {_clock() - t0:.0f}s", flush=True)
            return drawing
    raise ExtractionError("Gemini extraction failed on all models:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------
# Second-read cross-check
# --------------------------------------------------------------------------
def _same_num(x, y) -> bool:
    x, y = float(x), float(y)
    return abs(x - y) <= max(1.0, 0.005 * max(abs(x), abs(y)))


def _same_glass(x, y) -> bool:
    from .glass_types import canonical_glass_type

    cx, cy = canonical_glass_type(str(x)), canonical_glass_type(str(y))
    if cx is not None and cy is not None:
        return cx == cy
    return str(x).strip().lower() == str(y).strip().lower()


def _pair_panels(a: list[Panel], b: list[Panel]) -> list[tuple[Panel, Panel]]:
    """Pair panels of two reads: same ID, then same size, then (only when both
    reads found the same number of panels) by position."""
    pairs: list[tuple[Panel, Panel]] = []
    left = list(range(len(b)))
    unpaired: list[int] = []
    for i, p in enumerate(a):
        k = next((k for k in left if b[k].panel_id.strip().upper() == p.panel_id.strip().upper()), None)
        if k is None:
            k = next((k for k in left
                      if None not in (p.width_mm.value, p.height_mm.value, b[k].width_mm.value, b[k].height_mm.value)
                      and _same_num(p.width_mm.value, b[k].width_mm.value)
                      and _same_num(p.height_mm.value, b[k].height_mm.value)), None)
        if k is None:
            unpaired.append(i)
            continue
        pairs.append((p, b[k]))
        left.remove(k)
    if len(a) == len(b):
        for i, k in zip(unpaired, left):
            pairs.append((a[i], b[k]))
    return pairs


def cross_check(first: Drawing, second: Drawing) -> Drawing:
    """Keep the first read; wherever BOTH reads have a value and they differ,
    make the value CHECK (clear=False) and record why. A value only one read
    found is not a disagreement (the other read may simply have missed it)."""
    d = first.model_copy(deep=True)

    def check(label: str, fa: Field, fb: Field, issues: list[str], same=_same_num) -> None:
        if fa.value is None or fb.value is None or same(fa.value, fb.value):
            return
        fa.clear = False
        issues.append(f"second read disagreed on {label}: {_fmt_any(fa.value)} vs {_fmt_any(fb.value)}")

    check("overall width", d.overall_width_mm, second.overall_width_mm, d.extraction_issues)
    check("overall height", d.overall_height_mm, second.overall_height_mm, d.extraction_issues)
    if len(d.panels) != len(second.panels):
        d.extraction_issues.append(
            f"second read found {len(second.panels)} panels, first read {len(d.panels)} - check the panel list")

    for pa, pb in _pair_panels(d.panels, second.panels):
        iss = pa.extraction_issues
        check("width", pa.width_mm, pb.width_mm, iss)
        check("height", pa.height_mm, pb.height_mm, iss)
        if pa.quantity.shown and pb.quantity.shown:
            check("qty", pa.quantity, pb.quantity, iss)
        check("thickness", pa.thickness_mm, pb.thickness_mm, iss)
        check("glass type", pa.glass_type, pb.glass_type, iss, same=_same_glass)
        if len(pa.holes) != len(pb.holes):
            iss.append(f"second read found {len(pb.holes)} holes, first read {len(pa.holes)}")
        else:
            da = sorted(h.diameter_mm.value for h in pa.holes if h.diameter_mm.value is not None)
            db = sorted(h.diameter_mm.value for h in pb.holes if h.diameter_mm.value is not None)
            if len(da) == len(db) and not all(_same_num(x, y) for x, y in zip(da, db)):
                for h in pa.holes:
                    h.diameter_mm.clear = False
                iss.append(f"second read disagreed on hole diameters: {da} vs {db}")
    return d


def _fmt_any(v) -> str:
    return _fmt_num(v) if isinstance(v, (int, float)) else str(v)
