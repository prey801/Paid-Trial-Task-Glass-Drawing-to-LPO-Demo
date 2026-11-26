"""Rule checks on an extracted Drawing: adds short human-readable flags and
sets each panel's confidence.

Confidence:
  High   - no flags
  Medium - only soft warnings (near limits, qty assumed, layout not verifiable, ...)
  Low    - any CHECK value or a hard rule failure

Every rule that fires on a panel is kept in that panel's `flags`, so
`Panel.needs_check` always agrees with the confidence. Messages that repeat on
many panels (e.g. "qty not written") are merged into one drawing-level line at
render time by `export_excel.review_items()` - not here.

`validate()` is idempotent: it starts from a copy with flags cleared.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Optional

from .export_excel import fmt_num
from .glass_types import canonical_glass_type, is_generic_glass_word
from .schema import Confidence, Drawing, Field, Panel

MIN_DIM_MM = 50
MAX_DIM_MM = 6000
SHEET_SHORT_MM = 3210  # jumbo sheet 3210 x 6000
STANDARD_THICKNESS_MM = {3, 4, 5, 6, 8, 10, 12, 15, 19, 25}
HOLE_MIN_MM = 4
HOLE_MAX_MM = 100
QTY_MIN, QTY_MAX = 1, 500
MAX_GRID_PANELS = 12  # subset-sum layout check is skipped above this

HARD, SOFT = "hard", "soft"

# Panel flag texts that validate emits and export_excel.review_items() hoists.
QTY_NOT_WRITTEN = "qty not written - assumed 1"
THICKNESS_NOT_STATED = "thickness not stated"
HOLE_POS_NOT_DIMENSIONED = "hole positions not dimensioned"
OVERALL_NOT_VERIFIED = "overall size not verified"
LAYOUT_NOT_VERIFIED = "layout not verified"
LAYOUT_TOO_MANY = "layout not verified (too many panels)"
SIZES_DONT_ADD_UP = "panel sizes don't add up to overall (CHECK)"


def _n(v) -> str:
    """Number for messages: same formatting as the exports ('?' for None)."""
    if v is None:
        return "?"
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return v
    return fmt_num(v)


def tolerance(overall: float) -> float:
    return max(5.0, 0.01 * abs(overall))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= tolerance(b)


def _not_written(f: Field) -> bool:
    return not f.shown and f.value is None


class _Collector:
    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def hard(self, msg: str) -> None:
        self.items.append((HARD, msg))

    def soft(self, msg: str) -> None:
        self.items.append((SOFT, msg))


def _check_value(c: _Collector, label: str, f: Field, *, required: bool = False) -> bool:
    """CHECK flag for missing/unclear values. Returns True if value is usable."""
    if _not_written(f):
        if required:
            c.hard(f"{label} not written (CHECK)")
        return False
    if f.value is None:
        c.hard(f"{label} unreadable (CHECK)")
        return False
    if not f.clear:
        c.hard(f"{label} {_n(f.value)} unclear (CHECK)")
    return True


_POS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:from|fr\.?|off|to)\s+(?:the\s+)?(left|right|top|bottom)",
    re.IGNORECASE,
)
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*[x×X*]\s*(\d+(?:\.\d+)?)")
_DIGIT_RE = re.compile(r"\d")


def parse_offsets(text: Optional[str]) -> list[tuple[float, str]]:
    """'50 from left, 100 from top' -> [(50.0, 'left'), (100.0, 'top')]."""
    if not text:
        return []
    return [(float(m.group(1)), m.group(2).lower()) for m in _POS_RE.finditer(text)]


def parse_size(text: Optional[str]) -> Optional[tuple[float, float]]:
    """'150 x 100mm' -> (150.0, 100.0); None if not a W x H size."""
    m = _SIZE_RE.search(text or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def glass_type_recognised(text: str) -> bool:
    return canonical_glass_type(text) is not None


def _normalise_glass_type(p: Panel) -> None:
    """Map the written glass type onto the fixed vocabulary (glass_types.py).
    'glass' on its own (a glazed area, or a title) is not a type: it is
    dropped with a note. Unrecognised text is kept and flagged later."""
    gt = p.glass_type
    if gt.value is None or not str(gt.value).strip():
        return
    written = str(gt.value).strip()
    canon = canonical_glass_type(written)
    if canon and canon != written:
        p.glass_type = Field(value=canon, clear=gt.clear, shown=gt.shown)
        note = f"glass type written '{written}' = {canon}"
    elif canon is None and gt.clear and is_generic_glass_word(written):
        p.glass_type = Field(value=None, clear=False, shown=False)
        note = f"'{written}' is not a glass type - ignored"
    else:
        return
    if note not in p.notes:
        p.notes = f"{p.notes}; {note}" if p.notes else note


def _check_position(c: _Collector, label: str, text: Optional[str], w: Optional[float], h: Optional[float]) -> None:
    for off, edge in parse_offsets(text):
        limit = w if edge in ("left", "right") else h
        if limit is None:
            continue
        if off <= 0 or off >= limit:
            dim = "width" if edge in ("left", "right") else "height"
            c.hard(f"{label} {_n(off)} from {edge} outside panel {dim} {_n(limit)}")


def _check_feature_field(c: _Collector, kind: str, attr: str, fld: Field) -> None:
    if _not_written(fld):
        return
    if fld.value is None:
        c.hard(f"{kind} {attr} unreadable (CHECK)")
    elif not fld.clear:
        c.hard(f"{kind} {attr} '{fld.value}' unclear (CHECK)")


def _check_panel(p: Panel, skip: set[str] = frozenset()) -> _Collector:
    """`skip`: dimension names ("width"/"height") filled from the overall size;
    they carry their own soft flag and are not re-checked as unclear."""
    c = _Collector()

    for issue in p.extraction_issues:
        if issue.strip():
            c.hard(f"{issue.strip()} (CHECK)" if "CHECK" not in issue else issue.strip())

    def dim(label: str, f: Field) -> Optional[float]:
        if label in skip:
            return f.value
        return f.value if _check_value(c, label, f, required=True) else None

    w = dim("width", p.width_mm)
    h = dim("height", p.height_mm)

    for label, v in (("width", w), ("height", h)):
        if v is None:
            continue
        if v < MIN_DIM_MM or v > MAX_DIM_MM:
            c.hard(f"{label} {_n(v)} mm outside {MIN_DIM_MM}-{MAX_DIM_MM}")
    if w is not None and h is not None and MIN_DIM_MM <= w <= MAX_DIM_MM and MIN_DIM_MM <= h <= MAX_DIM_MM:
        if min(w, h) > SHEET_SHORT_MM:
            c.hard(f"{_n(w)}x{_n(h)} exceeds jumbo sheet {SHEET_SHORT_MM}x{MAX_DIM_MM}")
        elif max(w, h) > SHEET_SHORT_MM:
            label = "width" if w > SHEET_SHORT_MM else "height"
            c.soft(f"{label} {_n(max(w, h))} mm > {SHEET_SHORT_MM} (near sheet limit)")

    # Quantity: not written -> assumed 1 (soft, never silent); unreadable -> hard.
    q = p.quantity
    if not q.shown:
        c.soft(QTY_NOT_WRITTEN)
        if q.value is not None and q.value != 1:
            c.hard(f"qty {_n(q.value)} not on drawing (CHECK)")
    elif _check_value(c, "qty", q):
        if q.value < QTY_MIN or q.value > QTY_MAX:
            c.hard(f"qty {_n(q.value)} outside {QTY_MIN}-{QTY_MAX}")

    t = None
    if _not_written(p.thickness_mm):
        c.soft(THICKNESS_NOT_STATED)
    elif _check_value(c, "thickness", p.thickness_mm):
        t = p.thickness_mm.value
        if float(t) not in STANDARD_THICKNESS_MM:
            c.hard(f"thickness {_n(t)} mm not standard")

    _normalise_glass_type(p)
    gt = p.glass_type
    if gt.shown or gt.value is not None:
        if gt.value is None or not str(gt.value).strip():
            c.hard("glass type unreadable (CHECK)")
        elif not gt.clear:
            c.hard(f"glass type '{gt.value}' unclear (CHECK)")
        elif not glass_type_recognised(str(gt.value)):
            c.soft(f"glass type '{gt.value}' not recognised - confirm")

    undimensioned: list[int] = []
    for i, hole in enumerate(p.holes, start=1):
        lbl = f"hole {i}"
        if _check_value(c, f"{lbl} dia", hole.diameter_mm, required=True):
            d = hole.diameter_mm.value
            if d < HOLE_MIN_MM or d > HOLE_MAX_MM:
                c.hard(f"{lbl} Ø{_n(d)} outside {HOLE_MIN_MM}-{HOLE_MAX_MM} mm")
            if t is not None and d < t:
                c.hard(f"{lbl} Ø{_n(d)} < glass {_n(t)} mm (toughening min)")
        pos = hole.position
        if _not_written(pos):
            undimensioned.append(i)
        elif pos.value is None:
            c.hard(f"{lbl} position unreadable (CHECK)")
        elif not _DIGIT_RE.search(str(pos.value)):
            # "top left", "centre": a location but no dimension to drill to.
            undimensioned.append(i)
        else:
            if not pos.clear:
                c.hard(f"{lbl} position '{pos.value}' unclear (CHECK)")
            _check_position(c, lbl, pos.value, w, h)
    if undimensioned:
        if len(undimensioned) == len(p.holes):
            c.soft(HOLE_POS_NOT_DIMENSIONED)
        else:
            nums = ", ".join(str(i) for i in undimensioned)
            c.soft(f"{HOLE_POS_NOT_DIMENSIONED} (hole {nums})")

    for f in p.notches_offcuts:  # notch / offcut / cutout: a size is needed to cut it
        kind = f.kind or "notch"
        if f.size.value is None or not str(f.size.value).strip():
            c.hard(f"{kind} size {'unreadable' if f.size.shown else 'not given'} (CHECK)")
        elif not f.size.clear:
            c.hard(f"{kind} size '{f.size.value}' unclear (CHECK)")
        _check_feature_field(c, kind, "position", f.position)
        if f.size.value and w is not None and h is not None:
            sz = parse_size(str(f.size.value))
            if sz:
                a, b = sz
                if not ((a < w and b < h) or (b < w and a < h)):
                    c.hard(f"{kind} {_n(a)}x{_n(b)} larger than panel {_n(w)}x{_n(h)} (CHECK)")
        if f.position.value:
            _check_position(c, kind, f.position.value, w, h)

    for f in p.grooves_edgework:
        kind = f.kind or "edge work"
        _check_feature_field(c, kind, "size", f.size)
        _check_feature_field(c, kind, "position", f.position)
        if f.position.value:
            _check_position(c, kind, f.position.value, w, h)

    return c


def _subset_sums(values: list[float]) -> set[float]:
    sums: set[float] = set()
    for r in range(1, len(values) + 1):
        for combo in combinations(values, r):
            sums.add(round(sum(combo), 3))
    return sums


def _check_layout(d: Drawing, panel_collectors: dict[int, _Collector]) -> list[str]:
    """Compare panel sizes with the overall dimensions. Returns drawing flags."""
    ow = d.overall_width_mm.value
    oh = d.overall_height_mm.value
    if (ow is None and oh is None) or not d.panels:
        return []

    ws = [p.width_mm.value for p in d.panels]
    hs = [p.height_mm.value for p in d.panels]
    if any(v is None for v in ws + hs):
        for c in panel_collectors.values():
            c.soft(OVERALL_NOT_VERIFIED)
        return ["Overall size not verified: some panel sizes missing"]

    qs = [p.quantity.value if (p.quantity.value or 0) > 0 else 1 for p in d.panels]
    sum_w = sum(ws)
    sum_h = sum(hs)
    sum_wq = sum(w * q for w, q in zip(ws, qs))
    sum_hq = sum(h * q for h, q in zip(hs, qs))

    def w_ok(total: float) -> bool:
        return ow is None or _close(total, ow)

    def h_ok(total: float) -> bool:
        return oh is None or _close(total, oh)

    def all_w_fit() -> bool:  # stacked: every panel spans the overall width
        return ow is None or all(_close(w, ow) for w in ws)

    def all_h_fit() -> bool:  # side by side: every panel spans the overall height
        return oh is None or all(_close(h, oh) for h in hs)

    side_by_side = (w_ok(sum_w) or w_ok(sum_wq)) and all_h_fit()
    stacked = (h_ok(sum_h) or h_ok(sum_hq)) and all_w_fit()
    if side_by_side or stacked:
        return []

    too_big = (ow is not None and any(w > ow + tolerance(ow) for w in ws)) or (
        oh is not None and any(h > oh + tolerance(oh) for h in hs)
    )

    # Grid / mixed layouts: some combination of panels spans each direction.
    if not too_big:
        if len(d.panels) > MAX_GRID_PANELS:
            for c in panel_collectors.values():
                c.soft(LAYOUT_TOO_MANY)
            return ["Layout not verified (too many panels to check against overall size)"]
        w_sub = ow is None or any(_close(s, ow) for s in _subset_sums(ws))
        h_sub = oh is None or any(_close(s, oh) for s in _subset_sums(hs))
        if w_sub and h_sub:
            for c in panel_collectors.values():
                c.soft(LAYOUT_NOT_VERIFIED)
            return ["Layout not side-by-side or stacked; overall size only partly verified"]

    parts = []
    if ow is not None:
        parts.append(f"widths sum {_n(sum_w)} vs overall {_n(ow)}")
    if oh is not None:
        parts.append(f"heights sum {_n(sum_h)} vs overall {_n(oh)}")
    for c in panel_collectors.values():
        c.hard(SIZES_DONT_ADD_UP)
    return ["Sizes don't match overall: " + "; ".join(parts)]


def _fill_from_overall(d: Drawing) -> dict[int, tuple[str, str]]:
    """Stacked / side-by-side drawings often write each panel's height (or width)
    but not the dimension shared with the overall size. A glass processor reads
    those panels as spanning the overall width (or height). We fill that value,
    never silently:
      * only values NOT WRITTEN at all (shown=False, value None) are filled -
        a written-but-unreadable value stays CHECK;
      * every panel whose shared dimension IS written must match the overall;
      * the filled value is clear=False (renders "900 (CHECK)") and the panel
        gets a soft flag saying where it came from.
    Returns {panel index: (dimension name, flag message)}."""
    filled: dict[int, tuple[str, str]] = {}
    ow = d.overall_width_mm.value if d.overall_width_mm.clear else None
    oh = d.overall_height_mm.value if d.overall_height_mm.clear else None
    ps = d.panels
    if not ps or ow is None or oh is None:
        return filled

    def try_fill(span_attr: str, chain_attr: str, overall_span: float, overall_chain: float,
                 name: str) -> bool:
        spans = [getattr(p, span_attr) for p in ps]
        # A not-written value (or one we derived on an earlier run - validate is
        # idempotent) is a fill target; any written value must match the overall.
        derived = [not f.shown and not f.clear for f in spans]
        chain = [getattr(p, chain_attr).value for p in ps]
        if all(v is not None for v in chain) and _close(sum(chain), overall_chain):
            if any(f.value is not None and not der and not _close(f.value, overall_span)
                   for f, der in zip(spans, derived)):
                return False
            targets = [i for i, der in enumerate(derived) if der]
        else:
            # Several items on one sheet (e.g. 2 windows + a door): the overall
            # size belongs to ONE item. Fill only the panels with the span not
            # written, and only if THEIR chain alone adds up to the overall.
            # A panel whose chain value is unknown (e.g. the door's glass, with
            # no size written) can't be shown to belong, so it is left out.
            targets = [i for i, der in enumerate(derived) if der and chain[i] is not None]
            if not targets or not _close(sum(chain[i] for i in targets), overall_chain):
                return False
        if not targets:
            return False
        for i in targets:
            setattr(ps[i], span_attr, Field(value=float(overall_span), clear=False, shown=False))
            filled[i] = (name, f"{name} {_n(overall_span)} taken from overall {name} (not written)")
        return True

    # Stacked (heights chain to overall height) first, then side by side.
    if not try_fill("width_mm", "height_mm", ow, oh, "width"):
        try_fill("height_mm", "width_mm", oh, ow, "height")
    return filled


def validate(drawing: Drawing) -> Drawing:
    """Return a copy of `drawing` with flags appended and confidence set."""
    d = drawing.model_copy(deep=True)
    d.flags = []
    for p in d.panels:
        p.flags = []

    from .units import convert_to_mm  # local import keeps module graph simple

    # Convert cm / m / no-unit (assumed cm) drawings to mm before any rule runs.
    convert_to_mm(d)

    for issue in d.extraction_issues:
        if issue.strip():
            d.flags.append(f"{issue.strip()} (CHECK)" if "CHECK" not in issue else issue.strip())

    units = (d.units_written or "").strip().lower()
    units_bad = units not in ("mm", "")
    if units_bad:
        d.flags.append(f"Units written in {d.units_written.strip()} - values NOT converted (CHECK)")
    conv = d.units_converted_from
    if conv:
        d.flags.append(f"Units: {conv} - confirm")

    if not d.panels:
        d.flags.append("No panels found - check drawing")
        return d

    filled = _fill_from_overall(d)
    collectors = {
        i: _check_panel(p, {filled[i][0]} if i in filled else set())
        for i, p in enumerate(d.panels)
    }
    for i, (_, msg) in filled.items():
        collectors[i].soft(msg)
    if units_bad:
        for c in collectors.values():
            c.hard(f"units {d.units_written.strip()} - not converted (CHECK)")
    if conv:
        for c in collectors.values():
            c.soft(f"units: {conv}")

    # Overall dimensions themselves.
    for label, f in (("Overall width", d.overall_width_mm), ("Overall height", d.overall_height_mm)):
        if f.shown and f.value is None:
            d.flags.append(f"{label} unreadable (CHECK)")
        elif f.value is not None and not f.clear:
            d.flags.append(f"{label} {_n(f.value)} unclear (CHECK)")

    # Duplicate IDs.
    seen: dict[str, list[int]] = {}
    for i, p in enumerate(d.panels):
        seen.setdefault(p.panel_id.strip().upper(), []).append(i)
    for pid, idxs in seen.items():
        if len(idxs) > 1:
            d.flags.append(f"Duplicate panel ID {d.panels[idxs[0]].panel_id} (x{len(idxs)})")
            for i in idxs:
                collectors[i].hard("duplicate panel ID")

    d.flags.extend(_check_layout(d, collectors))

    for i, p in enumerate(d.panels):
        items = collectors[i].items
        # de-duplicate while keeping order (e.g. several identical holes)
        seen_msgs: set[str] = set()
        for _, msg in items:
            if msg not in seen_msgs:
                seen_msgs.add(msg)
                p.flags.append(msg)
        if any(sev == HARD for sev, _ in items):
            p.confidence = Confidence.LOW
        elif items:
            p.confidence = Confidence.MEDIUM
        else:
            p.confidence = Confidence.HIGH
    return d
