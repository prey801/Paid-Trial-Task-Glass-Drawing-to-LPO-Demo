"""Excel export: one row per panel, columns exactly `EXCEL_COLUMNS` (ERP upload).

Clear numeric values are written as real numbers so the ERP can import them.
Anything not read clearly is written as "CHECK" or "<value> (CHECK)" - we never
guess silently. Styling is deliberately light (no merged cells, no totals rows)
so the main sheet stays machine-uploadable.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .schema import CHECK, EXCEL_COLUMNS, LPO, Feature, Field, Panel

# ---------------------------------------------------------------- formatting
# These helpers are shared with export_pdf (and validate's messages) so every
# output formats numbers the same way.


def fmt_num(v: Any) -> str:
    """1200.0 -> '1200', 6.5 -> '6.5', 0.30000000000000004 -> '0.3'. Text passes through."""
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def num_value(field: Field, required: bool = False) -> Any:
    """Real number when clear; '' when not written (CHECK if required);
    'CHECK' / 'x (CHECK)' otherwise."""
    if field.clear and field.value is not None:
        v = field.value
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v
    return field.display(fmt_num, required=required)


def text_value(field: Field, required: bool = False) -> str:
    return field.display(fmt_num, required=required)


def qty_value(panel: Panel) -> Any:
    """Qty not written -> 1 (assumed, flagged softly); unreadable -> CHECK."""
    q = panel.quantity
    if not q.shown and (q.value is None or q.value == 1):
        return 1
    return num_value(q)


def hole_diameters(panel: Panel) -> str:
    return "; ".join(h.diameter_mm.display(fmt_num, required=True) for h in panel.holes)


def hole_positions(panel: Panel) -> str:
    parts = [h.position.display() for h in panel.holes]
    if not any(parts):
        return ""
    return "; ".join(p or "-" for p in parts)


def feature_text(f: Feature) -> str:
    bits = [f.kind]
    size = f.size.display()
    pos = f.position.display()
    if size:
        bits.append(size)
    if pos:
        bits.append(f"@ {pos}")
    if f.note:
        bits.append(f"({f.note})")
    return " ".join(bits)


def features_text(features: list[Feature]) -> str:
    return "; ".join(feature_text(f) for f in features)


def area_value(panel: Panel) -> Any:
    """Per-line total m² (qty x W x H, 3 dp). Blank only if no size at all;
    CHECK whenever the area rests on any value not clearly read."""
    a = panel.area_m2
    if a is None:
        if panel.width_mm.value is None and panel.height_mm.value is None \
                and not panel.width_mm.shown and not panel.height_mm.shown:
            return ""
        return CHECK
    if panel.area_uncertain:
        return f"{a:.3f} ({CHECK})"
    return a


def area_text(panel: Panel) -> str:
    v = area_value(panel)
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def holes_uncertain(lpo: LPO) -> bool:
    """Total hole count can't be trusted if any hole or the qty of a holed panel is unclear."""
    for p in lpo.drawing.panels:
        if not p.holes:
            continue
        q = p.quantity
        if q.shown and (q.value is None or not q.clear):
            return True
        if any(h.diameter_mm.value is None or not h.diameter_mm.clear for h in p.holes):
            return True
        if p.extraction_issues:
            return True
    return False


def pieces_uncertain(lpo: LPO) -> bool:
    return any(p.quantity.shown and (p.quantity.value is None or not p.quantity.clear)
               for p in lpo.drawing.panels)


def panel_row(lpo: LPO, line: int, p: Panel) -> list[Any]:
    """Values for one Excel row, in EXCEL_COLUMNS order."""
    row = {
        "LPO No": lpo.lpo_no,
        "Line": line,
        "Panel ID": p.panel_id,
        "Width (mm)": num_value(p.width_mm, required=True),
        "Height (mm)": num_value(p.height_mm, required=True),
        "Qty": qty_value(p),
        "Thickness (mm)": num_value(p.thickness_mm),
        "Glass type": text_value(p.glass_type),
        "Area (m²)": area_value(p),
        "Hole count": len(p.holes),
        "Hole diameters (mm)": hole_diameters(p),
        "Hole positions": hole_positions(p),
        "Notches/offcuts": features_text(p.notches_offcuts),
        "Grooves/edge work": features_text(p.grooves_edgework),
        "Notes": p.notes,
        "Confidence": p.confidence.value,
        "Needs check (Y/N)": "Y" if p.needs_check else "N",
    }
    assert list(row) == EXCEL_COLUMNS, "panel_row out of sync with EXCEL_COLUMNS"
    return list(row.values())


# Panel flags that repeat across panels are listed ONCE at drawing level in the
# review list. (pattern on the panel flag, drawing-level text or None if the
# drawing already carries its own flag, minimum panels to hoist.)
_HOIST_RULES: list[tuple[re.Pattern, str | None, int]] = [
    (re.compile(r"^qty not written"), "Qty not written - assumed 1 per panel", 1),
    (re.compile(r"^thickness not stated$"), "Thickness not stated - confirm with client", 2),
    (re.compile(r"^hole positions not dimensioned"),
     "Hole positions not dimensioned - confirm with client", 2),
    (re.compile(r"^units .* not converted"), None, 1),
    (re.compile(r"^units: "), None, 1),
    (re.compile(r"^overall size not verified$"), None, 2),
    (re.compile(r"^layout not verified"), None, 2),
    (re.compile(r"^panel sizes don't add up to overall"), None, 2),
]


def _hoist_index(flag: str) -> int | None:
    for k, (pat, _, _) in enumerate(_HOIST_RULES):
        if pat.search(flag):
            return k
    return None


def review_items(lpo: LPO) -> list[tuple[str, str]]:
    """Every item a human must check, as (scope, text), de-duplicated for display.

    Panel flags stay intact on the panels (so needs_check is always right); only
    the rendering merges a flag shared by several panels into one drawing-level
    line naming those panels. Also lists flagged-but-silent panels (Medium/Low
    confidence without an explicit flag) so nothing that needs a check is missed."""
    d = lpo.drawing
    out = [("Drawing", f) for f in d.flags]
    hits: dict[int, list[str]] = {}
    for p in d.panels:
        for f in p.flags:
            k = _hoist_index(f)
            if k is not None and p.panel_id not in hits.setdefault(k, []):
                hits[k].append(p.panel_id)
    hoisted = {k for k, ids in hits.items() if len(ids) >= _HOIST_RULES[k][2]}
    for k in sorted(hoisted):
        text = _HOIST_RULES[k][1]
        if text:
            ids = hits[k]
            scope = "all panels" if len(ids) == len(d.panels) else ", ".join(ids)
            out.append(("Drawing", f"{text} ({scope})"))
    for p in d.panels:
        for f in p.flags:
            if _hoist_index(f) not in hoisted:
                out.append((p.panel_id, f))
        if p.needs_check and not p.flags:
            out.append((p.panel_id, f"{p.confidence.value} confidence reading - verify all values"))
    return out


def info_items(lpo: LPO) -> list[tuple[str, str]]:
    """Context for the reviewer that is not a flag: drawing notes, units, model."""
    d = lpo.drawing
    out = []
    if d.general_notes.strip():
        out.append(("Drawing note", d.general_notes.strip()))
    units = (d.units_written or "").strip()
    if units and units.lower() != "mm":
        out.append(("Units", f"Written in {units} - values shown as written, NOT converted to mm"))
    if d.model_used:
        out.append(("Read by", d.model_used))
    return out


READ_ME = [
    "Hole count = holes in ONE piece. Total holes for a line = Hole count x Qty.",
    "Area (m²) = line total = Qty x Width x Height.",
    "CHECK = not clearly read from the drawing. Never guessed - confirm on the drawing / with the client.",
    "Rows with Needs check = Y must be resolved before ERP import.",
]


# ---------------------------------------------------------------- workbook

HEADER_FILL = PatternFill("solid", start_color="1F3A5F")
HEADER_FONT = Font(bold=True, color="FFFFFF")
CHECK_FILL = PatternFill("solid", start_color="FFD966")   # amber: this cell
ROW_FILL = PatternFill("solid", start_color="FFF5D6")     # pale amber: this row
CHECK_FONT = Font(bold=True, color="9C5700")
FLAG_FONT = Font(bold=True, color="C00000")
THIN = Side(style="thin", color="D0D0D0")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

WIDTHS = {
    "LPO No": 17, "Line": 6, "Panel ID": 10, "Width (mm)": 12, "Height (mm)": 12,
    "Qty": 6, "Thickness (mm)": 10, "Glass type": 18, "Area (m²)": 10,
    "Hole count": 7, "Hole diameters (mm)": 16, "Hole positions": 30,
    "Notches/offcuts": 30, "Grooves/edge work": 26, "Notes": 30,
    "Confidence": 11, "Needs check (Y/N)": 10,
}


def write_excel(lpo: LPO, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "LPO Lines"
    ws.append(EXCEL_COLUMNS)
    for c in ws[1]:
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER
    ws.row_dimensions[1].height = 32
    for name, note in (
        ("Hole count", "Holes in ONE piece. Line total = Hole count x Qty."),
        ("Area (m²)", "Line total: Qty x Width x Height, m²."),
        ("Needs check (Y/N)", "Y = resolve before ERP import (see Flags sheet)."),
    ):
        ws.cell(row=1, column=EXCEL_COLUMNS.index(name) + 1).comment = Comment(note, "glass_lpo")

    for i, p in enumerate(lpo.drawing.panels, start=1):
        ws.append(panel_row(lpo, i, p))
        flagged = p.needs_check
        for c in ws[ws.max_row]:
            c.border = BORDER
            c.alignment = Alignment(vertical="top", wrap_text=isinstance(c.value, str))
            if isinstance(c.value, str) and CHECK in c.value:
                c.fill = CHECK_FILL
                c.font = CHECK_FONT
            elif flagged:
                c.fill = ROW_FILL
        needs = ws.cell(row=ws.max_row, column=len(EXCEL_COLUMNS))
        needs.alignment = Alignment(horizontal="center", vertical="top")
        if flagged:
            needs.font = FLAG_FONT
        area = ws.cell(row=ws.max_row, column=EXCEL_COLUMNS.index("Area (m²)") + 1)
        if isinstance(area.value, float):
            area.number_format = "0.000"

    for idx, name in enumerate(EXCEL_COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = WIDTHS.get(name, 14)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(EXCEL_COLUMNS))}{max(ws.max_row, 1)}"

    # ---- Flags sheet (human review list; not for ERP upload)
    fs = wb.create_sheet("Flags")
    fs.append(["#", "LPO No", "Scope", "Flag"])
    for c in fs[1]:
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.border = BORDER
    items = review_items(lpo)
    for n, (scope, text) in enumerate(items, start=1):
        fs.append([n, lpo.lpo_no, scope, text])
    if not items:
        fs.append(["", lpo.lpo_no, "", "No items flagged"])
    info_start = fs.max_row + 2
    fs.append([])
    for scope, text in info_items(lpo) + [("Read me", t) for t in READ_ME]:
        fs.append(["", lpo.lpo_no, scope, text])
    for row in fs.iter_rows(min_row=2):
        for c in row:
            if c.row == info_start - 1:
                continue
            c.border = BORDER
            c.alignment = Alignment(vertical="top", wrap_text=True)
            if c.row >= info_start:
                c.font = Font(italic=True, color="5A6270")
    for col, w in zip("ABCD", (5, 17, 12, 90)):
        fs.column_dimensions[col].width = w
    fs.freeze_panes = "A2"

    wb.save(path)
    return path
