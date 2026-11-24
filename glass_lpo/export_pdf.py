"""One-page A4 (landscape) LPO draft PDF.

Guaranteed single page: fonts shrink to fit, and if it still doesn't fit the
line-item table and the review list are truncated with "+N more, see Excel".
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

from .export_excel import (
    area_text, features_text, fmt_num, holes_uncertain, info_items, pieces_uncertain,
    qty_value, review_items, text_value,
)
from .schema import CHECK, LPO, Panel

PAGE_W, PAGE_H = landscape(A4)
MARGIN = 26
CONTENT_W = PAGE_W - 2 * MARGIN

NAVY = colors.HexColor("#1F3A5F")
GREY = colors.HexColor("#5A6270")
LINE = colors.HexColor("#C9CED6")
ZEBRA = colors.HexColor("#F4F6F9")
AMBER = colors.HexColor("#FFD966")
AMBER_TEXT = colors.HexColor("#8A4B00")
PALE_AMBER = colors.HexColor("#FFF5D6")
RED = colors.HexColor("#B42318")
RED_BG = colors.HexColor("#FEF3F2")
GREEN = colors.HexColor("#067647")
GREEN_BG = colors.HexColor("#ECFDF3")

HEADERS = ["Line", "Panel", "W (mm)", "H (mm)", "Qty", "Thk", "Type", "Area m²",
           "Holes (count × Ø)", "Notches / offcuts", "Edge work", "Conf.", "Check?"]
_REL_W = [24, 44, 54, 54, 28, 30, 80, 50, 92, 134, 116, 36, 44]
COL_W = [w * CONTENT_W / sum(_REL_W) for w in _REL_W]

HEADER_H = 66
FOOTER_H = 18
TOTALS_H = 22
MIN_BOTTOM = 120  # space kept for the review box below the table


# ---------------------------------------------------------------- cell text

def holes_text(p: Panel) -> str:
    if not p.holes:
        return ""
    groups = Counter(h.diameter_mm.display(fmt_num, required=True) for h in p.holes)
    parts = [f"{n} × Ø{d}" for d, n in groups.items()]
    return ", ".join(parts)


def panel_cells(line: int, p: Panel) -> list[str]:
    return [
        str(line), p.panel_id,
        text_value(p.width_mm, required=True), text_value(p.height_mm, required=True),
        fmt_num(qty_value(p)),
        text_value(p.thickness_mm), text_value(p.glass_type), area_text(p),
        holes_text(p), features_text(p.notches_offcuts), features_text(p.grooves_edgework),
        p.confidence.value, "YES" if p.needs_check else "",
    ]


def detail_text(p: Panel) -> str:
    """Sub-line for the cutter: hole positions and notes (incl. raw-text alternatives)."""
    bits = []
    pos = [h.position.display() for h in p.holes]
    if any(pos):
        bits.append("Hole positions: " + "; ".join(
            f"{i}) {t or 'not given'}" for i, t in enumerate(pos, start=1)))
    if p.notes.strip():
        bits.append("Notes: " + p.notes.strip())
    return "   ·   ".join(bits)


# ---------------------------------------------------------------- table

def _styles(fs: float):
    base = ParagraphStyle("cell", fontName="Helvetica", fontSize=fs, leading=fs * 1.18)
    return {
        "cell": base,
        "num": ParagraphStyle("num", parent=base, alignment=2),
        "ctr": ParagraphStyle("ctr", parent=base, alignment=1),
        "head": ParagraphStyle("head", parent=base, fontName="Helvetica-Bold",
                               textColor=colors.white),
        "check": ParagraphStyle("check", parent=base, fontName="Helvetica-Bold",
                                textColor=AMBER_TEXT),
        "flag": ParagraphStyle("flag", parent=base, fontName="Helvetica-Bold",
                               textColor=RED, alignment=1),
        "detail": ParagraphStyle("detail", parent=base, fontName="Helvetica-Oblique",
                                 fontSize=fs - 0.5, leading=(fs - 0.5) * 1.18,
                                 textColor=GREY),
        "detail_check": ParagraphStyle("detail_check", parent=base, fontName="Helvetica-BoldOblique",
                                       fontSize=fs - 0.5, leading=(fs - 0.5) * 1.18,
                                       textColor=AMBER_TEXT),
    }


NUM_COLS = {2, 3, 4, 5, 7}
CTR_COLS = {0, 11}


def _build_table(lpo: LPO, fs: float, max_rows: int | None):
    st = _styles(fs)
    panels = lpo.drawing.panels
    shown = panels if max_rows is None else panels[:max_rows]
    hidden = len(panels) - len(shown)

    data = [[Paragraph(escape(h), st["head"]) for h in HEADERS]]
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    for line, p in enumerate(shown, start=1):
        i = len(data)
        cells = panel_cells(line, p)
        row = []
        detail = detail_text(p)
        last = i + (1 if detail else 0)
        bg = PALE_AMBER if p.needs_check else (ZEBRA if line % 2 == 0 else colors.white)
        cmds.append(("BACKGROUND", (0, i), (-1, last), bg))
        for c, txt in enumerate(cells):
            if CHECK in txt:
                style = st["check"]
                cmds.append(("BACKGROUND", (c, i), (c, i), AMBER))
            elif c == 12:
                style = st["flag"]
            elif c in NUM_COLS:
                style = st["num"]
            elif c in CTR_COLS:
                style = st["ctr"]
            else:
                style = st["cell"]
            row.append(Paragraph(escape(txt), style))
        data.append(row)
        if detail:
            style = st["detail_check"] if CHECK in detail else st["detail"]
            data.append([Paragraph(escape(detail), style)] + [""] * 12)
            cmds += [("SPAN", (0, last), (-1, last)),
                     ("LEFTPADDING", (0, last), (-1, last), COL_W[0] + 3),
                     ("TOPPADDING", (0, last), (-1, last), 0),
                     ("LINEABOVE", (0, last), (-1, last), 0.8, bg)]
    if not panels:
        data.append([Paragraph("No panels extracted - see review box.", st["check"])] + [""] * 12)
        cmds.append(("SPAN", (0, 1), (-1, 1)))
    if hidden:
        r = len(data)
        data.append([Paragraph(f"+{hidden} more line{'s' if hidden > 1 else ''}, see Excel",
                               st["check"])] + [""] * 12)
        cmds += [("SPAN", (0, r), (-1, r)), ("BACKGROUND", (0, r), (-1, r), AMBER)]

    t = Table(data, colWidths=COL_W, repeatRows=0)
    t.setStyle(TableStyle(cmds))
    _, h = t.wrap(CONTENT_W, 10_000)
    return t, h


def _fit_table(lpo: LPO, max_h: float):
    """Largest font that fits; else smallest font with truncated rows."""
    for fs in (8, 7.5, 7, 6.5, 6, 5.5):
        t, h = _build_table(lpo, fs, None)
        if h <= max_h:
            return t, h
    n = len(lpo.drawing.panels)
    for rows in range(n - 1, -1, -1):
        t, h = _build_table(lpo, 5.5, rows)
        if h <= max_h:
            return t, h
    return _build_table(lpo, 5.5, 0)


# ---------------------------------------------------------------- review box

def _review_table(items: list[tuple[str, str]], width: float, fs: float, max_items: int | None,
                  info: list[tuple[str, str]] = ()):
    st = ParagraphStyle("r", fontName="Helvetica", fontSize=fs, leading=fs * 1.2)
    sti = ParagraphStyle("ri", parent=st, fontName="Helvetica-Oblique", textColor=GREY)
    shown = items if max_items is None else items[:max_items]
    hidden = len(items) - len(shown)
    paras = [Paragraph(f"<b>{escape(scope)}</b>: {escape(text)}", st) for scope, text in shown]
    if hidden:
        paras.append(Paragraph(f"<b>+{hidden} more, see Excel (Flags sheet)</b>", st))
    paras += [Paragraph(f"<b>{escape(scope)}</b>: {escape(text)}", sti) for scope, text in info]
    labels = ([f"{i + 1}." for i in range(len(shown))] + ([""] if hidden else [])
              + ["i"] * len(info))
    cells = list(zip(labels, paras))
    if len(cells) > 6:  # two columns, filled top-to-bottom then left-to-right
        half = (len(cells) + 1) // 2
        left, right = cells[:half], cells[half:] + [("", "")] * (half - len(cells[half:]))
        rows = [[a, b, c2, d] for (a, b), (c2, d) in zip(left, right)]
        cw = [16, width / 2 - 16, 16, width / 2 - 16]
    else:
        rows = [[a, b] for a, b in cells]
        cw = [16, width - 16]
    t = Table(rows, colWidths=cw)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONT", (0, 0), (-1, -1), "Helvetica-Bold", fs),
        ("TEXTCOLOR", (0, 0), (-1, -1), RED),
        ("LEFTPADDING", (0, 0), (-1, -1), 1), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    _, h = t.wrap(width, 10_000)
    return t, h


def _fit_review(items, width: float, max_h: float, info=()):
    info = list(info)
    for fs in (8.5, 8, 7.5, 7, 6.5, 6):
        t, h = _review_table(items, width, fs, None, info)
        if h <= max_h:
            return t, h
    # Too long: drop notes before flags (notes are also on the Excel Flags sheet),
    # then truncate flags with "+N more".
    if info:
        info = [(s, (x[:160] + "…") if len(x) > 160 else x) for s, x in info]
        t, h = _review_table(items, width, 6, None, info)
        if h <= max_h:
            return t, h
    for k in range(len(items) - 1, -1, -1):
        t, h = _review_table(items, width, 6, k, info)
        if h <= max_h:
            return t, h
    return _review_table(items, width, 6, 0)


# ---------------------------------------------------------------- page

def _header(c: canvas.Canvas, lpo: LPO):
    top = PAGE_H - MARGIN
    c.setFillColor(NAVY)
    c.rect(MARGIN, top - 4, CONTENT_W, 4, stroke=0, fill=1)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(MARGIN, top - 26, "LOCAL PURCHASE ORDER — DRAFT")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(GREY)
    c.drawString(MARGIN, top - 40, "Glass cutting order extracted from a hand-drawn drawing. "
                                   "Not valid for cutting until checked.")
    # right-hand meta block
    x_lbl, x_val = PAGE_W - MARGIN - 230, PAGE_W - MARGIN - 160
    src = lpo.source_image or "-"
    src = Path(src).name
    if len(src) > 38:
        src = src[:17] + "…" + src[-18:]
    model = lpo.drawing.model_used or "-"
    if len(model) > 38:
        model = model[:37] + "…"
    for k, (label, val) in enumerate([("LPO No", lpo.lpo_no), ("Date", lpo.date),
                                      ("Source image", src), ("Read by", model)]):
        y = top - 18 - k * 12
        c.setFont("Helvetica", 8)
        c.setFillColor(GREY)
        c.drawString(x_lbl, y, label)
        c.setFont("Helvetica-Bold", 9 if k == 0 else 8)
        c.setFillColor(colors.black)
        c.drawString(x_val, y, val)
    # overall size + units, for the cutter
    d = lpo.drawing
    ow = d.overall_width_mm.display(fmt_num) or "not written"
    oh = d.overall_height_mm.display(fmt_num) or "not written"
    units = (d.units_written or "").strip()
    unit_lbl = units or "mm"
    x = MARGIN
    y = top - 55
    for label, val, warn in (
        ("Overall size", f"{ow} × {oh} {unit_lbl}", CHECK in ow + oh),
        ("Units on drawing", unit_lbl if unit_lbl.lower() == "mm"
         else f"{unit_lbl} - NOT converted ({CHECK})", unit_lbl.lower() != "mm"),
    ):
        c.setFont("Helvetica", 8)
        c.setFillColor(GREY)
        c.drawString(x, y, label)
        x += c.stringWidth(label, "Helvetica", 8) + 5
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(AMBER_TEXT if warn else colors.black)
        c.drawString(x, y, val)
        x += c.stringWidth(val, "Helvetica-Bold", 9) + 24
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(MARGIN, top - HEADER_H + 4, PAGE_W - MARGIN, top - HEADER_H + 4)


def _footer(c: canvas.Canvas, lpo: LPO):
    y = MARGIN - 6
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(MARGIN, y + 12, PAGE_W - MARGIN, y + 12)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(RED)
    c.drawString(MARGIN, y + 2, "Pricing not included — generated draft, verify before cutting")
    c.setFont("Helvetica", 7.5)
    c.setFillColor(GREY)
    c.drawRightString(PAGE_W - MARGIN, y + 2, f"{lpo.lpo_no}  ·  Page 1 of 1")


def _totals(c: canvas.Canvas, lpo: LPO, y_top: float):
    panels = lpo.drawing.panels
    pieces = sum(p.quantity.value or 1 for p in panels)
    n_unv = lpo.uncertain_area_lines
    c.setFillColor(ZEBRA)
    c.setStrokeColor(LINE)
    c.rect(MARGIN, y_top - TOTALS_H + 4, CONTENT_W, TOTALS_H - 6, stroke=1, fill=1)
    y = y_top - TOTALS_H + 10
    x = MARGIN + 8
    area = f"{lpo.total_area_m2:.3f} m²"
    if n_unv:
        area += f" (includes {n_unv} unverified line{'s' if n_unv > 1 else ''} – {CHECK})"
    holes = str(lpo.total_holes) + (f" ({CHECK})" if holes_uncertain(lpo) else "")
    items = [("Lines", str(len(panels)), False),
             ("Pieces", str(pieces) + (f" ({CHECK})" if pieces_uncertain(lpo) else ""),
              pieces_uncertain(lpo)),
             ("Total area", area, bool(n_unv)),
             ("Total holes", holes, holes_uncertain(lpo))]
    for label, val, warn in items:
        c.setFont("Helvetica", 8.5)
        c.setFillColor(GREY)
        c.drawString(x, y, label)
        x += c.stringWidth(label, "Helvetica", 8.5) + 5
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(AMBER_TEXT if warn else colors.black)
        c.drawString(x, y, val)
        x += c.stringWidth(val, "Helvetica-Bold", 10) + 22
    c.setFont("Helvetica", 7)
    c.setFillColor(GREY)
    c.drawRightString(PAGE_W - MARGIN - 8, y, "Area = Qty × W × H per line · Holes column is per piece")


def _thumbnail(path: str | None, max_w: float, max_h: float):
    if not path or max_w < 60 or max_h < 60:
        return None
    try:
        img = ImageReader(str(path))
        iw, ih = img.getSize()
    except Exception:
        return None
    s = min(max_w / iw, max_h / ih)
    return img, iw * s, ih * s


def write_pdf(lpo: LPO, path: str | Path, enhanced_image_path: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
    c.setTitle(f"{lpo.lpo_no} - LPO draft")
    c.setAuthor("glass_lpo (generated draft)")

    _header(c, lpo)
    body_top = PAGE_H - MARGIN - HEADER_H
    body_bottom = MARGIN + FOOTER_H

    # Line items
    avail = body_top - body_bottom - TOTALS_H - MIN_BOTTOM
    table, th = _fit_table(lpo, avail)
    table.drawOn(c, MARGIN, body_top - th)
    y = body_top - th - 6

    _totals(c, lpo, y)
    y -= TOTALS_H + 4

    # Bottom band: review box (left) + thumbnail (right, if room)
    band_h = y - body_bottom
    thumb = _thumbnail(enhanced_image_path, 220, band_h - 14)
    gap = 12
    box_w = CONTENT_W - ((thumb[1] + gap) if thumb else 0)

    items = review_items(lpo)
    info = [i for i in info_items(lpo) if i[0] != "Read by"]  # model is in the header
    title_h = 22
    pad = 8
    if items:
        inner_w = box_w - 2 * pad
        rt, rh = _fit_review(items, inner_w, band_h - title_h - 2 * pad, info)
        box_h = min(band_h, rh + title_h + 2 * pad)
        top = y
        c.setFillColor(RED_BG)
        c.setStrokeColor(RED)
        c.setLineWidth(2)
        c.rect(MARGIN, top - box_h, box_w, box_h, stroke=1, fill=1)
        c.setFillColor(RED)
        c.rect(MARGIN, top - title_h, box_w, title_h, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(MARGIN + pad, top - 15, f"NEEDS HUMAN CHECK — {len(items)} "
                                             f"item{'s' if len(items) != 1 else ''}")
        c.setFont("Helvetica", 7.5)
        c.drawRightString(MARGIN + box_w - pad, top - 14.5,
                          "Confirm on the original drawing / with the client before cutting")
        rt.drawOn(c, MARGIN + pad, top - title_h - pad + 2 - rh)
    else:
        nt, nh = (_fit_review([], box_w - 2 * pad, band_h - 34 - pad, info)
                  if info else (None, 0))
        box_h = 34 + (nh + pad if nt else 0)
        c.setFillColor(GREEN_BG)
        c.setStrokeColor(GREEN)
        c.setLineWidth(1.5)
        c.rect(MARGIN, y - box_h, box_w, box_h, stroke=1, fill=1)
        c.setFillColor(GREEN)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(MARGIN + pad, y - 21, "No items flagged")
        c.setFont("Helvetica", 8)
        c.drawString(MARGIN + pad + 110, y - 20.5,
                     "All values read clearly. Still verify against the drawing before cutting.")
        if nt:
            nt.drawOn(c, MARGIN + pad, y - 30 - nh)

    if thumb:
        img, w, h = thumb
        x = PAGE_W - MARGIN - w
        c.setFont("Helvetica", 7)
        c.setFillColor(GREY)
        c.drawString(x, y - 8, "Source drawing (enhanced)")
        c.drawImage(img, x, y - 12 - h, w, h, preserveAspectRatio=True)
        c.setStrokeColor(LINE)
        c.setLineWidth(0.6)
        c.rect(x, y - 12 - h, w, h, stroke=1, fill=0)

    _footer(c, lpo)
    c.showPage()
    c.save()
    return path
