"""Excel + PDF export tests. Fixtures are real `validate()` output."""
from __future__ import annotations

import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from openpyxl import load_workbook

from glass_lpo.export_excel import area_value, review_items, write_excel
from glass_lpo.export_pdf import write_pdf
from glass_lpo.lpo_number import new_lpo_number
from glass_lpo.schema import (
    CHECK, EXCEL_COLUMNS, LPO, Drawing, Feature, Field, Hole, Panel,
)
from glass_lpo.validate import validate

NOT_WRITTEN = dict(value=None, clear=False, shown=False)


def F(v, clear=True, shown=True):
    return Field(value=v, clear=clear, shown=shown)


def raw_drawing() -> Drawing:
    """As extraction would hand it over (new contract), before validation."""
    clean = Panel(
        panel_id="P1", width_mm=F(1200.0), height_mm=F(2100.0), quantity=F(2),
        thickness_mm=F(10.0), glass_type=F("Clear toughened"),
        grooves_edgework=[Feature(kind="edge work", size=F("polished all edges"))],
    )
    unclear = Panel(
        panel_id="P2", width_mm=Field(value=None, clear=False, shown=True), height_mm=F(1800.0),
        quantity=F(1), thickness_mm=F(10.0), glass_type=F("Clear toughened"),
        holes=[
            Hole(diameter_mm=F(12.0), position=F("50 from left, 100 from top")),
            Hole(diameter_mm=F(12.0), position=F("50 from left, 900 from top")),
            Hole(diameter_mm=F(14.0, clear=False), position=F("50 from right", clear=False)),
        ],
        notes="width written as '7?0' - could be 700 or 760",
    )
    notched = Panel(
        panel_id="P3", width_mm=F(600.0), height_mm=F(2100.0), quantity=F(1),
        thickness_mm=F(10.0), glass_type=F("Frosted toughened"),
        notches_offcuts=[Feature(kind="notch", size=F("50 x 50"), position=F("bottom left"))],
        notes="Door side panel",
    )
    return Drawing(panels=[clean, unclear, notched], general_notes="Customer to confirm handle side",
                   model_used="gemini-test-model")


def fixture_lpo(drawing: Drawing | None = None) -> LPO:
    return LPO(lpo_no="LPO-20261124-001", date="2026-11-24",
               source_image="samples/originals/shower_screen.jpg",
               drawing=validate(drawing or raw_drawing()))


def pdf_text(path: Path) -> str:
    if not shutil.which("pdftotext"):
        pytest.skip("pdftotext not installed")
    out = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True)
    return re.sub(r"\s+", " ", out.stdout)


def pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(path)).pages)
    except ImportError:
        return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


# ------------------------------------------------------------------ Excel

def test_excel_layout_and_values(tmp_path):
    lpo = fixture_lpo()
    out = write_excel(lpo, tmp_path / "lpo.xlsx")
    wb = load_workbook(out)
    ws = wb["LPO Lines"]
    rows = list(ws.iter_rows(values_only=True))
    assert list(rows[0]) == EXCEL_COLUMNS
    assert len(rows) == 1 + 3
    col = {name: i for i, name in enumerate(EXCEL_COLUMNS)}
    p1, p2, p3 = rows[1:]

    # clear numerics are real numbers
    for name in ("Width (mm)", "Height (mm)", "Qty", "Thickness (mm)", "Area (m²)", "Line",
                 "Hole count"):
        assert isinstance(p1[col[name]], (int, float)), name
    assert p1[col["Width (mm)"]] == 1200
    assert p1[col["Area (m²)"]] == pytest.approx(5.04)
    assert p1[col["Needs check (Y/N)"]] == "N"
    assert CHECK not in " ".join(str(v) for v in p1 if v is not None)

    # written-but-unreadable width -> CHECK; area therefore CHECK, never blank
    assert p2[col["Width (mm)"]] == CHECK
    assert p2[col["Area (m²)"]] == CHECK
    assert p2[col["Hole count"]] == 3
    assert p2[col["Hole diameters (mm)"]] == f"12; 12; 14 ({CHECK})"
    assert p2[col["Hole positions"]].count("; ") == 2
    assert "700 or 760" in p2[col["Notes"]]
    assert p2[col["Needs check (Y/N)"]] == "Y"
    assert isinstance(p2[col["Height (mm)"]], (int, float))

    assert "notch" in p3[col["Notches/offcuts"]]
    assert p3[col["Needs check (Y/N)"]] == "N"
    assert not ws.merged_cells.ranges
    assert ws.freeze_panes == "A2"
    assert "ONE piece" in ws.cell(row=1, column=col["Hole count"] + 1).comment.text
    assert "Qty x Width x Height" in ws.cell(row=1, column=col["Area (m²)"] + 1).comment.text

    fl = list(wb["Flags"].iter_rows(min_row=2, values_only=True))
    numbered = [r for r in fl if isinstance(r[0], int)]
    assert [(r[2], r[3]) for r in numbered] == review_items(lpo)
    assert len(numbered) == len(lpo.all_flags())
    info = {r[2]: r[3] for r in fl if r[0] in (None, "") and r[2]}
    assert info["Drawing note"] == "Customer to confirm handle side"
    assert info["Read by"] == "gemini-test-model"
    readme = " ".join(r[3] for r in fl if r[2] == "Read me")
    assert "ONE piece" in readme and "Qty x Width x Height" in readme
    assert "before ERP import" in readme


def test_excel_medium_panel_needs_check_y(tmp_path):
    d = raw_drawing()
    d.panels[0].thickness_mm = Field(**NOT_WRITTEN)  # soft -> Medium
    d.panels[0].quantity = Field(**NOT_WRITTEN)      # soft -> Medium
    lpo = fixture_lpo(d)
    assert lpo.drawing.panels[0].confidence.value == "Medium"
    ws = load_workbook(write_excel(lpo, tmp_path / "m.xlsx"))["LPO Lines"]
    row = [c.value for c in ws[2]]
    assert row[EXCEL_COLUMNS.index("Needs check (Y/N)")] == "Y"
    assert row[EXCEL_COLUMNS.index("Qty")] == 1  # assumed 1, flagged softly
    assert any("Qty not written" in t for _, t in review_items(lpo))


def test_area_check_rules():
    p = Panel(panel_id="X", width_mm=F(1000.0), height_mm=F(500.0), quantity=Field(shown=True))
    assert area_value(p) == f"0.500 ({CHECK})"   # qty unreadable is NOT silently 1
    p.quantity = F(3, clear=False)
    assert area_value(p) == f"1.500 ({CHECK})"
    p.quantity = F(3)
    assert area_value(p) == 1.5
    p.width_mm = Field(**NOT_WRITTEN)
    assert area_value(p) == CHECK
    p.height_mm = Field(**NOT_WRITTEN)
    assert area_value(p) == ""                     # no size at all


def test_excel_no_flags(tmp_path):
    d = raw_drawing()
    d.panels = [d.panels[0]]
    d.general_notes = ""
    lpo = fixture_lpo(d)
    assert lpo.drawing.flags == [] and lpo.drawing.panels[0].flags == []
    wb = load_workbook(write_excel(lpo, tmp_path / "clean.xlsx"))
    rows = list(wb["Flags"].iter_rows(min_row=2, values_only=True))
    assert rows[0][3] == "No items flagged"


# ------------------------------------------------------------------ PDF

def test_pdf_one_page(tmp_path):
    out = write_pdf(fixture_lpo(), tmp_path / "lpo.pdf")
    assert out.exists() and out.stat().st_size > 1000
    assert pdf_page_count(out) == 1


def test_pdf_content_for_cutter_and_review(tmp_path):
    d = raw_drawing()
    d.overall_width_mm = F(3000.0, clear=False)
    d.overall_height_mm = F(2100.0)
    lpo = fixture_lpo(d)
    txt = pdf_text(write_pdf(lpo, tmp_path / "lpo.pdf"))
    assert "3000 (CHECK) × 2100 mm" in txt
    assert "gemini-test-model" in txt
    assert "Hole positions: 1) 50 from left, 100 from top" in txt
    assert "700 or 760" in txt                              # raw-text alternative visible
    assert "Customer to confirm handle side" in txt         # general notes in review box
    assert "includes 1 unverified line" in txt              # P2 area rests on CHECK width
    assert re.search(r"Total holes 3 \(CHECK\)", txt)        # hole 3 dia unclear
    for _, text in review_items(lpo):
        assert text[:40] in txt


def test_pdf_units_warning(tmp_path):
    d = raw_drawing()
    d.units_written = "unknown"   # unrecognised units are never converted
    txt = pdf_text(write_pdf(fixture_lpo(d), tmp_path / "unknown.pdf"))
    assert "NOT converted" in txt and "Units written in unknown" in txt


def test_pdf_stress_still_one_page(tmp_path):
    base = raw_drawing().panels
    panels = []
    for i in range(14):
        p = base[i % 3].model_copy(deep=True)
        p.panel_id = f"P{i + 1}"
        p.notes = "Long note about this panel that keeps going " * 3
        p.notches_offcuts = p.notches_offcuts + [
            Feature(kind="offcut", size=F("120 x 300"), position=F("top right corner")),
            Feature(kind="notch")]
        p.holes = p.holes + [Hole(diameter_mm=F(12.0), position=F("near the hinge", clear=False))]
        panels.append(p)
    d = Drawing(panels=panels, general_notes="A long general note. " * 20,
                overall_width_mm=F(5000.0), overall_height_mm=F(2100.0),
                extraction_issues=[f"Drawing-level issue number {i}: dimension chain does not add up "
                                   f"to the overall width shown on the sketch" for i in range(20)])
    lpo = fixture_lpo(d)
    out = write_pdf(lpo, tmp_path / "stress.pdf")
    assert pdf_page_count(out) == 1


def test_pdf_no_flags_and_thumbnail(tmp_path):
    from PIL import Image
    img = tmp_path / "enh.png"
    Image.new("RGB", (800, 600), "white").save(img)
    d = raw_drawing()
    d.panels = [d.panels[0]]
    out = write_pdf(fixture_lpo(d), tmp_path / "clean.pdf", enhanced_image_path=str(img))
    assert pdf_page_count(out) == 1
    assert "No items flagged" in pdf_text(out)


# ------------------------------------------------------------------ LPO number

def test_lpo_number_increments_per_day(tmp_path):
    d = dt.date(2026, 11, 24)
    assert new_lpo_number(d, tmp_path) == "LPO-20261124-001"
    assert new_lpo_number(d, tmp_path) == "LPO-20261124-002"
    assert new_lpo_number(dt.date(2026, 11, 25), tmp_path) == "LPO-20261125-001"
    (tmp_path / "LPO-20261124-007.xlsx").touch()  # counter lost/behind -> respect files
    assert new_lpo_number(d, tmp_path) == "LPO-20261124-008"
