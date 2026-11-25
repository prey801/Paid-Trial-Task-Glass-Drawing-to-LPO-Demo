import json

import pytest

from glass_lpo.evaluate import format_score, score
from glass_lpo.schema import Drawing, Feature, Field, Hole, Panel


def F(v, clear=True, shown=True):
    return Field(value=v, clear=clear, shown=shown)


NOT_WRITTEN = dict(value=None, clear=False, shown=False)


def truth_file(tmp_path, panels, **overall):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({**overall, "panels": panels}))
    return p


def tpanel(pid="P1", w=900, h=400, **kw):
    return {"panel_id": pid, "width_mm": w, "height_mm": h, "quantity": 1,
            "thickness_mm": None, "glass_type": None, "hole_count": 0,
            "hole_diameters_mm": [], "notches_offcuts": 0, "notch_sizes": [], **kw}


def gpanel(pid="P1", w=900, h=400, **kw):
    kw.setdefault("quantity", Panel(panel_id="x").quantity)  # not written
    return Panel(panel_id=pid, width_mm=F(w), height_mm=F(h), **kw)


def test_perfect_match_and_unwritten_qty_not_free(tmp_path):
    t = truth_file(tmp_path, [tpanel()])
    s = score(Drawing(panels=[gpanel()]), t)
    # width, height, hole count, notch count -> no free point for an unwritten qty=1
    assert (s.correct, s.flagged, s.wrong_confident, s.missing) == (4, 0, 0, 0)
    assert "correct & confident" in format_score(s)


def test_qty_scored_when_written_or_not_one(tmp_path):
    t = truth_file(tmp_path, [tpanel(quantity=2)])
    assert score(Drawing(panels=[gpanel(quantity=F(2))]), t).correct == 5
    s = score(Drawing(panels=[gpanel()]), t)  # truth 2, extraction silently 1
    assert s.wrong_confident == 1


def test_unclear_hole_diameters_are_flagged_not_right_or_wrong(tmp_path):
    t = truth_file(tmp_path, [tpanel(hole_count=3, hole_diameters_mm=[25, 25, 25])])
    holes = [Hole(diameter_mm=F(25.0)), Hole(diameter_mm=F(30.0, clear=False)),
             Hole(diameter_mm=Field(shown=True))]
    s = score(Drawing(panels=[gpanel(holes=holes)]), t)
    assert s.flagged == 2 and s.wrong_confident == 0
    assert s.correct == 2 + 1 + 1 + 1  # w, h, hole count, notch count + one clear dia
    holes[0] = Hole(diameter_mm=F(12.0))
    assert score(Drawing(panels=[gpanel(holes=holes)]), t).wrong_confident == 1


def test_missing_hole_diameters(tmp_path):
    t = truth_file(tmp_path, [tpanel(hole_count=2, hole_diameters_mm=[25, 25])])
    s = score(Drawing(panels=[gpanel(holes=[Hole(diameter_mm=F(25.0))])]), t)
    assert s.missing == 1 and s.wrong_confident == 1  # hole count 1 vs 2


def test_index_fallback_never_matches_twice(tmp_path):
    t = truth_file(tmp_path, [tpanel("A"), tpanel("B", h=450)])
    s = score(Drawing(panels=[gpanel("B", h=450)]), t)
    # B matched by ID; A must not fall back onto that same panel
    assert any("MISSING panel A" in d for d in s.details)
    assert s.wrong_confident == 0


def test_extra_panels_penalised_each(tmp_path):
    t = truth_file(tmp_path, [tpanel("P1")])
    s = score(Drawing(panels=[gpanel("P1"), gpanel("P2"), gpanel("P3")]), t)
    assert s.wrong_confident == 2
    assert sum("EXTRA" in d for d in s.details) == 2


def test_value_reported_where_truth_is_null(tmp_path):
    t = truth_file(tmp_path, [tpanel()])
    s = score(Drawing(panels=[gpanel(glass_type=F("HARD - GLASS"), thickness_mm=F(10.0))]), t)
    assert s.wrong_confident == 2
    s = score(Drawing(panels=[gpanel(glass_type=F("HARD - GLASS", clear=False))]), t)
    assert s.wrong_confident == 0 and s.flagged == 1


def test_notch_size_scored_and_missing(tmp_path):
    t = truth_file(tmp_path, [tpanel(notches_offcuts=1, notch_sizes=["150x100"])])
    ok = gpanel(notches_offcuts=[Feature(kind="notch", size=F("150 x 100 mm"))])
    assert score(Drawing(panels=[ok]), t).correct == 5
    s = score(Drawing(panels=[gpanel()]), t)
    assert s.missing == 1 and s.wrong_confident == 1  # notch count 0 vs 1


def test_hole_positions_only_scored_when_in_truth(tmp_path):
    holes = [Hole(diameter_mm=F(25.0), position=F("50 from left, 60 from top"))]
    t = truth_file(tmp_path, [tpanel(hole_count=1, hole_diameters_mm=[25])])
    base = score(Drawing(panels=[gpanel(holes=holes)]), t).total
    t = truth_file(tmp_path, [tpanel(hole_count=1, hole_diameters_mm=[25],
                                     hole_positions=["60 from top, 50 from left"])])
    s = score(Drawing(panels=[gpanel(holes=holes)]), t)
    assert s.total == base + 1 and s.wrong_confident == 0
    holes[0].position = F("top left", clear=False)
    assert score(Drawing(panels=[gpanel(holes=holes)]), t).flagged == 1


def test_overall_and_not_written_missing(tmp_path):
    t = truth_file(tmp_path, [tpanel()], overall_width_mm=900, overall_height_mm=None)
    p = gpanel()
    p.width_mm = Field(**NOT_WRITTEN)
    s = score(Drawing(panels=[p], overall_width_mm=F(900.0)), t)
    assert s.missing == 1 and s.correct == 4


@pytest.mark.parametrize("name", ["synthetic_simple", "synthetic_realistic", "synthetic_hard"])
def test_truth_files_parse(name):
    from pathlib import Path
    t = Path(__file__).parent.parent / "samples" / "synthetic" / f"{name}.truth.json"
    if not t.exists():
        pytest.skip("sample truth missing")
    s = score(Drawing(), t)
    assert s.missing > 0 and s.correct == 0


def test_panels_matched_by_size_before_id(tmp_path):
    # Output numbered the window P2 (IDs not written); the truth calls it P3.
    t = truth_file(tmp_path, [tpanel("P1", w=None, h=None), tpanel("P2", w=None, h=None),
                              tpanel("P3", w=750, h=970)])
    s = score(Drawing(panels=[gpanel("P1", w=None, h=None), gpanel("P2", w=750, h=970)]), t)
    assert not any("P3 width" in d or "P3 height" in d for d in s.details)
    assert any("MISSING panel P2" in d for d in s.details)


def test_glass_type_scored_by_meaning(tmp_path):
    t = truth_file(tmp_path, [tpanel(glass_type="14mm unit (double glazed)")])
    s = score(Drawing(panels=[gpanel(glass_type=F("UNIT"))]), t)
    assert s.wrong_confident == 0
    t = truth_file(tmp_path, [tpanel(glass_type="clear toughened")])
    s = score(Drawing(panels=[gpanel(glass_type=F("frosted toughened"))]), t)
    assert s.wrong_confident == 1
