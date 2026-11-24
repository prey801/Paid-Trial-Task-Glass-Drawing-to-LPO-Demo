from glass_lpo.export_excel import fmt_num, review_items
from glass_lpo.schema import LPO, Confidence, Drawing, Feature, Field, Hole, Panel
from glass_lpo.validate import _n, parse_offsets, parse_size, validate


def F(v, clear=True, shown=True):
    return Field(value=v, clear=clear, shown=shown)


NOT_WRITTEN = dict(value=None, clear=False, shown=False)


def panel(pid="P1", w=1000, h=2000, t=12, **kw):
    kw.setdefault("quantity", F(1))
    return Panel(
        panel_id=pid,
        width_mm=F(w),
        height_mm=F(h),
        thickness_mm=F(t) if t is not None else Field(shown=False),
        glass_type=F("clear toughened"),
        **kw,
    )


def has(flags, text):
    return any(text in f for f in flags)


def test_clean_panel_is_high():
    d = validate(Drawing(panels=[panel()]))
    p = d.panels[0]
    assert p.flags == [] and d.flags == []
    assert p.confidence == Confidence.HIGH


def test_does_not_mutate_input_and_is_idempotent():
    src = Drawing(panels=[panel()])
    src.panels[0].width_mm.clear = False
    once = validate(src)
    twice = validate(once)
    assert src.panels[0].flags == []
    assert once.panels[0].flags == twice.panels[0].flags


def test_unclear_and_missing_values_are_check_and_low():
    p = panel()
    p.width_mm = F(1200, clear=False)
    p.height_mm = Field()  # written, unreadable
    d = validate(Drawing(panels=[p]))
    f = d.panels[0].flags
    assert has(f, "width 1200 unclear (CHECK)")
    assert has(f, "height unreadable (CHECK)")
    p.height_mm = Field(**NOT_WRITTEN)
    assert has(validate(Drawing(panels=[p])).panels[0].flags, "height not written (CHECK)")
    assert d.panels[0].confidence == Confidence.LOW


def test_missing_thickness_is_soft():
    d = validate(Drawing(panels=[panel(t=None)]))
    assert d.panels[0].flags == ["thickness not stated"]
    assert d.panels[0].confidence == Confidence.MEDIUM


def test_nonstandard_thickness():
    d = validate(Drawing(panels=[panel(t=7)]))
    assert has(d.panels[0].flags, "thickness 7 mm not standard")
    assert d.panels[0].confidence == Confidence.LOW


def test_size_limits():
    d = validate(Drawing(panels=[panel(w=30, h=7000)]))
    f = d.panels[0].flags
    assert has(f, "width 30 mm outside") and has(f, "height 7000 mm outside")

    d = validate(Drawing(panels=[panel(w=1000, h=4000)]))
    assert has(d.panels[0].flags, "near sheet limit")
    assert d.panels[0].confidence == Confidence.MEDIUM

    d = validate(Drawing(panels=[panel(w=3300, h=4000)]))
    assert has(d.panels[0].flags, "exceeds jumbo sheet")
    assert d.panels[0].confidence == Confidence.LOW


def test_quantity_rules():
    d = validate(Drawing(panels=[panel(quantity=F(0))]))
    assert has(d.panels[0].flags, "qty 0 outside")
    d = validate(Drawing(panels=[panel(quantity=F(2, clear=False))]))
    assert has(d.panels[0].flags, "qty 2 unclear (CHECK)")
    d = validate(Drawing(panels=[panel(quantity=Field(shown=True))]))
    assert has(d.panels[0].flags, "qty unreadable (CHECK)")
    assert d.panels[0].confidence == Confidence.LOW


def test_qty_not_written_is_soft_and_hoisted_once():
    for q in (Panel(panel_id="x").quantity, Field(**NOT_WRITTEN)):  # default + new contract
        ps = [panel("P1", quantity=q), panel("P2", quantity=q), panel("P3")]
        d = validate(Drawing(panels=ps))
        assert d.panels[0].confidence == d.panels[1].confidence == Confidence.MEDIUM
        assert d.panels[2].confidence == Confidence.HIGH
        assert has(d.panels[0].flags, "qty not written - assumed 1")
        items = review_items(LPO(lpo_no="L", date="2026-01-01", source_image="", drawing=d))
        qty_lines = [t for _, t in items if "Qty not written" in t or "qty not written" in t]
        assert qty_lines == ["Qty not written - assumed 1 per panel (P1, P2)"]


def test_hole_rules():
    holes = [
        Hole(diameter_mm=F(12), position=F("100 from left, 50 from top")),
        Hole(diameter_mm=F(8), position=F("100 from left, 80 from top")),  # < 12mm glass
        Hole(diameter_mm=F(150), position=F("1200 from left, 50 from top")),  # too big, outside
        Hole(diameter_mm=F(12, clear=False), position=Field(shown=False)),
        Hole(diameter_mm=Field(**NOT_WRITTEN), position=F("60 from left", clear=False)),
    ]
    d = validate(Drawing(panels=[panel(holes=holes)]))
    f = d.panels[0].flags
    assert not has(f, "hole 1")
    assert has(f, "hole 2 Ø8 < glass 12 mm")
    assert has(f, "hole 3 Ø150 outside 4-100")
    assert has(f, "hole 3 1200 from left outside panel width 1000")
    assert has(f, "hole 4 dia 12 unclear (CHECK)")
    assert has(f, "hole positions not dimensioned (hole 4)")
    assert has(f, "hole 5 dia not written (CHECK)")
    assert has(f, "hole 5 position '60 from left' unclear (CHECK)")  # numbers but unclear: hard
    assert d.panels[0].confidence == Confidence.LOW


def test_undimensioned_hole_positions_soft_and_hoisted():
    def holed(pid):
        return panel(pid, holes=[Hole(diameter_mm=F(12), position=F("top left", clear=False)),
                                 Hole(diameter_mm=F(12), position=Field(shown=False))])
    d = validate(Drawing(panels=[holed("P1"), holed("P2"), panel("P3")]))
    assert d.panels[0].flags == ["hole positions not dimensioned"]
    assert d.panels[0].confidence == Confidence.MEDIUM
    items = review_items(LPO(lpo_no="L", date="2026-01-01", source_image="", drawing=d))
    assert items == [("Drawing", "Hole positions not dimensioned - confirm with client (P1, P2)")]
    # single panel: listed against that panel
    d = validate(Drawing(panels=[holed("P1")]))
    items = review_items(LPO(lpo_no="L", date="2026-01-01", source_image="", drawing=d))
    assert items == [("P1", "hole positions not dimensioned")]


def test_notch_size_missing_or_too_big():
    feats = [Feature(kind="notch"), Feature(kind="cutout", size=Field(shown=True)),
             Feature(kind="offcut", size=F("1200 x 2100")), Feature(kind="notch", size=F("150x100"))]
    d = validate(Drawing(panels=[panel(notches_offcuts=feats)]))
    f = d.panels[0].flags
    assert has(f, "notch size not given (CHECK)")
    assert has(f, "cutout size unreadable (CHECK)")
    assert has(f, "offcut 1200x2100 larger than panel 1000x2000 (CHECK)")
    assert not has(f, "notch 150x100")
    assert d.panels[0].confidence == Confidence.LOW
    assert parse_size("150 x 100mm") == (150.0, 100.0) and parse_size("bottom left") is None


def test_extraction_issues_are_hard():
    p = panel(extraction_issues=["hole count implausible (9 holes)"])
    d = validate(Drawing(panels=[p], extraction_issues=["second read disagreed on P1 width"]))
    assert has(d.panels[0].flags, "hole count implausible (9 holes) (CHECK)")
    assert d.panels[0].confidence == Confidence.LOW
    assert has(d.flags, "second read disagreed on P1 width (CHECK)")


def test_units_not_mm_hard_and_all_low():
    # inch / unknown units are never converted -> hard CHECK (cm/m are converted, see test_units.py)
    d = validate(Drawing(panels=[panel("P1"), panel("P2")], units_written="inch"))
    assert has(d.flags, "Units written in inch - values NOT converted (CHECK)")
    assert all(p.confidence == Confidence.LOW for p in d.panels)
    items = review_items(LPO(lpo_no="L", date="2026-01-01", source_image="", drawing=d))
    assert sum("units" in t.lower() for _, t in items) == 1  # listed once
    for u in ("mm", "", "MM "):
        d = validate(Drawing(panels=[panel()], units_written=u))
        assert d.flags == [] and d.panels[0].confidence == Confidence.HIGH


def test_glass_type_vocabulary():
    p = panel()
    p.glass_type = F("HARD - GLASS SHOWER SCREEN DRAWING")
    assert not has(validate(Drawing(panels=[p])).panels[0].flags, "not recognised")  # has "glass"
    p.glass_type = F("KITCHEN SPLASHBACK")
    d = validate(Drawing(panels=[p]))
    assert has(d.panels[0].flags, "glass type 'KITCHEN SPLASHBACK' not recognised - confirm")
    assert d.panels[0].confidence == Confidence.MEDIUM
    for ok in ("10mm Clear Toughened", "frosted", "Low-E laminated", "bronze tinted", "mirror"):
        p.glass_type = F(ok)
        assert validate(Drawing(panels=[p])).panels[0].flags == [], ok


def test_medium_panel_always_needs_check():
    d = validate(Drawing(panels=[panel(t=None)]))
    p = d.panels[0]
    assert p.confidence == Confidence.MEDIUM and p.needs_check and p.flags


def test_number_format_matches_exports():
    for v in (1200.0, 6.5, 0.1 + 0.2, 12, 1234.5678):
        assert _n(v) == fmt_num(v)
    assert fmt_num(0.1 + 0.2) == "0.3" and fmt_num(1200.0) == "1200" and _n(None) == "?"


def test_feature_position_and_unclear():
    feats = [Feature(kind="notch", size=F("50x50"), position=F("2500 from bottom"))]
    feats[0].size = F("50x50")
    edge = [Feature(kind="polish", position=F("all edges", clear=False))]
    d = validate(Drawing(panels=[panel(notches_offcuts=feats, grooves_edgework=edge)]))
    f = d.panels[0].flags
    assert has(f, "notch 2500 from bottom outside panel height 2000")
    assert has(f, "polish position 'all edges' unclear (CHECK)")


def test_parse_offsets():
    assert parse_offsets("50mm from left, 100 from the top") == [(50.0, "left"), (100.0, "top")]
    assert parse_offsets("centre") == []


def test_zero_panels_and_duplicates():
    d = validate(Drawing())
    assert has(d.flags, "No panels found")
    d = validate(Drawing(panels=[panel("A"), panel("a")]))
    assert has(d.flags, "Duplicate panel ID")
    assert all(p.confidence == Confidence.LOW for p in d.panels)


def overall(w=None, h=None):
    return dict(
        overall_width_mm=F(w) if w is not None else Field(shown=False),
        overall_height_mm=F(h) if h is not None else Field(shown=False),
    )


def test_side_by_side_sum_ok():
    d = validate(Drawing(panels=[panel("P1", 1000, 2000), panel("P2", 1502, 2000)], **overall(2500, 2000)))
    assert d.flags == []
    assert all(p.confidence == Confidence.HIGH for p in d.panels)


def test_stacked_sum_ok():
    d = validate(Drawing(panels=[panel("P1", 1200, 500), panel("P2", 1200, 1500)], **overall(1200, 2000)))
    assert d.flags == []


def test_quantity_counts_in_sum():
    d = validate(
        Drawing(panels=[panel("P1", 800, 2000, quantity=F(2)), panel("P2", 900, 2000)], **overall(2500, 2000))
    )
    assert d.flags == []


def test_sum_mismatch_flags_drawing_and_panels():
    d = validate(Drawing(panels=[panel("P1", 1000, 2000), panel("P2", 1400, 2000)], **overall(2500)))
    assert has(d.flags, "widths sum 2400 vs overall 2500")
    assert all(p.confidence == Confidence.LOW for p in d.panels)


def test_many_panels_layout_soft_not_hard():
    ps = [panel(f"P{i}", 100 * (i % 4 + 1), 100 * (i % 3 + 1)) for i in range(14)]
    d = validate(Drawing(panels=ps, **overall(1000, 600)))
    assert has(d.flags, "too many panels")
    assert all(has(p.flags, "layout not verified (too many panels)") for p in d.panels)
    assert all(p.confidence == Confidence.MEDIUM for p in d.panels)


def test_grid_layout_is_soft():
    ps = [panel("P1", 1000, 1000), panel("P2", 1500, 1000), panel("P3", 2500, 1000)]
    d = validate(Drawing(panels=ps, **overall(2500, 2000)))
    assert has(d.flags, "partly verified")
    assert all(p.confidence == Confidence.MEDIUM for p in d.panels)


def test_layout_unverifiable_when_size_missing():
    p2 = panel("P2")
    p2.width_mm = Field()
    d = validate(Drawing(panels=[panel("P1"), p2], **overall(2000)))
    assert has(d.flags, "not verified")
    assert d.panels[0].confidence == Confidence.MEDIUM
    assert d.panels[1].confidence == Confidence.LOW


def test_overall_unclear_flagged():
    d = validate(
        Drawing(
            panels=[panel("P1", 1000, 2000)],
            overall_width_mm=F(1000, clear=False),
            overall_height_mm=Field(shown=False),
        )
    )
    assert has(d.flags, "Overall width 1000 unclear (CHECK)")


def _stacked(width_fields):
    return Drawing(
        overall_width_mm=F(900), overall_height_mm=F(2100),
        panels=[panel(f"P{i}", None, h, t=10) for i, h in enumerate([400, 450, 350, 500, 400], start=1)],
    ), width_fields


def test_stacked_panel_width_filled_from_overall_with_flag():
    d, _ = _stacked(None)
    for p in d.panels:
        p.width_mm = Field(**NOT_WRITTEN)
    d.panels[0].width_mm = F(900)  # a written width that matches the overall is fine
    out = validate(d)
    assert out.panels[0].width_mm.clear and out.panels[0].flags == []
    for p in out.panels[1:]:
        assert p.width_mm.value == 900 and not p.width_mm.clear  # derived -> CHECK
        assert p.width_mm.display(fmt_num, required=True) == "900 (CHECK)"
        assert p.flags == ["width 900 taken from overall width (not written)"]
        assert p.confidence == Confidence.MEDIUM and p.needs_check and p.area_uncertain
    assert not out.flags  # stacked layout now verifies
    again = validate(out)  # idempotent
    assert [p.flags for p in again.panels] == [p.flags for p in out.panels]


def test_unreadable_width_not_filled_from_overall():
    d, _ = _stacked(None)
    for p in d.panels:
        p.width_mm = Field(**NOT_WRITTEN)
    d.panels[2].width_mm = Field(value=None, clear=False, shown=True)  # written, unreadable
    out = validate(d)
    assert out.panels[2].width_mm.value is None
    assert has(out.panels[2].flags, "width unreadable (CHECK)")
    assert out.panels[2].confidence == Confidence.LOW
    assert out.panels[0].width_mm.value == 900  # the not-written ones still filled


def test_no_fill_when_written_width_disagrees_with_overall():
    d, _ = _stacked(None)
    for p in d.panels:
        p.width_mm = Field(**NOT_WRITTEN)
    d.panels[0].width_mm = F(600)
    out = validate(d)
    assert all(p.width_mm.value is None for p in out.panels[1:])
    assert has(out.panels[1].flags, "width not written (CHECK)")


def test_written_width_never_overridden():
    d = Drawing(
        overall_width_mm=F(900), overall_height_mm=F(800),
        panels=[panel("P1", 1700, 400), panel("P2", None, 400)],
    )
    d.panels[0].width_mm = F(1700, clear=False)
    out = validate(d)
    assert out.panels[0].width_mm.value == 1700 and not out.panels[0].width_mm.clear
