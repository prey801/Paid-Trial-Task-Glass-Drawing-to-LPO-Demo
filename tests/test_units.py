"""Client rule: no unit written -> assume cm and convert to mm, always flagged."""
from glass_lpo.schema import Confidence, Drawing, Feature, Field, Hole, Panel
from glass_lpo.validate import validate


def F(v, clear=True):
    return Field(value=v, clear=clear, shown=True)


def door(units: str) -> Drawing:
    return Drawing(
        units_written=units,
        overall_width_mm=F(90), overall_height_mm=F(210),
        panels=[
            Panel(panel_id="P1", width_mm=F(90), height_mm=F(100), thickness_mm=F(10),
                  quantity=F(1), holes=[Hole(diameter_mm=F(25), position=F("5 from left, 10 from top"))]),
            Panel(panel_id="P2", width_mm=F(90), height_mm=F(110), thickness_mm=F(10), quantity=F(1),
                  notches_offcuts=[Feature(kind="notch", size=F("15x10"), position=F("10 from right, 5 from bottom"))]),
        ],
    )


def test_no_unit_assumed_cm_converted_and_flagged():
    d = validate(door("none"))
    assert (d.overall_width_mm.value, d.overall_height_mm.value) == (900, 2100)
    assert [p.height_mm.value for p in d.panels] == [1000, 1100]
    assert d.panels[0].holes[0].diameter_mm.value == 25          # Ø25 stays mm
    assert d.panels[0].holes[0].position.value == "50 from left, 100 from top"
    assert d.panels[1].notches_offcuts[0].size.value == "150x100"
    assert d.panels[0].thickness_mm.value == 10                  # thickness never converted
    assert d.units_written == "mm" and "assumed" in d.units_converted_from
    assert any("assumed cm" in f and "converted to mm" in f for f in d.flags)
    assert all(p.confidence == Confidence.MEDIUM for p in d.panels)  # flagged, not silent
    assert not any("don't add up" in f for f in d.flags)          # converted sizes verify


def test_written_cm_converted_too():
    d = validate(door("cm"))
    assert d.overall_width_mm.value == 900 and any("written in cm" in f for f in d.flags)


def test_written_mm_untouched():
    d = door("mm")
    out = validate(d)
    assert out.overall_width_mm.value == 90 and not out.units_converted_from


def test_small_hole_diameter_treated_as_drawing_unit():
    d = door("none")
    d.panels[0].holes[0].diameter_mm = F(2.5)
    out = validate(d)
    assert out.panels[0].holes[0].diameter_mm.value == 25
    assert any("hole diameter" in f and "confirm" in f for f in out.flags)


def test_validate_twice_does_not_double_convert():
    once = validate(door("none"))
    twice = validate(once)
    assert twice.overall_width_mm.value == 900 and twice.panels[0].height_mm.value == 1000


def test_safety_rule_keeps_plausible_mm_drawing_as_mm():
    # "900 x 2100" with no unit: as cm it would be 9000 x 21000 mm (impossible),
    # as written it is a sensible mm door -> keep mm, flagged for confirmation.
    d = door("none")
    d.overall_width_mm, d.overall_height_mm = F(900), F(2100)
    for p, h in zip(d.panels, (1000, 1100)):
        p.width_mm, p.height_mm = F(900), F(h)
    out = validate(d)
    assert out.overall_width_mm.value == 900 and out.panels[0].height_mm.value == 1000
    assert out.panels[0].holes[0].position.value == "5 from left, 10 from top"   # untouched
    assert any("read as mm" in f for f in out.flags)
    assert all(p.confidence == Confidence.MEDIUM for p in out.panels)
    assert not any("don't add up" in f for f in out.flags)


def test_safety_rule_not_used_when_as_written_is_implausible_too():
    # 9000 as written is already > 6000 mm, so mm isn't "sensible" either -> normal cm rule.
    d = door("none")
    d.overall_width_mm = F(9000)
    out = validate(d)
    assert "assumed cm" in out.units_converted_from
