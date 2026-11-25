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


# --- units read from the text written beside each number -----------------
import pytest

from glass_lpo.extract import normalise_units, wire_to_drawing
from glass_lpo.units import parse_length


@pytest.mark.parametrize("text,expected", [
    ("2.95m", (2.95, "m")), ("75cm", (75, "cm")), ("1200", (1200, None)),
    ("107 7/8", (107.875, None)), ("3/8\"", (0.375, "inch")), ("4 feet", (4, "ft")),
    ("3.5 feets", (3.5, "ft")), ("4'6\"", (54, "inch")), ("2050 O/A", (2050, None)),
    ("3⅜", (3.375, None)), ("12mm", (12, "mm")), ("", (None, None)),
])
def test_parse_length(text, expected):
    assert parse_length(text) == expected


@pytest.mark.parametrize("written,expected", [
    ("feet", "ft"), ("ft", "ft"), ("m and cm", "mixed"), ("m, cm", "mixed"), ("inch", "inch"),
])
def test_normalise_units_feet_and_mixed(written, expected):
    assert normalise_units(written) == expected


def n(raw, v, legible="clear"):
    return {"raw_text": raw, "other_possible_readings": [], "value": v, "legible": legible}


def wpanel(w, h, **kw):
    return {"width_mm": w, "width_source": "own_dimension",
            "height_mm": h, "height_source": "own_dimension", **kw}


def test_mixed_m_and_cm_each_converted():
    d = validate(wire_to_drawing({
        "units_written": "m, cm",
        "overall_width_mm": n("2.95m", 2.95), "overall_height_mm": n("2.4m", 2.4),
        "panels": [wpanel(n("75cm", 75), n("97cm", 97))]}))
    assert (d.overall_width_mm.value, d.overall_height_mm.value) == (2950, 2400)
    p = d.panels[0]
    assert (p.width_mm.value, p.height_mm.value) == (750, 970)
    assert p.width_mm.clear and p.height_mm.clear
    assert not any("NOT converted" in f for f in d.flags)
    assert any("each converted to mm" in f for f in d.flags)   # still asks for a confirm


def test_model_already_converted_value_is_not_scaled_twice():
    d = wire_to_drawing({"units_written": "m", "panels": [wpanel(n("2.95m", 2950), n("2.4m", 2.4))]})
    assert (d.panels[0].width_mm.value, d.panels[0].height_mm.value) == (2950, 2400)
    assert d.panels[0].width_mm.clear


def test_model_value_disagreeing_with_its_raw_text_is_check():
    d = wire_to_drawing({"units_written": "cm", "panels": [wpanel(n("75cm", 57), n("97cm", 97))]})
    assert d.panels[0].width_mm.value == 750 and not d.panels[0].width_mm.clear


def test_feet_converted():
    d = validate(wire_to_drawing({"units_written": "feet",
                                  "panels": [wpanel(n("2 ft", 2), n("4 feet", 4))]}))
    p = d.panels[0]
    assert (p.width_mm.value, p.height_mm.value) == (609.6, 1219.2)


def test_inches_with_fraction_no_unit_beside_number():
    # unit only stated for the drawing: converted by units.convert_to_mm
    d = validate(wire_to_drawing({"units_written": "inch",
                                  "overall_width_mm": n("107 7/8", 107.875),
                                  "panels": [wpanel(n("20", 20), n("40", 40))]}))
    assert d.overall_width_mm.value == 2740.0
    assert d.panels[0].width_mm.value == 508.0
    assert any("inch" in f and "converted" in f for f in d.flags)


def test_unitless_number_on_mixed_drawing_is_check():
    d = wire_to_drawing({"units_written": "mixed",
                         "overall_width_mm": n("2.95m", 2.95),
                         "panels": [wpanel(n("75", 75), n("97cm", 97))]})
    p = d.panels[0]
    assert p.width_mm.value == 75 and not p.width_mm.clear
    assert "no unit written next to it" in p.notes


def test_fraction_thickness_read_as_inches_and_checked():
    d = validate(wire_to_drawing({"units_written": "inch", "panels": [
        wpanel(n("20", 20), n("40", 40), thickness_mm=n("3/8", 0.375))]}))
    t = d.panels[0].thickness_mm
    assert t.value == 9.5 and not t.clear
    assert "read as inches" in d.panels[0].notes


def test_mm_thickness_untouched():
    d = wire_to_drawing({"units_written": "mm", "panels": [wpanel(n("500", 500), n("500", 500),
                                                                 thickness_mm=n("12mm", 12))]})
    assert d.panels[0].thickness_mm.value == 12 and d.panels[0].thickness_mm.clear
