import json
from types import SimpleNamespace

import pytest

from glass_lpo import extract
from glass_lpo.extract import (
    ExtractionError,
    WireDrawing,
    extract_drawing,
    parse_wire_json,
    wire_to_drawing,
)
from glass_lpo.export_excel import fmt_num
from glass_lpo.schema import CHECK, Confidence
from glass_lpo.validate import validate


def num(v, legible="clear", raw=None, alts=()):
    return {"raw_text": raw if raw is not None else ("" if v is None else str(v)),
            "other_possible_readings": list(alts), "value": v, "legible": legible}


def panel(**kw):
    """Minimal wire panel with clear own 500x500 dimensions."""
    base = {"width_mm": num(500), "width_source": "own_dimension",
            "height_mm": num(500), "height_source": "own_dimension"}
    base.update(kw)
    return base


NOT = {"value": None, "legible": "not_shown", "raw_text": ""}

SAMPLE = {
    "units_written": "mm",
    "overall_width_mm": num(2500),
    "overall_height_mm": NOT,
    "panels": [
        {
            "panel_id": "A",
            "id_written_on_drawing": True,
            "width_mm": num(1000),
            "width_source": "own_dimension",
            "height_mm": num(2100),
            "height_source": "own_dimension",
            "quantity": num(2, raw="x2"),
            "thickness_mm": num(12, raw="12mm toughened"),
            "glass_type": {"value": "clear toughened", "legible": "clear", "raw_text": "12mm clear toughened"},
            "holes": [
                {
                    "count": 2,
                    "diameter_mm": num(12, raw="Ø12"),
                    "from_horizontal_edge": "left",
                    "horizontal_offset_mm": num(50),
                    "from_vertical_edge": "top",
                    "vertical_offset_mm": num(100),
                    "position_text": "",
                }
            ],
            "notches_offcuts": [
                {"kind": "Notch", "size": {"value": "50x50", "legible": "clear", "raw_text": "50x50"},
                 "position": {"value": "bottom left corner", "legible": "clear", "raw_text": ""}, "note": ""}
            ],
            "grooves_edgework": [
                {"kind": "polish", "size": {"value": None, "legible": "not_shown", "raw_text": ""},
                 "position": {"value": "all edges", "legible": "clear", "raw_text": "polish all round"}, "note": ""}
            ],
            "notes": "",
        },
        {
            "panel_id": "",
            "id_written_on_drawing": False,
            "width_mm": num(1700, "unclear", "1200 or 1700, smudged"),
            "width_source": "own_dimension",
            "height_mm": num(None, "unclear", "smudged"),
            "height_source": "own_dimension",
            "quantity": NOT,
            "thickness_mm": NOT,
            "glass_type": {"value": None, "legible": "not_shown", "raw_text": ""},
            "holes": [],
            "notches_offcuts": [],
            "grooves_edgework": [],
            "notes": "",
        },
    ],
    "general_notes": "Client: site 4",
}


def test_wire_schema_roundtrip():
    w = WireDrawing.model_validate(SAMPLE)
    assert WireDrawing.model_validate_json(w.model_dump_json()) == w


def test_wire_to_drawing_clear_values():
    d = wire_to_drawing(SAMPLE)
    a = d.panels[0]
    assert a.panel_id == "A"
    assert a.width_mm.value == 1000 and a.width_mm.clear
    assert a.quantity.value == 2 and isinstance(a.quantity.value, int) and a.quantity.clear
    assert a.thickness_mm.value == 12 and a.thickness_mm.clear
    assert a.glass_type.value == "clear toughened"
    assert len(a.holes) == 2  # count expanded
    assert a.holes[0].diameter_mm.value == 12
    assert a.holes[0].position.value == "50 from left, 100 from top" and a.holes[0].position.clear
    assert a.notches_offcuts[0].kind == "notch"
    assert a.grooves_edgework[0].position.value == "all edges"
    assert d.overall_width_mm.value == 2500 and d.overall_width_mm.clear
    assert not d.overall_height_mm.shown
    assert d.overall_height_mm.display() == ""
    assert "Client: site 4" in d.general_notes


def test_wire_to_drawing_unclear_never_silent():
    d = wire_to_drawing(SAMPLE)
    b = d.panels[1]
    assert b.panel_id == "P2"  # assigned
    assert "ID assigned" in b.notes
    assert b.width_mm.value == 1700 and not b.width_mm.clear
    assert b.width_mm.display(fmt_num) == f"1700 ({CHECK})"
    assert b.height_mm.value is None and b.height_mm.shown and b.height_mm.display() == CHECK
    assert "1200 or 1700" in b.notes  # raw text kept for the checker
    # quantity not written -> default 1, not flagged
    assert b.quantity.value == 1 and not b.quantity.shown
    assert not b.thickness_mm.shown


def test_required_dims_not_shown_is_not_shown_but_renders_check():
    data = {"panels": [panel(panel_id="X", width_mm=NOT, width_source="not_shown")]}
    p = wire_to_drawing(data).panels[0]
    assert not p.width_mm.shown and p.width_mm.value is None and not p.width_mm.clear
    assert p.width_mm.display(required=True) == CHECK


def test_written_but_unreadable_is_shown():
    data = {"panels": [panel(width_mm=num(None, "unclear", "smudge"))]}
    w = wire_to_drawing(data).panels[0].width_mm
    assert w.shown and w.value is None and not w.clear and w.display() == CHECK


def test_not_shown_hole_diameter_renders_check_when_required():
    data = {"panels": [panel(holes=[{"diameter_mm": NOT, "position_text": "centre"}])]}
    dia = wire_to_drawing(data).panels[0].holes[0].diameter_mm
    assert not dia.shown and dia.display(required=True) == CHECK


def test_clear_with_null_value_is_not_clear():
    data = {"panels": [panel(width_mm={"value": None, "legible": "clear", "raw_text": "?"})]}
    p = wire_to_drawing(data).panels[0]
    assert p.width_mm.value is None and not p.width_mm.clear


def test_fractional_quantity_forced_to_check():
    q = wire_to_drawing({"panels": [panel(quantity=num(2.5))]}).panels[0].quantity
    assert not q.clear


def test_alternatives_force_unclear_even_if_legible_clear():
    data = {"panels": [panel(width_mm=num(1200, "clear", "1200", alts=["1700"]))]}
    p = wire_to_drawing(data).panels[0]
    assert p.width_mm.value == 1200 and not p.width_mm.clear
    assert "width read 1200, could be 1700" in p.notes


def test_alternative_equal_to_value_is_ignored():
    data = {"panels": [panel(width_mm=num(1200, "clear", "1200", alts=["1200", " "]))]}
    assert wire_to_drawing(data).panels[0].width_mm.clear


def test_text_alternatives_force_unclear():
    gt = {"raw_text": "clear tough", "other_possible_readings": ["clear rough"],
          "value": "clear toughened", "legible": "clear"}
    p = wire_to_drawing({"panels": [panel(glass_type=gt)]}).panels[0]
    assert not p.glass_type.clear and "could be clear rough" in p.notes


@pytest.mark.parametrize("source", ["overall_dimension", "inferred", "not_shown"])
def test_panel_dim_not_own_dimension_is_not_shown(source):
    data = {"panels": [panel(width_mm=num(2500), width_source=source)]}
    p = wire_to_drawing(data).panels[0]
    assert p.width_mm.value is None and not p.width_mm.shown and not p.width_mm.clear
    assert p.width_mm.display(required=True) == CHECK
    assert "width not written for this panel" in p.notes and "2500" in p.notes
    assert p.height_mm.value == 500 and p.height_mm.clear  # own dimension untouched


def test_hole_unknown_edge_is_unclear():
    data = {"panels": [panel(holes=[{"diameter_mm": num(10), "horizontal_offset_mm": num(50),
                                     "from_horizontal_edge": "unknown",
                                     "vertical_offset_mm": num(60), "from_vertical_edge": "top"}])]}
    h = wire_to_drawing(data).panels[0].holes[0]
    assert h.position.shown and not h.position.clear


def test_hole_position_one_axis_only_is_unclear():
    data = {"panels": [panel(holes=[{"diameter_mm": num(10), "horizontal_offset_mm": num(50),
                                     "from_horizontal_edge": "left"}])]}
    p = wire_to_drawing(data).panels[0]
    pos = p.holes[0].position
    assert pos.value == "50 from left" and pos.shown and not pos.clear
    assert "one axis only" in p.notes


def test_hole_offset_alternatives_make_position_unclear():
    data = {"panels": [panel(holes=[{"diameter_mm": num(10),
                                     "horizontal_offset_mm": num(50, alts=["60"]), "from_horizontal_edge": "left",
                                     "vertical_offset_mm": num(100), "from_vertical_edge": "top"}])]}
    assert not wire_to_drawing(data).panels[0].holes[0].position.clear


def test_hole_drawn_but_not_dimensioned():
    data = {"panels": [panel(holes=[{"diameter_mm": num(10), "position_text": "near top right"}])]}
    h = wire_to_drawing(data).panels[0].holes[0]
    assert h.position.value == "near top right" and not h.position.clear


@pytest.mark.parametrize("count", [500, 0, -2])
def test_implausible_hole_count_kept_as_one_and_reported(count):
    data = {"panels": [panel(holes=[{"count": count, "diameter_mm": num(12)}])]}
    p = wire_to_drawing(data).panels[0]
    assert len(p.holes) == 1
    assert p.extraction_issues == [f"hole count '{count}' implausible (CHECK)"]


def test_notch_numeric_size():
    data = {"panels": [panel(notches_offcuts=[
        {"kind": "Notch", "width_mm": num(150), "height_mm": num(100), "size": NOT,
         "position": {"raw_text": "top left", "value": "top left", "legible": "clear"}}])]}
    f = wire_to_drawing(data).panels[0].notches_offcuts[0]
    assert f.kind == "notch" and f.size.value == "150x100" and f.size.clear


def test_notch_size_with_alternative_is_unclear():
    data = {"panels": [panel(notches_offcuts=[
        {"kind": "notch", "width_mm": num(150, alts=["160"]), "height_mm": num(100)}])]}
    f = wire_to_drawing(data).panels[0].notches_offcuts[0]
    assert f.size.value == "150x100" and not f.size.clear
    assert "width read 150, could be 160" in f.note


def test_notch_one_number_only_is_unclear():
    data = {"panels": [panel(notches_offcuts=[{"kind": "notch", "width_mm": num(150)}])]}
    f = wire_to_drawing(data).panels[0].notches_offcuts[0]
    assert f.size.value == "150x?" and not f.size.clear


@pytest.mark.parametrize("kind", ["notch", "offcut", "cutout"])
def test_sized_feature_without_size_is_check(kind):
    data = {"panels": [panel(notches_offcuts=[{"kind": kind}])]}
    f = wire_to_drawing(data).panels[0].notches_offcuts[0]
    assert f.size.shown and f.size.value is None and f.size.display() == CHECK


def test_groove_without_size_is_blank():
    data = {"panels": [panel(grooves_edgework=[{"kind": "polish"}])]}
    f = wire_to_drawing(data).panels[0].grooves_edgework[0]
    assert not f.size.shown and f.size.display() == ""


@pytest.mark.parametrize("written,expected", [
    ("mm", "mm"), ("MM", "mm"), ("millimetres", "mm"), ("", "none"), ("none written", "none"),
    ("cm", "cm"), ("Centimeters", "cm"), ("m", "m"), ("metres", "m"),
    ("inches", "inch"), ('"', "inch"), ("furlongs", "unknown"),
])
def test_units_normalised_not_converted(written, expected):
    d = wire_to_drawing({"units_written": written, "panels": [panel()]})
    assert d.units_written == expected
    assert d.panels[0].width_mm.value == 500  # never converted


def test_parse_wire_json_handles_fences():
    text = "```json\n" + json.dumps(SAMPLE) + "\n```"
    assert len(parse_wire_json(text).panels) == 2
    with pytest.raises(ValueError):
        parse_wire_json("")


def test_end_to_end_with_validate():
    # Exact flag wording belongs to validate.py's own tests; here we only check
    # that unclear extraction survives validation as a CHECK.
    d = validate(wire_to_drawing(json.dumps(SAMPLE)))
    b = d.panels[1]
    assert b.confidence == Confidence.LOW
    assert any("1700" in f and CHECK in f for f in b.flags)


def test_response_schema_accepted_by_sdk():
    pytest.importorskip("google.genai")
    from google.genai import types

    schema = extract.gemini_response_schema(None)
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
    assert "panels" in cfg.response_schema.required
    panel_s = cfg.response_schema.properties["panels"].items
    assert "width_mm" in panel_s.required and "width_source" in panel_s.required
    w = panel_s.properties["width_mm"]
    # transcribe before interpreting
    assert w.property_ordering == ["raw_text", "other_possible_readings", "value", "legible"]
    assert set(w.required) == set(w.property_ordering)
    assert w.properties["legible"].enum == ["clear", "unclear", "not_shown"]
    assert panel_s.properties["width_source"].enum == ["own_dimension", "overall_dimension", "inferred", "not_shown"]
    feat = panel_s.properties["notches_offcuts"].items
    assert {"width_mm", "height_mm"} <= set(feat.required)


# ---- extract_drawing with a fake Gemini client (no network) ---------------

class APIErr(Exception):
    """Stands in for google.genai.errors.APIError (has an int .code)."""

    def __init__(self, code, msg="err"):
        super().__init__(f"{code} {msg}")
        self.code = code


def ok(payload=None, finish="STOP"):
    return SimpleNamespace(text=json.dumps(payload or SAMPLE), prompt_feedback=None,
                           candidates=[SimpleNamespace(finish_reason=finish)])


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeClient:
    """responses: {model: [response | Exception, ...]}; each call pops one.
    Each call advances the fake clock by `call_s` seconds."""

    def __init__(self, responses, clock, call_s=10.0):
        self.responses = {m: list(r) for m, r in responses.items()}
        self.calls: list[str] = []
        self.kwargs: list[dict] = []
        self.clock, self.call_s = clock, call_s
        self.models = self

    def generate_content(self, **kw):
        self.calls.append(kw["model"])
        self.kwargs.append(kw)
        self.clock.t += self.call_s
        r = self.responses[kw["model"]].pop(0)
        if isinstance(r, Exception):
            raise r
        return r


MODELS = "m1,m2,m3"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Never touch the real .env / key / model list."""
    import dotenv

    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_DEADLINE_S", "GEMINI_CALL_TIMEOUT_S"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(dotenv, "find_dotenv", lambda *a, **k: "")


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(extract, "_clock", c)
    monkeypatch.setattr(extract, "_sleep", lambda s: setattr(c, "t", c.t + s))
    return c


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "drawing.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    return p


def run_fake(image, clock, responses, **kw):
    pytest.importorskip("google.genai")
    client = FakeClient(responses, clock, call_s=kw.pop("call_s", 10.0))
    try:
        return extract_drawing(image, client=client, model=kw.pop("model", MODELS), **kw), client
    except ExtractionError as e:
        e.client = client
        raise


def test_extract_no_key(image):
    with pytest.raises(ExtractionError, match="GEMINI_API_KEY"):
        extract_drawing(image)


def test_success_records_model_and_config(image, clock):
    d, c = run_fake(image, clock, {"m1": [ok()]})
    assert [p.panel_id for p in d.panels] == ["A", "P2"]
    assert d.model_used == "m1" and c.calls == ["m1"]
    cfg = c.kwargs[0]["config"]
    assert cfg.temperature == 0 and cfg.response_mime_type == "application/json"
    assert cfg.http_options.timeout == 75_000


def test_one_image_by_default_two_with_original(image, clock, tmp_path):
    _, c = run_fake(image, clock, {"m1": [ok()]})
    parts = [p for p in c.kwargs[0]["contents"] if not isinstance(p, str)]
    assert len(parts) == 1
    orig = tmp_path / "orig.jpg"
    orig.write_bytes(image.read_bytes())
    _, c = run_fake(image, clock, {"m1": [ok()]}, original_path=orig)
    contents = c.kwargs[0]["contents"]
    parts = [p for p in contents if not isinstance(p, str)]
    assert len(parts) == 2
    assert all(p.media_resolution is not None for p in parts)
    assert "Image 1 is the ORIGINAL" in contents[-1]


def test_429_goes_to_next_model(image, clock):
    d, c = run_fake(image, clock, {"m1": [APIErr(429)], "m2": [ok()]})
    assert c.calls == ["m1", "m2"] and d.model_used == "m2"


def test_404_goes_to_next_model(image, clock):
    d, c = run_fake(image, clock, {"m1": [APIErr(404)], "m2": [ok()]})
    assert c.calls == ["m1", "m2"]


def test_503_retries_once_then_next_model(image, clock):
    d, c = run_fake(image, clock, {"m1": [APIErr(503), APIErr(503)], "m2": [ok()]})
    assert c.calls == ["m1", "m1", "m2"] and d.model_used == "m2"


def test_503_then_success_on_retry(image, clock):
    d, c = run_fake(image, clock, {"m1": [APIErr(503), ok()]})
    assert c.calls == ["m1", "m1"] and d.model_used == "m1"


def test_timeout_retries_once(image, clock):
    d, c = run_fake(image, clock, {"m1": [TimeoutError("slow"), ok()]})
    assert c.calls == ["m1", "m1"]


@pytest.mark.parametrize("code", [400, 401, 403])
def test_auth_or_bad_request_fails_fast(image, clock, code):
    with pytest.raises(ExtractionError, match="rejected") as ei:
        run_fake(image, clock, {"m1": [APIErr(code, "API key not valid")], "m2": [ok()]})
    assert ei.value.client.calls == ["m1"]  # no other model tried


def test_parse_failure_goes_to_next_model_not_retry(image, clock):
    bad = SimpleNamespace(text="not json {", prompt_feedback=None, candidates=[])
    d, c = run_fake(image, clock, {"m1": [bad, ok()], "m2": [ok()]})
    assert c.calls == ["m1", "m2"] and d.model_used == "m2"


def test_schema_mismatch_goes_to_next_model(image, clock):
    bad = SimpleNamespace(text=json.dumps({"panels": "nope"}), prompt_feedback=None, candidates=[])
    _, c = run_fake(image, clock, {"m1": [bad], "m2": [ok()]})
    assert c.calls == ["m1", "m2"]


def test_all_models_fail(image, clock):
    with pytest.raises(ExtractionError, match="all models") as ei:
        run_fake(image, clock, {"m1": [APIErr(429)], "m2": [APIErr(429)], "m3": [APIErr(404)]})
    assert ei.value.client.calls == ["m1", "m2", "m3"]


def test_deadline_respected(image, clock):
    # each call "takes" 60 s; deadline 150 s -> at most 3 calls, then give up
    responses = {m: [APIErr(503), APIErr(503)] for m in ("m1", "m2", "m3")}
    with pytest.raises(ExtractionError, match="gave up after 150s") as ei:
        run_fake(image, clock, responses, call_s=60.0)
    assert len(ei.value.client.calls) == 3
    assert clock.t <= 150 + 60


def test_deadline_env_override_and_call_timeout_capped(image, clock, monkeypatch):
    monkeypatch.setenv("GEMINI_DEADLINE_S", "40")
    _, c = run_fake(image, clock, {"m1": [ok()]})
    assert c.kwargs[0]["config"].http_options.timeout == 40_000  # min(75, time left)


@pytest.mark.parametrize("finish,match", [("SAFETY", "safety"), ("MAX_TOKENS", "MAX_TOKENS")])
def test_finish_reason_raises_clear_error(image, clock, finish, match):
    with pytest.raises(ExtractionError, match=match) as ei:
        run_fake(image, clock, {"m1": [ok(finish=finish)], "m2": [ok()]})
    assert ei.value.client.calls == ["m1"]


def test_prompt_block_reason_raises(image, clock):
    blocked = SimpleNamespace(text=None, prompt_feedback=SimpleNamespace(block_reason="PROHIBITED_CONTENT"),
                              candidates=[])
    with pytest.raises(ExtractionError, match="blocked"):
        run_fake(image, clock, {"m1": [blocked]})


def test_model_list_from_env(image, clock, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "x1, x2")
    pytest.importorskip("google.genai")
    c = FakeClient({"x1": [APIErr(429)], "x2": [ok()]}, clock)
    d = extract_drawing(image, client=c)
    assert c.calls == ["x1", "x2"] and d.model_used == "x2"


def test_default_model_order_starts_with_reliable_model():
    assert extract.DEFAULT_MODEL.split(",")[0] == "gemini-3.5-flash"


def test_real_sdk_errors_classified():
    errors = pytest.importorskip("google.genai.errors")
    assert extract._classify(errors.ClientError(401, {"error": {"message": "bad key"}})) == "fatal"
    assert extract._classify(errors.ClientError(429, {"error": {"message": "quota"}})) == "next"
    assert extract._classify(errors.ServerError(503, {"error": {"message": "busy"}})) == "retry"
