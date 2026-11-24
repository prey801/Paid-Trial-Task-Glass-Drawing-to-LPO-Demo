"""Offline pipeline tests: no network, tiny generated image."""
import datetime as dt
import json

import cv2
import numpy as np
import pytest

from glass_lpo import pipeline
from glass_lpo.extract import ExtractionError
from glass_lpo.lpo_number import COUNTER_FILE, new_lpo_number
from glass_lpo.schema import Drawing, Field, Panel


@pytest.fixture
def photo(tmp_path):
    img = np.full((480, 640, 3), 235, np.uint8)
    cv2.rectangle(img, (80, 60), (560, 420), (20, 20, 20), 3)
    cv2.putText(img, "1200", (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    p = tmp_path / "photo.jpg"
    cv2.imwrite(str(p), img)
    return p


def drawing():
    return Drawing(panels=[Panel(panel_id="A", width_mm=Field(value=1200.0, clear=True),
                                 height_mm=Field(value=2000.0, clear=True))],
                   model_used="fake")


def lpo_dirs(out):
    return sorted(p.name for p in out.iterdir() if p.is_dir())


def test_offline_run_writes_outputs(photo, tmp_path):
    out = tmp_path / "outputs"
    res = pipeline.run(photo, out, drawing=drawing())
    today = dt.date.today().strftime("%Y%m%d")
    assert res.lpo.lpo_no == f"LPO-{today}-001"
    assert res.run_dir == out / res.lpo.lpo_no
    for f in (res.excel, res.pdf, res.json, res.original, res.enhanced):
        assert f.is_file() and f.parent == res.run_dir
    assert json.loads(res.json.read_text())["lpo_no"] == res.lpo.lpo_no
    assert lpo_dirs(out) == [res.lpo.lpo_no]  # no temp dir left behind


def test_pipeline_sends_original_and_enhanced(photo, tmp_path, monkeypatch):
    seen = {}

    def fake_extract(image_path, *, original_path=None, **kw):
        seen["enhanced"], seen["original"] = image_path, original_path
        assert image_path.is_file() and original_path.is_file()
        return drawing()

    monkeypatch.setattr(pipeline, "extract_drawing", fake_extract)
    pipeline.run(photo, tmp_path / "out")
    assert seen["enhanced"].name == "enhanced.png"
    assert seen["original"].name == "original.jpg"


def test_failed_extraction_leaves_no_dir_and_keeps_counter(photo, tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    first = pipeline.run(photo, out, drawing=drawing()).lpo.lpo_no
    counter_before = (out / COUNTER_FILE).read_text()

    def boom(*a, **k):
        raise ExtractionError("model down")

    monkeypatch.setattr(pipeline, "extract_drawing", boom)
    with pytest.raises(ExtractionError):
        pipeline.run(photo, out)
    assert lpo_dirs(out) == [first]
    assert (out / COUNTER_FILE).read_text() == counter_before

    monkeypatch.undo()
    second = pipeline.run(photo, out, drawing=drawing()).lpo.lpo_no
    assert int(second[-3:]) == int(first[-3:]) + 1  # no number was skipped


def test_lpo_number_ignores_underscore_dirs(tmp_path):
    d = dt.date(2026, 11, 24)
    fx = tmp_path / "_fixture_demo"
    fx.mkdir()
    (fx / "LPO-20261124-009.xlsx").touch()
    (tmp_path / "_tmp-abc").mkdir()
    (tmp_path / "_tmp-abc" / "LPO-20261124-005.pdf").touch()
    assert new_lpo_number(d, tmp_path) == "LPO-20261124-001"
    (tmp_path / "LPO-20261124-004").mkdir()  # a real run dir still counts
    assert new_lpo_number(d, tmp_path) == "LPO-20261124-005"


def test_rejects_non_image(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF")
    with pytest.raises(ValueError):
        pipeline.run(p, tmp_path / "out", drawing=drawing())
