"""End-to-end pipeline: photo -> enhanced image -> extraction -> validation -> Excel + PDF.

The LPO number is allocated only AFTER enhancement and extraction succeed:
the work happens in a temporary `outputs/_tmp-*` directory which is renamed
to `outputs/<LPO-no>` at the end. A failed run consumes no number and leaves
no directory behind.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .enhance import enhance_file
from .export_excel import write_excel
from .export_pdf import write_pdf
from .extract import extract_drawing
from .lpo_number import new_lpo_number
from .schema import LPO, Drawing
from .validate import validate


@dataclass
class RunResult:
    lpo: LPO
    run_dir: Path
    original: Path
    enhanced: Path
    excel: Path
    pdf: Path
    json: Path
    enhance_steps: list[str]


def run(image_path: str | Path, out_dir: str | Path = "outputs", *,
        drawing: Drawing | None = None, extract_from: str = "both",
        on_step: Callable[[str], None] | None = None) -> RunResult:
    """Run the full pipeline on one photo.

    drawing: pass a pre-extracted Drawing to skip the model call (testing/offline).
    extract_from: "both" (default: original + enhanced in one call),
      "enhanced" or "original" - which image(s) the model reads.
    on_step: optional callback, called with "enhance" / "extract" / "validate" /
      "export" as each stage starts (the web page uses it for live progress).
    """
    step = on_step or (lambda _name: None)
    image_path = Path(image_path)
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Expected a JPG or PNG photo, got {image_path.name}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if extract_from not in {"both", "enhanced", "original"}:
        raise ValueError(f"extract_from must be both/enhanced/original, got {extract_from!r}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # "_" prefix: ignored by lpo_number, so a crashed run never affects numbering.
    work = Path(tempfile.mkdtemp(prefix="_tmp-", dir=out))
    try:
        original = work / f"original{image_path.suffix.lower()}"
        shutil.copy2(image_path, original)
        enhanced = work / "enhanced.png"
        step("enhance")
        enh = enhance_file(original, enhanced)

        step("extract")
        if drawing is None:
            if extract_from == "both":
                drawing = extract_drawing(enhanced, original_path=original)
            else:
                drawing = extract_drawing(enhanced if extract_from == "enhanced" else original)
        step("validate")
        drawing = validate(drawing)

        # Extraction succeeded - only now take an LPO number.
        today = dt.date.today()
        lpo_no = new_lpo_number(today, outputs_dir=out)
        run_dir = out / lpo_no
        work.rename(run_dir)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise

    original, enhanced = run_dir / original.name, run_dir / enhanced.name
    lpo = LPO(lpo_no=lpo_no, date=today.isoformat(), source_image=image_path.name, drawing=drawing)

    step("export")
    json_path = run_dir / f"{lpo_no}.json"
    json_path.write_text(json.dumps(lpo.model_dump(mode="json"), indent=2))
    excel = write_excel(lpo, run_dir / f"{lpo_no}.xlsx")
    pdf = write_pdf(lpo, run_dir / f"{lpo_no}.pdf", enhanced_image_path=str(enhanced))

    return RunResult(lpo, run_dir, original, enhanced, Path(excel), Path(pdf), json_path, enh.steps)
