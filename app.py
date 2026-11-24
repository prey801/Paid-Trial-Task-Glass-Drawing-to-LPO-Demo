"""Simple web UI:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from glass_lpo.cli import area_text
from glass_lpo.export_excel import fmt_num, review_items
from glass_lpo.pipeline import run
from glass_lpo.schema import CHECK

st.set_page_config(page_title="Glass Drawing → LPO", layout="wide")
st.title("Glass drawing → LPO draft")
st.caption("Upload a phone photo of a glass drawing. Every value that can't be read clearly is marked CHECK.")

up = st.file_uploader("Drawing photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
truth = st.file_uploader("Optional: ground-truth JSON to score against", type=["json"])

if up and st.button("Generate LPO", type="primary"):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / up.name
        src.write_bytes(up.getvalue())
        with st.spinner("Enhancing image and reading drawing…"):
            try:
                res = run(src, "outputs")
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.stop()

    lpo = res.lpo
    st.subheader(f"{lpo.lpo_no} · {lpo.date}")
    st.caption(f"Read by {lpo.drawing.model_used or '-'} · units written: {lpo.drawing.units_written}")

    c1, c2 = st.columns(2)
    c1.image(str(res.original), caption="Original", width="stretch")
    c2.image(str(res.enhanced), caption="Enhanced", width="stretch")
    st.caption("Enhancement: " + "; ".join(res.enhance_steps))

    def area(p):
        if p.area_m2 is None:
            return CHECK
        return f"{fmt_num(p.area_m2)} ({CHECK})" if p.area_uncertain else fmt_num(p.area_m2)

    rows = [{
        "Panel": p.panel_id,
        "Width (mm)": p.width_mm.display(fmt_num, required=True),
        "Height (mm)": p.height_mm.display(fmt_num, required=True),
        "Qty": p.quantity.display(fmt_num), "Thk": p.thickness_mm.display(fmt_num),
        "Type": p.glass_type.display(), "Area m²": area(p), "Holes": len(p.holes),
        "Hole Ø": "; ".join(h.diameter_mm.display(fmt_num, required=True) for h in p.holes),
        "Notches/offcuts": "; ".join(
            f"{f.kind} {f.size.display()}".strip() for f in p.notches_offcuts),
        "Confidence": p.confidence.value, "Check": "Y" if p.needs_check else "N",
    } for p in lpo.drawing.panels]
    st.dataframe(rows, width="stretch", hide_index=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Panels", len(lpo.drawing.panels))
    n_unc = lpo.uncertain_area_lines
    m2.metric("Total area", f"{fmt_num(lpo.total_area_m2)} m²" + (f" ({CHECK})" if n_unc else ""),
              help=area_text(lpo))
    m3.metric("Total holes", lpo.total_holes)
    if n_unc:
        st.caption(area_text(lpo))

    flags = review_items(lpo)   # same merged list as the PDF box
    if flags:
        st.warning("**Needs human check**\n\n" + "\n".join(f"- **{s}**: {t}" for s, t in flags))
    else:
        st.success("No items flagged.")

    if truth:
        from glass_lpo.evaluate import format_score, score
        tp = Path(res.run_dir) / "truth.json"
        tp.write_bytes(truth.getvalue())
        st.code(format_score(score(lpo.drawing, tp)))

    d1, d2 = st.columns(2)
    d1.download_button("Download Excel", res.excel.read_bytes(), res.excel.name)
    d2.download_button("Download PDF", res.pdf.read_bytes(), res.pdf.name)
