"""Simple web UI:  streamlit run app.py"""
import html
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from glass_lpo.cli import area_text
from glass_lpo.export_excel import fmt_num, review_items
from glass_lpo.pipeline import run
from glass_lpo.schema import CHECK

STEP_LABELS = {
    "enhance": "Enhancing image…",
    "extract": "Reading drawing…",
    "validate": "Checking values…",
    "export": "Writing Excel and PDF…",
}

st.set_page_config(page_title="Glass Drawing → LPO", page_icon="🪟", layout="wide")
st.title("Glass drawing → LPO draft")
st.caption("Upload a phone photo of a glass drawing. Every value that can't be read clearly is marked CHECK.")

SAMPLES = sorted(Path(__file__).parent.glob("samples/*/*.[jp][pn]g"))

# Drop-zone look for the photo uploader. Colours are translucent so it works in
# both light and dark mode.
FOLDER_SVG = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 52'>"
              "<path d='M4 4h20l6 6h28a4 4 0 0 1 4 4v34a4 4 0 0 1-4 4H4a4 4 0 0 1-4-4V8a4 4 0 0 1 4-4z' fill='%234F5BFF'/>"
              "<path d='M10 18h50l-6 30H2z' fill='%23111827' opacity='.9'/>"
              "<path d='M14 20h50l-6 30H6z' fill='%235B67FF'/>"
              "<rect x='40' y='30' width='16' height='3' rx='1.5' fill='%23283094'/>"
              "<rect x='38' y='37' width='16' height='3' rx='1.5' fill='%23283094'/></svg>")
st.markdown(f"""
<style>
.st-key-photo_drop [data-testid="stFileUploaderDropzone"] {{
  flex-direction: column; justify-content: center; align-items: center; text-align: center;
  gap: .85rem; padding: 2.25rem 1.5rem; min-height: 270px;
  border: 1.5px dashed rgba(128,140,170,.55); border-radius: 1rem;
  background: rgba(79,91,255,.06);
}}
.st-key-photo_drop [data-testid="stFileUploaderDropzone"]:hover {{
  border-color: #4F5BFF; background: rgba(79,91,255,.10);
}}
.st-key-photo_drop [data-testid="stFileUploaderDropzone"]::before {{
  content: ""; order: -2; width: 64px; height: 52px;
  background: url("{FOLDER_SVG}") center / contain no-repeat;
}}
.st-key-photo_drop [data-testid="stFileUploaderDropzoneInstructions"] {{
  order: -1; flex-direction: column; align-items: center; margin: 0;
}}
.st-key-photo_drop [data-testid="stFileUploaderDropzoneInstructions"]::before {{
  content: "Drag and drop your drawing photo here";
  display: block; font-size: 1.1rem; font-weight: 600; margin-bottom: .2rem;
}}
.st-key-photo_drop [data-testid="stFileUploaderDropzone"]::after {{
  content: "OR"; order: 1; width: 210px; text-align: center;
  color: rgba(128,140,170,.95); letter-spacing: .05em;
  background:
    linear-gradient(rgba(128,140,170,.6), rgba(128,140,170,.6)) left center / 70px 1.5px no-repeat,
    linear-gradient(rgba(128,140,170,.6), rgba(128,140,170,.6)) right center / 70px 1.5px no-repeat;
}}
.st-key-photo_drop [data-testid="stFileUploaderDropzone"] button {{
  font-size: 0; background: #4F5BFF; border: none; border-radius: .5rem;
  padding: .65rem 1.6rem;
}}
.st-key-photo_drop [data-testid="stFileUploaderDropzone"] > :has(button),
.st-key-photo_drop [data-testid="stFileUploaderDropzone"] > button {{ order: 2; }}
.st-key-photo_drop [data-testid="stFileUploaderDropzone"] button > * {{ display: none; }}
.st-key-photo_drop [data-testid="stFileUploaderDropzone"] button:hover {{ background: #3D48E0; }}
.st-key-photo_drop [data-testid="stFileUploaderDropzone"] button::after {{
  content: "Choose File"; font-size: 1rem; font-weight: 600; color: #fff;
}}
/* checks card: amber accent, works in both themes */
.st-key-checks {{border:1px solid rgba(245,158,11,.4); background:rgba(245,158,11,.06);
                border-radius:.9rem; padding:1.15rem 1.4rem 1.2rem; margin:.5rem 0 1.5rem;}}
.checks-head {{font-weight:700; font-size:1.05rem; display:flex; flex-wrap:wrap; align-items:baseline;
              gap:.25rem .75rem; margin-bottom:.4rem;}}
.checks-sub {{font-weight:400; font-size:.85rem; opacity:.6;}}
.ck-label {{font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
           opacity:.6; margin:.2rem 0 .35rem;}}
.st-key-checks ul.ck-list {{list-style:none; margin:0 !important; padding:0 !important;}}
.st-key-checks ul.ck-list li {{margin:0;}}
.ck-list li {{position:relative; padding:.5rem 0 .5rem 1.15rem; font-size:.9rem; line-height:1.45;
             border-top:1px solid rgba(128,128,128,.16);}}
.ck-list li:first-child {{border-top:none;}}
.ck-list li::before {{content:""; position:absolute; left:.1rem; top:1.05rem; width:.42rem; height:.42rem;
                     border-radius:50%; background:#F59E0B;}}
.ck-tag {{display:inline-block; margin-left:.5rem; padding:.05rem .5rem; border-radius:999px; font-size:.74rem;
         font-weight:600; background:rgba(245,158,11,.16); border:1px solid rgba(245,158,11,.35);
         vertical-align:.08em; white-space:nowrap;}}
.ck-anim {{animation:ck-fade .25s ease-out;}}
/* panel picker: compact pills, active one pops + glows */
.st-key-checks [data-testid='stButtonGroup'] {{margin:.1rem 0 .3rem;}}
.st-key-checks button[data-variant='segmented_control'] {{font-weight:600; min-width:4.2rem;
  transition:background .2s, color .2s, transform .2s;}}
.st-key-checks button[data-variant='segmented_control'][aria-checked='true'] {{
  background:#4F5BFF !important; border-color:#4F5BFF !important; color:#fff !important;
  animation:ck-pop .3s ease-out, ck-glow 2.2s ease-in-out .3s infinite; position:relative; z-index:1;}}
.st-key-checks button[data-variant='segmented_control'][aria-checked='true'] * {{color:#fff !important;}}
@keyframes ck-pop {{0% {{transform:scale(.92);}} 60% {{transform:scale(1.05);}} 100% {{transform:scale(1);}}}}
@keyframes ck-glow {{0%,100% {{box-shadow:0 0 0 0 rgba(79,91,255,0);}} 50% {{box-shadow:0 0 0 4px rgba(79,91,255,.28);}}}}
@keyframes ck-fade {{from {{opacity:0; transform:translateY(-4px);}} to {{opacity:1; transform:none;}}}}
/* result photos side by side, capped so they don't dominate the page */
.st-key-photos img {{max-height:360px; object-fit:contain;}}
/* small, fixed-height preview of the chosen photo (no page scroll) */
.st-key-preview img {{max-height:min(300px, 34vh); object-fit:contain;}}
/* step indicator: centred, circle above label, connecting lines between circles */
.steps {{display:flex; justify-content:center; align-items:flex-start; flex-wrap:nowrap;
        max-width:640px; margin:.75rem auto 2rem;}}
.steps .s {{display:flex; flex-direction:column; align-items:center; gap:.45rem; flex:0 0 8rem;
           text-align:center; font-size:.92rem; font-weight:600; line-height:1.2; opacity:.5;}}
.steps .n {{width:2.1rem; height:2.1rem; border-radius:50%; display:grid; place-items:center;
           border:2px solid rgba(128,140,170,.7); font-size:.95rem;}}
.steps .s.on, .steps .s.done {{opacity:1;}}
.steps .s.on .n {{background:#4F5BFF; border-color:#4F5BFF; color:#fff;
                 box-shadow:0 0 0 4px rgba(79,91,255,.2);}}
.steps .s.done .n {{background:rgba(79,91,255,.15); border-color:#4F5BFF; color:#4F5BFF;}}
.steps .bar {{flex:1 1 60px; height:2px; margin:1.05rem -2.7rem 0; border-radius:2px;
             background:rgba(128,140,170,.45);}}
.steps .bar.done {{background:#4F5BFF;}}
@media (max-width: 640px) {{
  .steps {{margin:.25rem auto 1.4rem;}}
  .steps .s {{flex-basis:5.6rem; font-size:.74rem; gap:.35rem;}}
  .steps .n {{width:1.8rem; height:1.8rem; font-size:.82rem;}}
  .steps .bar {{flex-basis:24px; margin:.9rem -1.75rem 0;}}
}}
</style>
""", unsafe_allow_html=True)



def show_steps(active: int) -> None:
    """Progress indicator: 0 = upload, 1 = read & check, 2 = review & download."""
    names = ["Upload photo", "Read & check", "Review & download"]
    parts = []
    for i, name in enumerate(names):
        cls = "done" if i < active else "on" if i == active else ""
        if i:
            parts.append(f'<div class="bar {"done" if i <= active else ""}"></div>')
        num = "✓" if i < active else str(i + 1)
        parts.append(f'<div class="s {cls}"><span class="n">{num}</span>{name}</div>')
    stepper.markdown(f'<div class="steps">{"".join(parts)}</div>', unsafe_allow_html=True)


def start_over() -> None:
    for k in ("result", "score"):
        st.session_state.pop(k, None)
    st.session_state["upload_n"] = st.session_state.get("upload_n", 0) + 1  # clears the uploader


stepper = st.empty()
res = st.session_state.get("result")

# ---------------------------------------------------------------- step 1 + 2
if res is None:
    n = st.session_state.get("upload_n", 0)
    left, right = st.columns([3, 2], gap="large", vertical_alignment="center")
    with left:
        st.markdown("**Drawing photo**")
        with st.container(border=True, key="photo_drop"):
            up = st.file_uploader("Drawing photo", type=["jpg", "jpeg", "png"],
                                  label_visibility="collapsed", key=f"photo_{n}")
        sample = None
        if not up and SAMPLES:
            sample = st.selectbox("…or try a sample drawing", SAMPLES, index=None, key=f"sample_{n}",
                                  placeholder="Choose a sample", format_func=lambda p: p.stem)

    ready = bool(up or sample)
    with right:
        if ready:
            st.markdown("**Selected photo**")
            with st.container(key="preview"):
                st.image(up if up else str(sample), width="stretch")
        go = st.button("Generate LPO", type="primary", disabled=not ready, width="stretch")
        st.caption("Takes about 40 seconds." if ready else "Upload a photo (or pick a sample) to start.")
        with st.expander("Optional: score against ground truth"):
            truth = st.file_uploader("Ground-truth JSON", type=["json"],
                                     label_visibility="collapsed", key=f"truth_{n}")
            auto_truth = None
            if not truth and sample:
                p = sample.with_name(f"{sample.stem}.truth.json")
                auto_truth = p if p.is_file() else None
                if auto_truth:
                    st.caption(f"Using `{auto_truth.name}` for this sample.")

    show_steps(1 if go else 0)
    if not go:
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        if up:
            src = Path(tmp) / up.name
            src.write_bytes(up.getvalue())
        else:
            src = sample
        with st.status("Working on the drawing…", expanded=True) as status:
            try:
                res = run(src, "outputs", on_step=lambda s: st.write(STEP_LABELS.get(s, s)))
            except Exception as exc:
                status.update(label="Could not produce an LPO", state="error")
                st.error(str(exc))
                show_steps(0)
                st.stop()
            status.update(label=f"Done · {res.lpo.lpo_no}", state="complete")
    score_txt = None
    if truth or auto_truth:
        from glass_lpo.evaluate import format_score, score
        tp = auto_truth
        if truth:
            tp = Path(res.run_dir) / "truth.json"
            tp.write_bytes(truth.getvalue())
        score_txt = format_score(score(res.lpo.drawing, tp))
    # Kept across reruns so clicking a download doesn't wipe the page.
    st.session_state["result"] = res
    st.session_state["score"] = score_txt
    st.rerun()

# ---------------------------------------------------------------- step 3
show_steps(2)
lpo = res.lpo

h1, h2 = st.columns([3, 1], vertical_alignment="center")
h1.subheader(f"{lpo.lpo_no} · {lpo.date}")
h1.caption(f"From {lpo.source_image} · read by {lpo.drawing.model_used or '-'} · "
           f"units written: {lpo.drawing.units_written}")
h2.button("New drawing", icon=":material/add:", on_click=start_over, width="stretch")

d1, d2, _ = st.columns([1, 1, 2])
d1.download_button("Download PDF", res.pdf.read_bytes(), res.pdf.name, "application/pdf",
                   type="primary", width="stretch", on_click="ignore", icon=":material/download:")
d2.download_button("Download Excel", res.excel.read_bytes(), res.excel.name,
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   width="stretch", on_click="ignore", icon=":material/table_view:")

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
    drawing_items = [t for scope, t in flags if scope == "Drawing"]
    per_panel: dict[str, list[str]] = {}
    for scope, t in flags:
        if scope != "Drawing":
            per_panel.setdefault(scope, []).append(t)
    # The same warning on several panels is shown once, under "Whole drawing".
    seen: dict[str, list[str]] = {}
    for pid, texts in per_panel.items():
        for t in texts:
            if pid not in seen.setdefault(t, []):
                seen[t].append(pid)
    all_ids = [pp.panel_id for pp in lpo.drawing.panels]
    for t, pids in seen.items():
        if len(pids) > 1:
            where = "all panels" if set(pids) == set(all_ids) else ", ".join(pids)
            drawing_items.append(f"{t} ({where})")
            for pid in pids:
                per_panel[pid] = [x for x in per_panel[pid] if x != t]
    per_panel = {k: v for k, v in per_panel.items() if v}

    n_items = len(drawing_items) + sum(len(v) for v in per_panel.values())

    def item_list(items: list[str], anim: bool = False) -> str:
        """Rows with an amber dot; a trailing '(all panels)' / '(P1, P3)' becomes a small tag."""
        rows = []
        for t in items:
            m = re.match(r"^(.*\S)\s+\(([^()]*)\)$", t)
            tag = ""
            if m and (m.group(2) == "all panels"
                      or all(x.strip() in all_ids for x in m.group(2).split(","))):
                t, tag = m.group(1), f'<span class="ck-tag">{html.escape(m.group(2))}</span>'
            rows.append(f"<li>{html.escape(t)}{tag}</li>")
        return f'<ul class="ck-list{" ck-anim" if anim else ""}">{"".join(rows)}</ul>'

    with st.container(key="checks"):
        n_pan = len(per_panel)
        sub = f"{len(drawing_items)} on the whole drawing" + (
            f" · {n_items - len(drawing_items)} on {n_pan} panel{'s' if n_pan != 1 else ''}" if n_pan else "")
        st.markdown(f'<div class="checks-head">⚠ Check {n_items} item{"s" if n_items != 1 else ""} '
                    f'before cutting<span class="checks-sub">{sub}</span></div>', unsafe_allow_html=True)
        if drawing_items and per_panel:
            c_all, c_pan = st.columns([3, 2], gap="large")
        else:
            c_all = c_pan = st.container()
        if drawing_items:
            c_all.markdown('<div class="ck-label">Whole drawing</div>' + item_list(drawing_items),
                           unsafe_allow_html=True)
        if per_panel:
            with c_pan:
                if len(per_panel) == 1:
                    pick = next(iter(per_panel))
                    st.markdown(f'<div class="ck-label">Panel {html.escape(pick)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ck-label">By panel</div>', unsafe_allow_html=True)
                    pick = st.segmented_control(
                        "Panel", list(per_panel), default=next(iter(per_panel)),
                        format_func=lambda k: f"{k} · {len(per_panel[k])}",
                        key=f"panel_pick_{lpo.lpo_no}", label_visibility="collapsed")
                if pick:
                    st.markdown(item_list(per_panel[pick], anim=True), unsafe_allow_html=True)
else:
    st.success("Nothing flagged - still confirm before cutting.")


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
if rows:
    def highlight(v):
        # Amber on any CHECK value; translucent so it reads in light and dark mode.
        if CHECK in str(v):
            return "background-color: rgba(245,158,11,.25); font-weight: 600"
        if v == "Low":
            return "color: #E5484D; font-weight: 600"
        return ""

    st.dataframe(pd.DataFrame(rows).style.map(highlight), width="stretch", hide_index=True)
    if any(CHECK in str(v) for r in rows for v in r.values()):
        st.caption(f"Amber = {CHECK}: not clearly read, so not guessed - confirm before cutting.")
else:
    st.info("No panels were found on this drawing.")

st.markdown("**Photo**")
with st.container(key="photos"):
    c1, c2 = st.columns(2, gap="medium")
    c1.image(str(res.original), caption="Original", width="stretch")
    c2.image(str(res.enhanced), caption="Enhanced (the model reads both)", width="stretch")

if st.session_state.get("score"):
    st.subheader("Accuracy vs ground truth")
    st.code(st.session_state["score"])
