# Glass Drawing → LPO (demo)

A phone photo of a glass drawing goes in, and a **clean, checkable LPO draft** comes out: an Excel file ready for ERP upload and a one-page PDF.

The system never guesses silently. Any value it cannot read clearly is shown as **`CHECK`**, and every flagged item appears in a "Needs human check" box.

```
photo.jpg ─► enhance (OpenCV) ─► extract (Gemini vision → JSON) ─► validate (rules, CHECK, confidence) ─► Excel + PDF
```

---

## Setup (about 5 minutes)

You need Python 3.10 or newer and a free Gemini API key from <https://aistudio.google.com/apikey>.

```bash
git clone git@github.com:prey801/Paid-Trial-Task-Glass-Drawing-to-LPO-Demo.git
cd Paid-Trial-Task-Glass-Drawing-to-LPO-Demo
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # or: uv venv && uv pip install -e ".[dev]"
cp .env.example .env                         # then paste your key into GEMINI_API_KEY
```

Check the install. The tests run offline and need no API key:

```bash
.venv/bin/python -m pytest -q
```

## Run it

**Command line**

```bash
.venv/bin/glass-lpo run samples/originals/real_16_door_and_window.jpg
# optional: score against a ground-truth file
.venv/bin/glass-lpo run samples/originals/real_16_door_and_window.jpg \
    --truth samples/ground_truth/real_16_door_and_window.json
# score a whole set, 3 runs each
.venv/bin/python scripts/evaluate_set.py samples/originals samples/ground_truth --runs 3
```

**Web page** (upload a photo, see the before/after images, panels and flags, then download the Excel and PDF):

```bash
.venv/bin/streamlit run app.py
```

Each run writes to `outputs/<LPO number>/`:

| File | What it is |
|---|---|
| `original.jpg`, `enhanced.png` | The photo before and after enhancement |
| `LPO-YYYYMMDD-NNN.xlsx` | One row per panel with the 17 ERP columns, plus a **Flags** sheet |
| `LPO-YYYYMMDD-NNN.pdf` | One-page LPO draft: line items, total m², total holes, **Needs human check** box |
| `LPO-YYYYMMDD-NNN.json` | Everything that was extracted, with each value's clear/unclear status |

The LPO number is allocated only after a successful extraction, so a failed run does not use up a number.

## Results on the real client drawings

See [`samples/RESULTS.md`](samples/RESULTS.md): the four handwritten phone photos, their ground truth, the Excel and PDF outputs, and accuracy before and after fixing the unit handling they exposed.

## How it works

| Stage | File | What it does |
|---|---|---|
| Enhance | `glass_lpo/enhance.py` | Finds the paper sheet and corrects perspective, or deskews small rotations if no sheet is found. Then denoises, flattens uneven lighting (background division), applies CLAHE contrast, upscales and sharpens. Each step is logged and skipped if it fails. |
| Extract | `glass_lpo/extract.py` | Sends **both the original and the enhanced photo** to Gemini with a glass-industry prompt and a strict JSON schema. For every number, the model returns the text as written, any other possible readings, and whether it is clear, unclear or not shown. |
| Validate | `glass_lpo/validate.py` | Turns unclear values into `CHECK` and applies the business rules (below). Sets each panel's confidence to High, Medium or Low. |
| Export | `glass_lpo/export_excel.py`, `glass_lpo/export_pdf.py` | Writes the Excel file and the one-page PDF. |
| Score | `glass_lpo/evaluate.py` | Compares a run with hand-written ground truth. A confidently wrong value is counted separately from an honest `CHECK`. |
| Data model | `glass_lpo/schema.py` | Every value is stored with its *value*, whether it was read *clearly*, and whether it was *shown* on the drawing. |

### When a value becomes CHECK
- The model marks it unclear, or offers any other possible reading (e.g. a "150" that could be "160").
- The original and enhanced images disagree.
- The value was inferred rather than written. For example, a stacked panel's width taken from the overall width is shown as `900 (CHECK)` with a note explaining where it came from.

### Validation rules
- Panel widths or heights must add up to the overall size (side-by-side and stacked layouts are both detected). The tolerance is ±5 mm or 1%.
- Width and height must be 50–6000 mm, with a warning above the 3210 mm jumbo-sheet limit.
- Thickness must be a standard size: 3, 4, 5, 6, 8, 10, 12, 15, 19 or 25 mm.
- Hole diameter must be 4–100 mm and at least the glass thickness, which is required for toughened glass.
- Holes and notches must lie inside the panel, and a notch must have a size.
- **Units** (`glass_lpo/units.py`):
  - **No unit written:** assumed to be **cm** (local drawings write "90 × 210" for a 900 × 2100 mm door) and converted to mm.
  - **Safety rule:** if converting would push sizes past the 6000 mm sheet limit while the numbers as written are sensible mm (e.g. "900 × 2100"), they are kept as mm instead.
  - **cm, m, inches or feet written:** converted to mm. Fractions (`107 7/8`, `3⅜`) and feet-and-inches (`4'6"`) are read from the text.
  - **Mixed units** ("2.95m" beside "75cm"): the unit written beside each number is used for that number. A number with no unit on a mixed drawing is marked `CHECK`.
  - **Unknown unit:** not converted, and every size is marked `CHECK`.
  - **Thickness:** treated as mm, except a fraction or inch thickness (e.g. "3/8"), which is converted and marked `CHECK`.
  - Any assumption about units is flagged.
- Glass type must be a recognised type, so a drawing title is not mistaken for one.
- Unwritten quantities are assumed to be 1, and that assumption is flagged.
- Panel IDs must not repeat, and the drawing must contain at least one panel.

### Confidence
- **High:** nothing to check.
- **Medium:** only minor warnings, e.g. thickness not stated, quantity assumed to be 1, or hole positions not dimensioned.
- **Low:** at least one `CHECK` value or a broken rule.

Any panel below High has **Needs check = Y**.

## Model and cost

- **Model:** Google Gemini Flash through the `google-genai` SDK. `GEMINI_MODEL` in `.env` is a comma-separated fallback list:
  `gemini-3.5-flash,gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash-lite`.
  If a model is busy (503) it is retried once and the next model is tried. If it is out of quota (429), the next model is tried straight away. The whole extraction gives up after about 150 s, and the model that was used is printed on the LPO.
- **Tokens per drawing:** about 3.3k input (two images and the prompt), about 1.3k output and about 3.5k "thinking". That is one call per drawing, taking 30–50 s.
- **Cost per drawing:** **KES 0** on Gemini's free tier, which is enough for this demo. On paid Flash pricing it is roughly **KES 1.5–3 per drawing** (about USD 0.01–0.02 at KES 129 = USD 1). Thinking tokens make up most of that. <!-- TODO: confirm against the current Gemini price sheet -->
- **Other tools:** OpenCV, NumPy, Pydantic, openpyxl, ReportLab, Streamlit and pytest.

## Reading the outputs

- **Amber cells** in the Excel and the PDF are `CHECK` values. Never cut from them without confirming.
- **Flags sheet** (Excel) and **Needs human check** box (PDF) list every flagged item, plus the drawing-level notes.
- **Hole count** in the Excel is *per piece*. **Area** is the *line total* (W × H × Qty). The PDF's total holes takes quantity into account.
- Resolve rows with **Needs check = Y** before importing into the ERP. Numeric columns contain the text `CHECK` until then.

## Test drawings and results

**Real client drawings** (handwritten, phone photos). Full results: [`samples/RESULTS.md`](samples/RESULTS.md). Ground truth was committed before any run (commit `9b13916`).

| # | Drawing | Photo | Enhanced | Ground truth | LPO |
|---|---|---|---|---|---|
| 1 | Door frame + sliding window, m and cm | [jpg](samples/originals/real_16_door_and_window.jpg) | [png](samples/enhanced/real_16_door_and_window.png) | [json](samples/ground_truth/real_16_door_and_window.json) | [xlsx](samples/outputs/real_16_door_and_window/LPO-20261125-013.xlsx) / [pdf](samples/outputs/real_16_door_and_window/LPO-20261125-013.pdf) |
| 2 | 2 windows + 1 door, feet | [jpg](samples/originals/real_17_windows_and_door.jpg) | [png](samples/enhanced/real_17_windows_and_door.png) | [json](samples/ground_truth/real_17_windows_and_door.json) | [xlsx](samples/outputs/real_17_windows_and_door/LPO-20261125-014.xlsx) / [pdf](samples/outputs/real_17_windows_and_door/LPO-20261125-014.pdf) |
| 3 | Oak double door, 14 mm units, mm | [jpg](samples/originals/real_18_oak_double_door.jpg) | [png](samples/enhanced/real_18_oak_double_door.png) | [json](samples/ground_truth/real_18_oak_double_door.json) | [xlsx](samples/outputs/real_18_oak_double_door/LPO-20261125-015.xlsx) / [pdf](samples/outputs/real_18_oak_double_door/LPO-20261125-015.pdf) |
| 4 | Three-panel screen, inches with fractions | [jpg](samples/originals/real_sd42_three_panel.jpg) | [png](samples/enhanced/real_sd42_three_panel.png) | [json](samples/ground_truth/real_sd42_three_panel.json) | [xlsx](samples/outputs/real_sd42_three_panel/LPO-20261125-016.xlsx) / [pdf](samples/outputs/real_sd42_three_panel/LPO-20261125-016.pdf) |

**Accuracy (48 fields):** 34 correct (71%), 4 CHECK, 5 missing, 5 wrong. No size or dimension was stated wrong with confidence; the 5 wrong fields are glass-type wording, a quantity split and a misread cut-out (details in `samples/RESULTS.md`). The first run, before the unit fixes, scored 56% with 11 wrong; it is kept in `samples/outputs/run1_before_fix/`.

**Synthetic set** (`samples/synthetic/`): three computer-generated drawings made by `scripts/make_synthetic.py`, with exact ground truth, used as a repeatable regression set.

## Known limitations

- **Real photos:** tested on the four handwritten client photos plus the synthetic set. More fabricator handwriting will throw up cases these didn't cover.
- **Self-reported confidence:** the model's own "clear" label is not fully reliable. On a distorted synthetic drawing it once read "150" as "160" with confidence. Sending both images and asking for alternative readings fixed that case, but a confidently wrong reading is still possible. Hence the human check before cutting.
- **Rectangles only:** panels are assumed to be rectangles. Rakes, arches and L-shapes appear only as notes or features, and their area is the bounding rectangle.
- **One assembly per photo:** each photo is treated as one drawing with one overall size. Several separate drawings on one page will raise "sizes don't add up" flags.
- **Unit guess:** a drawing with no unit is assumed to be cm, with size-based safety rules: kept as mm if cm would exceed the 6 m sheet limit, and read as feet if cm and inches would both be too small for glass (feet sizes stay CHECK). A small drawing in mm could still be read as cm (e.g. a 400 × 300 mirror would become 4000 × 3000 mm). It is always flagged, but confirm the unit on every unlabelled drawing.
- **No pricing:** pricing is out of scope.
- **Single user:** the LPO counter is a local file and is not safe for several people running the system at once.
- **Free-tier variability:** the newest Flash models are often busy or rate-limited on the free tier, so the fallback list matters.

## Project layout

```
glass_lpo/      pipeline code (see "How it works")
app.py          Streamlit web page
scripts/        synthetic test-drawing generator
samples/        test photos, enhanced images, ground truth, outputs
tests/          offline pytest suite (no API key needed)
```

API keys live only in `.env`, which git ignores.
