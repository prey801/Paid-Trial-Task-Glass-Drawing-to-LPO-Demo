# Results on the four real client drawings

The four handwritten drawings that were sent as phone photos, run through the
pipeline and scored against ground truth that was written **before** the
first run. The ground truth was committed on its own first (commit `9b13916`),
so the git history shows it predates every output.

| # | Drawing | Original | Enhanced | Ground truth | LPO (Excel / PDF) |
|---|---|---|---|---|---|
| 1 | Door frame + sliding window, m and cm | [jpg](originals/real_16_door_and_window.jpg) | [png](enhanced/real_16_door_and_window.png) | [json](ground_truth/real_16_door_and_window.json) | [xlsx](outputs/real_16_door_and_window/LPO-20261125-013.xlsx) / [pdf](outputs/real_16_door_and_window/LPO-20261125-013.pdf) |
| 2 | 2 windows + 1 door, feet | [jpg](originals/real_17_windows_and_door.jpg) | [png](enhanced/real_17_windows_and_door.png) | [json](ground_truth/real_17_windows_and_door.json) | [xlsx](outputs/real_17_windows_and_door/LPO-20261125-014.xlsx) / [pdf](outputs/real_17_windows_and_door/LPO-20261125-014.pdf) |
| 3 | Oak double door, 14 mm units, mm | [jpg](originals/real_18_oak_double_door.jpg) | [png](enhanced/real_18_oak_double_door.png) | [json](ground_truth/real_18_oak_double_door.json) | [xlsx](outputs/real_18_oak_double_door/LPO-20261125-015.xlsx) / [pdf](outputs/real_18_oak_double_door/LPO-20261125-015.pdf) |
| 4 | Three-panel screen, inches with fractions | [jpg](originals/real_sd42_three_panel.jpg) | [png](enhanced/real_sd42_three_panel.png) | [json](ground_truth/real_sd42_three_panel.json) | [xlsx](outputs/real_sd42_three_panel/LPO-20261125-016.xlsx) / [pdf](outputs/real_sd42_three_panel/LPO-20261125-016.pdf) |

Each output folder also has the full JSON and `run.log`, the console output
including the per-field scoring.

## How the ground truth was written

Only what is **written** on the drawing counts. Where a size is not written
(e.g. the glass in the oak door, where only the timber is dimensioned), the
truth is `null`. If the system invents a number there, it scores as wrong; a
`CHECK` scores as flagged. Imperial sizes were converted at 1 ft = 304.8 mm
and 1 in = 25.4 mm. Each truth file has a `_reading` note explaining how the
drawing was read.

## Accuracy (per field)

*Correct* = right and stated with confidence. *CHECK* = shown as CHECK for a
human to confirm. *Wrong* = wrong but stated with confidence (the dangerous
case). *Missing* = on the drawing, not extracted.

### Final run (after the unit fix)

| Drawing | Fields | Correct | CHECK | Wrong | Missing |
|---|---|---|---|---|---|
| 1 Door + window | 11 | 9 (82%) | 0 | 1 | 1 |
| 2 Windows + door | 16 | 11 (69%) | 0 | 1 | 4 |
| 3 Oak double door | 10 | 8 (80%) | 0 | 2 | 0 |
| 4 Three-panel, inches | 11 | 6 (55%) | 4 | 1 | 0 |
| **Total** | **48** | **34 (71%)** | **4** | **5** | **5** |

**No size or dimension was stated wrong with confidence.** The 5 wrong fields are:

- Drawing 3, glass type ×2: read "UNIT" against the truth "14mm unit (double
  glazed)". The thickness (14 mm) was read correctly and flagged as
  non-standard. The scorer compares text exactly.
- Drawing 1, qty: the two door leaves were put on one line with qty 2 instead
  of two lines with qty 1. That is the same glass, but counted as wrong.
- Drawing 2, door glass type: read the word "glass" on the drawing as a
  glass type.
- Drawing 4, offcuts: the fitting marks at the top were read as a cut-out
  (its size is shown as CHECK).

The 5 missing fields are mostly on drawing 2, where the overwritten "4 feet"
height was not picked up. These show as CHECK on the LPO, not as a number.

### First run (before the fix), kept for comparison

The first-run outputs are in [`outputs/run1_before_fix/`](outputs/run1_before_fix/).

| Drawing | Correct | CHECK | Wrong | Missing |
|---|---|---|---|---|
| 1 | 7 / 14 | 0 | 3 | 4 |
| 2 | 9 / 15 | 2 | 0 | 4 |
| 3 | 8 / 10 | 0 | 2 | 0 |
| 4 | 4 / 11 | 1 | 6 | 0 |
| **Total** | **28 / 50 (56%)** | **3** | **11** | **8** |

## What the real drawings showed, and what was fixed

The synthetic test drawings were all in mm/cm. The real ones were not:

1. **Feet and inches were not converted.** Drawing 2 came out with a width of
   "2" (from "2 ft") and drawing 4 with "107.875" (from 107 ⅞″). They were
   flagged, but useless.
   **Fix:** inches and feet are now converted (×25.4, ×304.8) and flagged
   "confirm". Fractions such as `107 7/8`, `3/8` and `3⅜` and feet-and-inches
   such as `4'6"` are read from the text.
2. **Mixed units on one drawing** ("2.95m" beside "75cm") gave
   "unknown units". **Fix:** the unit is now read from the text written
   beside *each* number, and each value is converted on its own. A number with
   no unit on a mixed drawing is left as written and marked CHECK. If the
   model's number disagrees with its own transcription, the value becomes
   CHECK.
3. **A fraction thickness** ("3/8") came out as 0.375 mm. **Fix:** a
   fractional or inch thickness is converted (3/8″ = 9.5 mm) and always
   marked CHECK.
4. **Scoring:** panels are now matched to the truth by size before ID.
   Hand drawings rarely number their panels, so the IDs are assigned by the
   system, and matching on them alone scored drawing 1's correctly read window
   (750 × 970) as wrong.

Each fix has unit tests (`tests/test_units.py`, `tests/test_evaluate.py`);
172 tests pass. The ground truth was not changed between runs.

## Caveats

- The model's output varies a little between runs (e.g. drawing 3's glass type
  was "double glazed unit" in run 1 and "UNIT" in run 2). Both runs are kept.
- The Gemini API returned many 503 "high demand" errors and timeouts while
  these were run. The pipeline retries and falls back to other models, but on
  a bad day a photo can take several attempts.
- Three of the four photos arrived small (about 400–740 px wide). The
  pipeline reads them, but a full-resolution phone photo (like drawing 4)
  gives the model more to work with.
