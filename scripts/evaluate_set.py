"""Score a whole set of drawings against their ground truth, N runs each.

    .venv/bin/python scripts/evaluate_set.py samples/originals samples/ground_truth --runs 3
    .venv/bin/python scripts/evaluate_set.py ... --second-read --out outputs/eval

Each photo `<name>.jpg|.jpeg|.png` is matched with `<truth_dir>/<name>.json`.
Prints per-drawing and total field counts (correct / CHECK / wrong / missing),
averaged over the runs with the min-max range, so one lucky or unlucky model
answer does not decide the score. Writes the same as JSON to <out>/summary.json.

Rates reported:
  correct = right and stated confidently
  safe    = correct + honest CHECK (nothing cut wrong without a human seeing it)
  wrong   = wrong but stated confidently (the dangerous case)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glass_lpo.evaluate import score  # noqa: E402
from glass_lpo.pipeline import run  # noqa: E402

KEYS = ("correct", "flagged", "wrong_confident", "missing")


def _run_one(img: Path, truth: Path, out: Path, attempts: int) -> dict | None:
    for a in range(1, attempts + 1):
        try:
            res = run(img, out)
        except Exception as exc:  # noqa: BLE001 - API flakiness: retry, then skip
            print(f"    attempt {a} failed: {str(exc).splitlines()[0][:100]}", flush=True)
            time.sleep(20)
            continue
        s = score(res.lpo.drawing, truth)
        sp = score(res.lpo.drawing, truth, by_pieces=True)
        return ({k: getattr(s, k) for k in KEYS} | {"total": s.total, "lpo": res.lpo.lpo_no,
                                                    "details": s.details}
                | {f"pieces_{k}": getattr(sp, k) for k in KEYS} | {"pieces_total": sp.total})
    return None


def _pct(n: float, d: float) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("truths")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--attempts", type=int, default=3, help="API attempts per run")
    ap.add_argument("--second-read", action="store_true")
    ap.add_argument("--out", default="outputs/eval")
    a = ap.parse_args(argv)
    if a.second_read:
        os.environ["GLASS_LPO_SECOND_READ"] = "1"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    results: dict[str, list[dict]] = {}
    for img in imgs:
        truth = Path(a.truths) / f"{img.stem}.json"
        if not truth.is_file():
            print(f"skip {img.name}: no {truth}")
            continue
        results[img.stem] = []
        for r in range(1, a.runs + 1):
            print(f"{img.stem} run {r}/{a.runs}", flush=True)
            res = _run_one(img, truth, out, a.attempts)
            if res:
                results[img.stem].append(res)

    print(f"\n{'drawing':<34}{'runs':>5}{'fields':>7}{'correct':>9}{'CHECK':>7}{'wrong':>7}{'missing':>9}")
    tot = {k: 0.0 for k in KEYS} | {"total": 0.0}
    per_run_correct: list[float] = []
    summary = {}
    for name, runs in results.items():
        if not runs:
            print(f"{name:<34}{0:>5}  (all runs failed)")
            continue
        mean = {k: statistics.mean(r[k] for r in runs) for k in KEYS + ("total",)}
        rng = {k: (min(r[k] for r in runs), max(r[k] for r in runs)) for k in KEYS}
        for k in tot:
            tot[k] += mean[k]
        summary[name] = {"runs": runs, "mean": mean, "range": rng}
        cell = lambda k: f"{mean[k]:.1f}" if rng[k][0] == rng[k][1] else f"{mean[k]:.1f}({rng[k][0]}-{rng[k][1]})"
        print(f"{name:<34}{len(runs):>5}{mean['total']:>7.0f}{cell('correct'):>9}{cell('flagged'):>7}"
              f"{cell('wrong_confident'):>7}{cell('missing'):>9}")
    # Per-run totals (run i of every drawing) give the spread of the overall score.
    n = min((len(r) for r in results.values() if r), default=0)
    for i in range(n):
        c = sum(r[i]["correct"] for r in results.values() if r)
        t = sum(r[i]["total"] for r in results.values() if r)
        per_run_correct.append(100 * c / t if t else 0)
    T = tot["total"]
    print(f"\nTOTAL (mean of runs): {T:.0f} fields   correct {_pct(tot['correct'], T)}   "
          f"safe {_pct(tot['correct'] + tot['flagged'], T)}   wrong {_pct(tot['wrong_confident'], T)}   "
          f"missing {_pct(tot['missing'], T)}")
    pc = sum(statistics.mean(r["pieces_correct"] for r in rs) for rs in results.values() if rs)
    pf = sum(statistics.mean(r["pieces_flagged"] for r in rs) for rs in results.values() if rs)
    pw = sum(statistics.mean(r["pieces_wrong_confident"] for r in rs) for rs in results.values() if rs)
    pt = sum(statistics.mean(r["pieces_total"] for r in rs) for rs in results.values() if rs)
    print(f"BY PIECES (identical lines merged): {pt:.0f} fields   correct {_pct(pc, pt)}   "
          f"safe {_pct(pc + pf, pt)}   wrong {_pct(pw, pt)}")
    if len(per_run_correct) > 1:
        print(f"correct % per run: {', '.join(f'{x:.0f}%' for x in per_run_correct)}")
    (out / "summary.json").write_text(json.dumps(
        {"drawings": summary, "total": tot, "correct_pct_per_run": per_run_correct,
         "second_read": a.second_read}, indent=2))
    print(f"\nwrote {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
