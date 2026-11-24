"""Command-line entry point.

    glass-lpo run photo.jpg [--out outputs] [--truth truth.json]
    glass-lpo enhance photo.jpg enhanced.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .export_excel import fmt_num, review_items


def area_text(lpo) -> str:
    """'Total area 3.15 m²' or '... (includes 2 CHECK lines)' when it rests on unclear values."""
    n = lpo.uncertain_area_lines
    tail = f" (includes {n} CHECK line{'s' if n != 1 else ''})" if n else ""
    return f"Total area {fmt_num(lpo.total_area_m2)} m²{tail}"


def _print_summary(res) -> None:
    lpo = res.lpo
    d = lpo.drawing
    print(f"\nLPO {lpo.lpo_no}  ({lpo.date})  from {lpo.source_image}")
    print(f"Read by: {d.model_used or '-'}   Units written: {d.units_written}")
    print("Enhancement: " + "; ".join(res.enhance_steps))
    ow, oh = d.overall_width_mm.display(fmt_num), d.overall_height_mm.display(fmt_num)
    if ow or oh:
        print(f"Overall: {ow or '-'} x {oh or '-'} mm")
    print(f"{'Panel':<8}{'W':>14}{'H':>14}{'Qty':>6}{'Holes':>7}  Conf.   Check")
    for p in d.panels:
        print(f"{p.panel_id:<8}{p.width_mm.display(fmt_num, required=True):>14}"
              f"{p.height_mm.display(fmt_num, required=True):>14}"
              f"{p.quantity.display(fmt_num):>6}{len(p.holes):>7}  {p.confidence.value:<7} "
              f"{'Y' if p.needs_check else 'N'}")
    print(f"{area_text(lpo)}   Total holes: {lpo.total_holes}")
    items = review_items(lpo)   # same merged list as the PDF box
    print(f"\nNEEDS HUMAN CHECK ({len(items)}):" if items else "\nNo items flagged.")
    for scope, text in items:
        print(f"  - {scope}: {text}")
    print(f"\nOutputs in {res.run_dir}/:\n  {res.excel.name}\n  {res.pdf.name}\n  {res.enhanced.name}\n  {res.json.name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="glass-lpo", description="Glass drawing photo -> checkable LPO")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="full pipeline on one photo")
    r.add_argument("image")
    r.add_argument("--out", default="outputs")
    src = r.add_mutually_exclusive_group()
    src.add_argument("--from-original", action="store_true",
                     help="send only the original image to the model (default: original + enhanced)")
    src.add_argument("--from-enhanced", action="store_true",
                     help="send only the enhanced image to the model")
    r.add_argument("--truth", help="ground-truth JSON to score this run against")

    e = sub.add_parser("enhance", help="enhancement only")
    e.add_argument("image")
    e.add_argument("dst")

    args = ap.parse_args(argv)

    if args.cmd == "enhance":
        from .enhance import enhance_file
        res = enhance_file(args.image, args.dst)
        print("; ".join(res.steps))
        return 0

    from .pipeline import run
    try:
        extract_from = "original" if args.from_original else "enhanced" if args.from_enhanced else "both"
        res = run(args.image, args.out, extract_from=extract_from)
    except Exception as exc:  # surface a clean message for the demo
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    _print_summary(res)

    if args.truth:
        from .evaluate import score, format_score
        print("\n" + format_score(score(res.lpo.drawing, Path(args.truth))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
