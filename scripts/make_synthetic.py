"""Generate synthetic "phone photos" of hand-drawn glass drawings + ground truth.

Writes to samples/synthetic/:
  synthetic_simple.jpg     one 1200 x 800 panel with two 25 mm holes
  synthetic_realistic.jpg  a 900 x 2100 door split into P1..P5, holes + a notch
  synthetic_hard.jpg       the realistic door, wobblier, perspective-warped,
                           dim/uneven lighting, noise, blur, on a dark table
and <name>.truth.json for each. Drawing and truth come from the same data.

Usage:  python scripts/make_synthetic.py [--out samples/synthetic] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

INK = (70, 40, 25)  # dark blue-black ballpoint (BGR)
PAPER = (235, 238, 240)


# --------------------------------------------------------------------------- #
# drawing data (single source of truth)
# --------------------------------------------------------------------------- #
@dataclass
class Hole:
    x_mm: float  # centre, from panel left
    y_mm: float  # centre, from panel top
    d_mm: float = 25


@dataclass
class Notch:
    corner: str  # "br" | "bl" | "tr" | "tl"
    w_mm: float
    h_mm: float


@dataclass
class Panel:
    panel_id: str
    x_mm: float  # offset inside the overall outline
    y_mm: float
    w_mm: float
    h_mm: float
    holes: list[Hole] = field(default_factory=list)
    notches: list[Notch] = field(default_factory=list)
    quantity: int = 1


@dataclass
class Drawing:
    name: str
    overall_w: float
    overall_h: float
    panels: list[Panel]
    note: str = ""

    def truth(self) -> dict:
        return {
            "overall_width_mm": self.overall_w,
            "overall_height_mm": self.overall_h,
            "panels": [
                {
                    "panel_id": p.panel_id,
                    "width_mm": p.w_mm,
                    "height_mm": p.h_mm,
                    "quantity": p.quantity,
                    "thickness_mm": None,
                    "glass_type": None,
                    "hole_count": len(p.holes),
                    "hole_diameters_mm": [h.d_mm for h in p.holes],
                    "notches_offcuts": len(p.notches),
                    "notch_sizes": [f"{n.w_mm:g}x{n.h_mm:g}" for n in p.notches],
                }
                for p in self.panels
            ],
        }


def simple_drawing() -> Drawing:
    return Drawing(
        "synthetic_simple", 1200, 800,
        [Panel("P1", 0, 0, 1200, 800, holes=[Hole(150, 400), Hole(1050, 400)])],
    )


def realistic_drawing(name: str = "synthetic_realistic") -> Drawing:
    W = 900
    heights = [400, 450, 350, 500, 400]  # sums to 2100
    holes = {
        "P1": [Hole(100, 100), Hole(800, 100)],
        "P2": [],
        "P3": [Hole(100, 175), Hole(100, 275), Hole(800, 175)],
        "P4": [Hole(450, 100)],
        "P5": [Hole(100, 200), Hole(800, 150)],
    }
    notches = {"P5": [Notch("br", 150, 100)]}
    panels, y = [], 0
    for i, h in enumerate(heights, start=1):
        pid = f"P{i}"
        panels.append(Panel(pid, 0, y, W, h, holes=holes[pid], notches=notches.get(pid, [])))
        y += h
    return Drawing(name, W, sum(heights), panels)


# --------------------------------------------------------------------------- #
# handwriting-ish primitives
# --------------------------------------------------------------------------- #
class Pen:
    def __init__(self, img: np.ndarray, rng: np.random.Generator, wobble: float = 1.0,
                 thickness: int = 2):
        self.img, self.rng, self.wobble, self.t = img, rng, wobble, thickness

    def line(self, p0, p1, thickness=None):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        length = np.linalg.norm(p1 - p0)
        n = max(2, int(length / 12))
        t = np.linspace(0, 1, n)
        normal = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / (length + 1e-9)
        # smooth low-frequency wobble + slight overshoot at the ends
        phase = self.rng.uniform(0, 2 * np.pi, 2)
        amp = self.wobble * self.rng.uniform(1.0, 3.0)
        off = amp * (np.sin(2 * np.pi * t * self.rng.uniform(0.6, 1.6) + phase[0])
                     + 0.5 * np.sin(2 * np.pi * t * 3.1 + phase[1]))
        off += self.rng.normal(0, 0.35 * self.wobble, n)
        ext = self.rng.uniform(-2, 5) * self.wobble
        d = (p1 - p0) / (length + 1e-9)
        a, b = p0 - d * ext, p1 + d * self.rng.uniform(-2, 5) * self.wobble
        pts = a + np.outer(t, b - a) + np.outer(off, normal)
        cv2.polylines(self.img, [pts.astype(np.int32).reshape(-1, 1, 2)], False, INK,
                      thickness or self.t, cv2.LINE_AA)

    def circle(self, c, r):
        n = 40
        th = np.linspace(0, 2 * np.pi * 1.06, n) + self.rng.uniform(0, 2 * np.pi)
        rr = r * (1 + self.wobble * 0.05 * np.sin(2 * th + self.rng.uniform(0, 6)))
        pts = np.stack([c[0] + rr * np.cos(th), c[1] + rr * np.sin(th)], 1)
        cv2.polylines(self.img, [pts.astype(np.int32).reshape(-1, 1, 2)], False, INK, self.t,
                      cv2.LINE_AA)

    def text(self, s: str, org, scale=1.0, rot_ok=True):
        """Hand-ish text: per-char jitter, script font. 'Ø' drawn as O + slash."""
        x, y = float(org[0]), float(org[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        th = max(2, int(round(2 * scale)))
        for ch in s:
            jx, jy = self.rng.normal(0, 0.8 * self.wobble, 2)
            sc = scale * self.rng.uniform(0.93, 1.07)
            draw = "O" if ch == "Ø" else ch
            (cw, chh), _ = cv2.getTextSize(draw, font, sc, th)
            cv2.putText(self.img, draw, (int(x + jx), int(y + jy)), font, sc, INK, th, cv2.LINE_AA)
            if ch == "Ø":
                cv2.line(self.img, (int(x + jx - 2), int(y + jy + 3)),
                         (int(x + jx + cw + 2), int(y + jy - chh - 3)), INK, th, cv2.LINE_AA)
            x += cw * self.rng.uniform(0.95, 1.1) + 1
        return x

    def text_size(self, s, scale):
        (w, h), _ = cv2.getTextSize(s.replace("Ø", "O"), cv2.FONT_HERSHEY_SIMPLEX, scale,
                                    max(2, int(round(2 * scale))))
        return w * 1.05, h

    def arrow_head(self, tip, direction, size=12):
        d = np.array(direction, float)
        d /= np.linalg.norm(d) + 1e-9
        n = np.array([-d[1], d[0]])
        for sgn in (1, -1):
            self.line(tip, np.array(tip) - d * size + sgn * n * size * 0.45, thickness=2)

    def dim_h(self, x0, x1, y, label, scale=1.0, ext_from=None):
        """Horizontal dimension line with arrows and label above."""
        if ext_from is not None:
            self.line((x0, ext_from), (x0, y - 8), thickness=1)
            self.line((x1, ext_from), (x1, y - 8), thickness=1)
        self.line((x0, y), (x1, y), thickness=1)
        self.arrow_head((x0, y), (-1, 0))
        self.arrow_head((x1, y), (1, 0))
        tw, _ = self.text_size(label, scale)
        self.text(label, ((x0 + x1) / 2 - tw / 2, y - 10), scale)

    def dim_v(self, x, y0, y1, label, scale=1.0, ext_from=None, left=False):
        """Vertical dimension line, label written horizontally beside it."""
        if ext_from is not None:
            self.line((ext_from, y0), (x + (8 if left else -8), y0), thickness=1)
            self.line((ext_from, y1), (x + (8 if left else -8), y1), thickness=1)
        self.line((x, y0), (x, y1), thickness=1)
        self.arrow_head((x, y0), (0, -1))
        self.arrow_head((x, y1), (0, 1))
        tw, th = self.text_size(label, scale)
        tx = x - tw - 10 if left else x + 10
        self.text(label, (tx, (y0 + y1) / 2 + th / 2), scale)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render(drawing: Drawing, rng: np.random.Generator, wobble: float) -> np.ndarray:
    """Render the drawing onto a clean portrait/landscape A4-ish sheet."""
    landscape = drawing.overall_w > drawing.overall_h
    sheet_w, sheet_h = (2400, 1700) if landscape else (1700, 2400)
    img = np.full((sheet_h, sheet_w, 3), PAPER, np.uint8)
    pen = Pen(img, rng, wobble)

    margin_l, margin_r, margin_t, margin_b = 330, 380, 300, 260
    avail_w, avail_h = sheet_w - margin_l - margin_r, sheet_h - margin_t - margin_b
    s = min(avail_w / drawing.overall_w, avail_h / drawing.overall_h)
    ox = margin_l + (avail_w - drawing.overall_w * s) / 2
    oy = margin_t + (avail_h - drawing.overall_h * s) / 2
    X = lambda mm: ox + mm * s  # noqa: E731
    Y = lambda mm: oy + mm * s  # noqa: E731
    txt = 1.15

    pen.text(drawing.name.replace("synthetic_", "").upper() + " - GLASS", (80, 110), 1.3)

    for p in drawing.panels:
        x0, y0, x1, y1 = X(p.x_mm), Y(p.y_mm), X(p.x_mm + p.w_mm), Y(p.y_mm + p.h_mm)
        notch = {n.corner: n for n in p.notches}
        # outline (with a rectangular notch cut from a corner if present)
        if "br" in notch:
            n = notch["br"]
            nx, ny = X(p.x_mm + p.w_mm - n.w_mm), Y(p.y_mm + p.h_mm - n.h_mm)
            pen.line((x0, y0), (x1, y0)); pen.line((x1, y0), (x1, ny))
            pen.line((x1, ny), (nx, ny)); pen.line((nx, ny), (nx, y1))
            pen.line((nx, y1), (x0, y1)); pen.line((x0, y1), (x0, y0))
            pen.text("NOTCH", (nx - 150, ny - 12), 0.75)
            pen.text(f"{n.w_mm:g}x{n.h_mm:g}", (nx - 150, ny + 22), 0.75)
        else:
            pen.line((x0, y0), (x1, y0)); pen.line((x1, y0), (x1, y1))
            pen.line((x1, y1), (x0, y1)); pen.line((x0, y1), (x0, y0))
        # holes
        for h in p.holes:
            cx, cy = X(p.x_mm + h.x_mm), Y(p.y_mm + h.y_mm)
            pen.circle((cx, cy), max(9, h.d_mm * s / 2 * 1.6))
            pen.text(f"Ø{h.d_mm:g}", (cx + 18, cy - 14), 0.8)
        # label
        tw, th = pen.text_size(p.panel_id, 1.5)
        pen.text(p.panel_id, ((x0 + x1) / 2 - tw / 2, (y0 + y1) / 2 + th / 2), 1.5)
        # per-panel height dimension on the right
        if len(drawing.panels) > 1:
            pen.dim_v(X(drawing.overall_w) + 60, y0, y1, f"{p.h_mm:g}", txt * 0.9,
                      ext_from=x1 + 6)

    # overall width on top, overall height on the left
    pen.dim_h(X(0), X(drawing.overall_w), Y(0) - 70, f"{drawing.overall_w:g}", txt,
              ext_from=Y(0) - 6)
    pen.dim_v(X(0) - 70, Y(0), Y(drawing.overall_h), f"{drawing.overall_h:g}", txt,
              ext_from=X(0) - 6, left=True)
    if len(drawing.panels) == 1:
        pen.text("2 HOLES Ø25", (X(0), Y(drawing.overall_h) + 80), 1.0)
    else:
        pen.text("DOOR 900 x 2100  -  ALL HOLES Ø25", (X(0) - 200, Y(drawing.overall_h) + 110), 1.0)
    return img


def photo_light(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Mild phone-photo look: slight gradient, noise, soft blur."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 1 - strength * (0.6 * xx / w + 0.4 * yy / h)
    out = img.astype(np.float32) * grad[..., None]
    out += rng.normal(0, 3 + 10 * strength, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(out, (0, 0), 0.6 + strength)


def camera_quad(sw, sh, W, H, tilt_deg, yaw_deg, roll_deg, fill=0.75, f_scale=0.85):
    """Image corners of an sw x sh sheet seen by a pinhole camera (phone-like)."""
    t, y, r = np.radians([tilt_deg, yaw_deg, roll_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    f = f_scale * max(W, H)
    pts = np.array([[-sw / 2, -sh / 2, 0], [sw / 2, -sh / 2, 0],
                    [sw / 2, sh / 2, 0], [-sw / 2, sh / 2, 0]], float)
    Z = f * max(sw, sh) / (fill * max(W, H))
    cam = pts @ (Rz @ Ry @ Rx).T + np.array([0, 0, Z])
    return np.float32(np.stack([f * cam[:, 0] / cam[:, 2] + W / 2,
                                f * cam[:, 1] / cam[:, 2] + H / 2], 1))


def hard_photo(sheet: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Place the sheet on a table, perspective-warped ~20 deg, bad light, noise."""
    H, W = 4000, 3000  # 12 MP portrait phone frame
    # dark wood-ish table
    table = np.zeros((H, W, 3), np.float32)
    base = np.array([45, 70, 95], np.float32)
    grain = cv2.resize(rng.normal(0, 1, (H // 40, 8)).astype(np.float32), (W, H))
    table[:] = base
    table += grain[..., None] * 12
    table += rng.normal(0, 6, table.shape)

    sh, sw = sheet.shape[:2]
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    # Real pinhole camera: tilted ~20 deg, a little yaw and roll.
    dst = camera_quad(sw, sh, W, H, tilt_deg=20, yaw_deg=6, roll_deg=4, fill=0.78)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sheet.astype(np.float32), M, (W, H))
    mask = cv2.warpPerspective(np.ones((sh, sw), np.float32), M, (W, H))
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None]
    img = table * (1 - mask) + warped * mask

    # Uneven, dim lighting: radial falloff + a soft shadow band (phone/hand).
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((xx - 0.3 * W) / W, (yy - 0.25 * H) / H)
    light = 0.95 - 0.55 * r
    shadow = 1 - 0.35 * np.exp(-(((xx - 0.75 * W) * 0.6 + (yy - 0.7 * H)) / (0.12 * W)) ** 2)
    light = np.clip(light * shadow, 0.25, 1.0)
    img = img * light[..., None] * 0.85
    img[..., 0] *= 0.9  # warm (yellowish) indoor light
    img = cv2.GaussianBlur(img, (0, 0), 2.0)  # slight defocus
    img += rng.normal(0, 7, img.shape)  # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "samples" / "synthetic"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    jobs = [
        (simple_drawing(), lambda d: photo_light(render(d, rng, 0.8), rng, 0.15)),
        (realistic_drawing(), lambda d: photo_light(render(d, rng, 1.0), rng, 0.25)),
        (realistic_drawing("synthetic_hard"), lambda d: hard_photo(render(d, rng, 2.0), rng)),
    ]
    for drawing, make in jobs:
        img = make(drawing)
        jpg = out / f"{drawing.name}.jpg"
        cv2.imwrite(str(jpg), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        truth = out / f"{drawing.name}.truth.json"
        truth.write_text(json.dumps(drawing.truth(), indent=2) + "\n")
        print(f"wrote {jpg} ({img.shape[1]}x{img.shape[0]}) and {truth.name}")


if __name__ == "__main__":
    main()
