"""Image enhancement for phone photos of hand-drawn glass cutting drawings.

Pipeline (every step is best-effort: on failure it is skipped and logged,
never raised):

1. Normalise input (grayscale / BGRA -> BGR, downscale huge photos for speed).
2. Perspective: find the paper sheet (largest 4-corner contour, > ~20% of the
   frame) and four-point warp it flat. If no sheet is found, deskew small
   rotations (< 15 deg) using the dominant angle of long ink lines.
3. Lighting: divide by a heavily smoothed "paper only" background (morphological
   closing removes the ink, a blur smooths it) to flatten shadows / gradients,
   then CLAHE on the L channel for local contrast.
4. Denoise: moderate non-local-means (keeps thin pen lines and small digits).
5. Upscale (INTER_CUBIC) by ``upscale``, capping the longest side at ``max_side``.
6. Sharpen with an unsharp mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Longest side of the working image before upscaling. Big enough to keep pen
# detail, small enough that denoising stays well under a few seconds.
_WORK_MAX_SIDE = 2000
# Longest side used for paper / skew detection.
_DETECT_MAX_SIDE = 800


@dataclass
class EnhanceResult:
    image: np.ndarray  # enhanced BGR image
    steps: list[str] = field(default_factory=list)  # human-readable log


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _resize_max(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img, 1.0
    out = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                     interpolation=cv2.INTER_AREA)
    return out, s


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _quad_is_reasonable(q: np.ndarray, img_area: float) -> bool:
    if len({tuple(np.round(p, 1)) for p in q}) < 4:
        return False
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)):
        return False
    area = cv2.contourArea(q)
    if area < 0.2 * img_area:
        return False
    # interior angles must be roughly rectangular (perspective allows slack)
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = abs(float(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        if cos > 0.6:  # angle outside ~53..127 deg
            return False
    return True


def _edges_have_paper_contrast(gray: np.ndarray, q: np.ndarray) -> bool:
    """True if the quad looks like a sheet edge (bright inside, darker outside).

    A pen-drawn rectangle on the paper has paper on BOTH sides of its edges, so
    this rejects panel outlines that would otherwise crop away dimensions.
    """
    h, w = gray.shape[:2]
    d = max(4.0, 0.025 * max(h, w))
    c = q.mean(axis=0)
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    good = bad = 0
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        e = b - a
        n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-9)
        if np.dot(n, ((a + b) / 2) - c) < 0:
            n = -n  # make n point outward
        ins, outs = [], []
        for t in np.linspace(0.1, 0.9, 25):
            p = a + t * e
            pi, po = p - n * d, p + n * d
            if 0 <= po[0] < w and 0 <= po[1] < h and 0 <= pi[0] < w and 0 <= pi[1] < h:
                ins.append(g[int(pi[1]), int(pi[0])])
                outs.append(g[int(po[1]), int(po[0])])
        if len(outs) < 5:
            continue  # edge hugs the image border: neither confirms nor refutes
        if float(np.median(ins)) - float(np.median(outs)) > 18:
            good += 1
        else:
            bad += 1
    # Allow one ambiguous side (e.g. a shadow or a white object next to the sheet).
    return good >= 2 and bad <= 1


def _find_paper_quad(img_bgr: np.ndarray) -> np.ndarray | None:
    """Return the paper outline (4x2, in img_bgr coords) or None."""
    small, s = _resize_max(img_bgr, _DETECT_MAX_SIDE)
    h, w = small.shape[:2]
    img_area = float(h * w)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Candidate binary maps: edges (several thresholds) and bright-paper mask.
    maps = []
    med = float(np.median(blur))
    for lo, hi in ((0.66 * med, 1.33 * med), (30, 90), (10, 50)):
        e = cv2.Canny(blur, lo, hi)
        maps.append(cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=2))
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    maps.append(otsu)

    best, best_area = None, 0.0
    for m in maps:
        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            if cv2.contourArea(c) < 0.2 * img_area:
                break
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.01, 0.02, 0.03, 0.05):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    q = _order_quad(approx)
                    a = cv2.contourArea(q)
                    # Ignore the image frame itself (nothing to correct).
                    if a > 0.97 * img_area:
                        break
                    if (a > best_area and _quad_is_reasonable(q, img_area)
                            and _edges_have_paper_contrast(gray, q)):
                        best, best_area = q, a
                    break
    if best is None:
        return None
    return best / s


def _rect_aspect(q: np.ndarray, img_shape) -> float | None:
    """Real width/height ratio of the rectangle imaged as quad ``q``.

    Zhang & He, "Whiteboard scanning and image enhancement" (2007): assumes a
    pinhole camera with square pixels and the principal point at the image
    centre (true for phone photos). Returns None when degenerate.
    """
    h, w = img_shape[:2]
    u0, v0 = w / 2.0, h / 2.0
    tl, tr, br, bl = [np.array([p[0], p[1], 1.0]) for p in q]
    m1, m2, m3, m4 = tl, tr, bl, br
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1
    if abs(n2[2]) < 1e-9 or abs(n3[2]) < 1e-9:
        # (near) affine view: ratio of edge vectors is exact
        r2 = (n2[0] ** 2 + n2[1] ** 2) / max(n3[0] ** 2 + n3[1] ** 2, 1e-9)
        return float(np.sqrt(r2)) if r2 > 0 else None
    f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0)
           + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)
           ) / (n2[2] * n3[2])
    if not np.isfinite(f2) or f2 <= 0:
        return None
    f = np.sqrt(f2)
    Ainv = np.linalg.inv(np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]]))
    B = Ainv.T @ Ainv
    r2 = (n2 @ B @ n2) / (n3 @ B @ n3)
    return float(np.sqrt(r2)) if np.isfinite(r2) and r2 > 0 else None


def _warp_quad(img: np.ndarray, q: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = q
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    w, h = max(w, 2), max(h, 2)
    r = _rect_aspect(q, img.shape)
    # Trust the camera-model ratio only if it is plausible vs the naive one.
    if r is not None and 0.5 < r / (w / h) < 2.0:
        if w / h >= r:
            h = max(2, int(round(w / r)))
        else:
            w = max(2, int(round(h * r)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    # Trim a thin margin so the paper edge / table doesn't creep in.
    m = max(1, int(0.006 * min(w, h)))
    if w > 4 * m and h > 4 * m:
        out = out[m:h - m, m:w - m]
    return out


def _estimate_skew(img_bgr: np.ndarray) -> float | None:
    """Dominant small skew angle in degrees (positive = counter-clockwise)."""
    small, _ = _resize_max(img_bgr, _DETECT_MAX_SIDE)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 15)
    if ink.mean() < 0.5:  # essentially blank
        return None
    min_len = max(20, int(0.12 * max(small.shape[:2])))
    lines = cv2.HoughLinesP(ink, 1, np.pi / 360, threshold=60,
                            minLineLength=min_len, maxLineGap=8)
    if lines is None:
        return None
    angles, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        ang = np.degrees(np.arctan2(dy, dx))
        # fold to [-45, 45) around the nearest horizontal / vertical axis
        ang = (ang + 45) % 90 - 45
        if abs(ang) < 15:
            angles.append(ang)
            weights.append(length)
    if len(angles) < 3:
        return None
    order = np.argsort(angles)
    a = np.array(angles)[order]
    wts = np.cumsum(np.array(weights)[order])
    median = float(a[np.searchsorted(wts, wts[-1] / 2)])
    return -median  # image y axis points down


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _flatten_lighting(img_bgr: np.ndarray) -> np.ndarray:
    """Divide each channel by its paper-only background estimate."""
    h, w = img_bgr.shape[:2]
    # Background is estimated on a small copy (fast, and ink is removed by
    # closing with a kernel wider than any pen stroke / character).
    small, s = _resize_max(img_bgr, 500)
    sh, sw = small.shape[:2]
    k = max(5, int(0.03 * max(sh, sw)) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, k if k <= 255 else 255)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = img_bgr.astype(np.float32) / np.maximum(bg, 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _noise_sigma(gray: np.ndarray) -> float:
    """Robust noise estimate (Immerkaer-style Laplacian MAD)."""
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.median(np.abs(lap)) * 1.4826 / np.sqrt(20.0))


def _denoise(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Non-local means on lightness only (fast, keeps thin strokes); chroma smoothed."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    h = float(np.clip(2.2 * _noise_sigma(l), 3.0, 12.0))
    l = cv2.fastNlMeansDenoising(l, None, h, 7, 21)
    a, b = cv2.medianBlur(a, 5), cv2.medianBlur(b, 5)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR), h


def _clahe_l(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    tiles = 8
    l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(tiles, tiles)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _stretch_ink(img_bgr: np.ndarray) -> np.ndarray:
    """Push faint ink darker and paper to clean white (gentle levels curve)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    lo = float(np.percentile(l, 0.5))
    # Most of a drawing is paper; put the white point a little BELOW the paper
    # level so faint paper texture / residual noise clips to clean white.
    hi = float(np.percentile(l, 60)) - 10
    if hi - lo < 20:
        return img_bgr
    lut = np.clip((np.arange(256) - lo) / (hi - lo), 0, 1) ** 1.3 * 255
    lab[:, :, 0] = cv2.LUT(l, lut.astype(np.uint8))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def enhance(img_bgr: np.ndarray, upscale: float = 2.0, max_side: int = 3000) -> EnhanceResult:
    steps: list[str] = []
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        raise ValueError("enhance() needs a non-empty image array")

    img = img_bgr
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        steps.append(f"converted {img_bgr.dtype} to uint8")
    if img.ndim == 2 or img.shape[2] != 3:
        steps.append("converted input to BGR")
    img = _to_bgr(img)

    # Work at a bounded resolution. Keep extra detail when the result will
    # not be upscaled much (so we don't throw away pixels only to re-create them).
    work_cap = max(_WORK_MAX_SIDE, int(max_side / max(upscale, 1.0)))
    work_cap = min(work_cap, max_side) if max_side >= 64 else work_cap
    img, s = _resize_max(img, work_cap)
    if s < 1.0:
        steps.append(f"downscaled input to {img.shape[1]}x{img.shape[0]} for processing")

    tiny = min(img.shape[:2]) < 32

    # 1. perspective / skew ------------------------------------------------- #
    if tiny:
        steps.append("skipped perspective/deskew: image too small")
    else:
        try:
            q = _find_paper_quad(img)
            if q is not None:
                img = _warp_quad(img, q)
                steps.append(f"perspective corrected (paper quad found) -> {img.shape[1]}x{img.shape[0]}")
            else:
                steps.append("skipped perspective: no paper outline found")
                ang = _estimate_skew(img)
                if ang is not None and 0.3 <= abs(ang) < 15:
                    img = _rotate(img, ang)
                    steps.append(f"deskewed by {ang:+.1f} deg (line angle)")
                else:
                    steps.append("skipped deskew: no significant skew detected")
        except Exception as e:  # pragma: no cover - defensive
            steps.append(f"skipped perspective/deskew: error {e!r}")

    # 2. denoise (before lighting, so shadowed noise isn't amplified first) - #
    try:
        img, h_strength = _denoise(img)
        steps.append(f"denoised (non-local means on L, h={h_strength:.1f})")
    except Exception as e:  # pragma: no cover
        steps.append(f"skipped denoise: {e!r}")

    # 3. lighting ----------------------------------------------------------- #
    try:
        if tiny:
            raise ValueError("image too small")
        img = _flatten_lighting(img)
        steps.append("flattened uneven lighting (background division)")
    except Exception as e:
        steps.append(f"skipped lighting flatten: {e}")
    try:
        img = _clahe_l(img)
        img = _stretch_ink(img)
        steps.append("contrast: CLAHE on L channel + ink/paper levels")
    except Exception as e:  # pragma: no cover
        steps.append(f"skipped contrast: {e!r}")

    # 4. upscale ------------------------------------------------------------ #
    try:
        h, w = img.shape[:2]
        f = min(float(upscale), max_side / max(h, w)) if upscale > 0 else 1.0
        if f > 1.01 or f < 0.99:
            img = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                             interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
            steps.append(f"resized x{f:.2f} -> {img.shape[1]}x{img.shape[0]}")
        else:
            steps.append("skipped upscale: already at max_side")
    except Exception as e:  # pragma: no cover
        steps.append(f"skipped upscale: {e!r}")

    # 5. sharpen ------------------------------------------------------------ #
    try:
        sigma = 1.0 + 0.5 * max(0.0, min(upscale, 3.0) - 1.0)
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        img = cv2.addWeighted(img, 1.6, blur, -0.6, 0)
        steps.append(f"sharpened (unsharp mask, sigma={sigma:.1f})")
    except Exception as e:  # pragma: no cover
        steps.append(f"skipped sharpen: {e!r}")

    return EnhanceResult(image=img, steps=steps)


def enhance_file(src: str | Path, dst: str | Path, **kw) -> EnhanceResult:
    """Read a JPG/PNG, enhance it, write to ``dst`` (format from extension)."""
    src, dst = Path(src), Path(dst)
    data = np.fromfile(str(src), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {src}")
    res = enhance(img, **kw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ext = dst.suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, res.image, params)
    if not ok:
        raise ValueError(f"could not encode image as {ext}")
    buf.tofile(str(dst))
    return res
