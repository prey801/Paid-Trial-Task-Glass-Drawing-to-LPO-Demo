import cv2
import numpy as np
import pytest

from glass_lpo.enhance import EnhanceResult, enhance, enhance_file


def _drawing_sheet(w=600, h=850):
    """White sheet with some pen-like lines and digits."""
    sheet = np.full((h, w, 3), 240, np.uint8)
    cv2.rectangle(sheet, (80, 120), (w - 80, h - 120), (30, 30, 30), 3)
    cv2.line(sheet, (80, h // 2), (w - 80, h // 2), (30, 30, 30), 2)
    cv2.putText(sheet, "1200", (w // 2 - 60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2)
    cv2.circle(sheet, (150, 200), 15, (30, 30, 30), 2)
    return sheet


def _camera_quad(sw, sh, W, H, tilt_deg=25, yaw_deg=10, roll_deg=5, fill=0.7, f_scale=0.9):
    """Corners of an sw x sh sheet seen by a pinhole camera (centre principal point)."""
    t, y, r = np.radians([tilt_deg, yaw_deg, roll_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    f = f_scale * max(W, H)
    pts = np.array([[-sw / 2, -sh / 2, 0], [sw / 2, -sh / 2, 0],
                    [sw / 2, sh / 2, 0], [-sw / 2, sh / 2, 0]], float)
    Z = f * max(sw, sh) / (fill * min(W, H))
    cam = pts @ R.T + np.array([0, 0, Z])
    return np.float32(np.stack([f * cam[:, 0] / cam[:, 2] + W / 2,
                                f * cam[:, 1] / cam[:, 2] + H / 2], 1))


def _warped_on_table(sheet, W=1000, H=1200):
    sh, sw = sheet.shape[:2]
    table = np.full((H, W, 3), (40, 60, 80), np.uint8)
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    M = cv2.getPerspectiveTransform(src, _camera_quad(sw, sh, W, H))
    warped = cv2.warpPerspective(sheet, M, (W, H))
    mask = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), M, (W, H))
    table[mask > 0] = warped[mask > 0]
    return table


def test_returns_image_and_steps():
    res = enhance(_drawing_sheet())
    assert isinstance(res, EnhanceResult)
    assert res.image.dtype == np.uint8 and res.image.ndim == 3 and res.image.shape[2] == 3
    assert res.steps and all(isinstance(s, str) for s in res.steps)


@pytest.mark.parametrize("img", [
    np.full((400, 300, 3), 255, np.uint8),   # blank white
    np.zeros((400, 300, 3), np.uint8),       # blank black
    np.full((5, 7, 3), 128, np.uint8),       # tiny
    np.full((1, 1, 3), 10, np.uint8),        # 1 pixel
])
def test_degenerate_inputs_do_not_crash(img):
    res = enhance(img)
    assert res.image.ndim == 3 and res.image.shape[2] == 3
    assert res.image.size > 0


def test_grayscale_and_bgra_inputs():
    gray = cv2.cvtColor(_drawing_sheet(), cv2.COLOR_BGR2GRAY)
    assert enhance(gray).image.shape[2] == 3
    bgra = cv2.cvtColor(_drawing_sheet(), cv2.COLOR_BGR2BGRA)
    assert enhance(bgra).image.shape[2] == 3


def test_perspective_recovers_sheet_aspect():
    sheet = _drawing_sheet(600, 850)
    res = enhance(_warped_on_table(sheet), upscale=1.0)
    assert any("perspective corrected" in s for s in res.steps), res.steps
    h, w = res.image.shape[:2]
    assert abs((h / w) - 850 / 600) / (850 / 600) < 0.12
    # table colour must be gone: the border of the result should be paper-bright
    border = np.concatenate([res.image[:5].reshape(-1, 3), res.image[-5:].reshape(-1, 3)])
    assert border.mean() > 180


def test_no_paper_outline_is_logged():
    res = enhance(_drawing_sheet())  # sheet fills the frame: nothing to warp
    assert any("no paper outline" in s for s in res.steps), res.steps


def test_deskew_small_rotation():
    sheet = _drawing_sheet(800, 800)
    M = cv2.getRotationMatrix2D((400, 400), 6, 1.0)
    rot = cv2.warpAffine(sheet, M, (800, 800), borderMode=cv2.BORDER_REPLICATE)
    res = enhance(rot, upscale=1.0)
    assert any("deskewed" in s for s in res.steps), res.steps


@pytest.mark.parametrize("upscale,max_side", [(2.0, 3000), (4.0, 1000), (1.0, 500)])
def test_max_side_respected(upscale, max_side):
    res = enhance(_drawing_sheet(1200, 1700), upscale=upscale, max_side=max_side)
    assert max(res.image.shape[:2]) <= max_side


def test_upscale_applied_when_room():
    res = enhance(_drawing_sheet(300, 400), upscale=2.0, max_side=3000)
    assert max(res.image.shape[:2]) >= 700


def test_enhance_file_roundtrip(tmp_path):
    src = tmp_path / "in.jpg"
    cv2.imwrite(str(src), _drawing_sheet())
    dst = tmp_path / "out" / "enh.png"
    res = enhance_file(src, dst)
    assert dst.exists()
    back = cv2.imread(str(dst))
    assert back.shape == res.image.shape


def test_enhance_file_bad_input(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        enhance_file(bad, tmp_path / "x.png")
