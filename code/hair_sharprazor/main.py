"""SharpRazor-inspired hair / ruler detection (mask + optional inpaint).

Paper: Kasmi et al., Skin Res Technol 2023 — SharpRazor
(https://doi.org/10.1111/srt.13203). Lab port for side-by-side compare with
code/hair_frangi_mask/. Random-forest FP classifier is replaced by geometric
gates (eccentricity / major / solidity). Same exclude_crops RGBA input.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# --- tune here ---
RED_SCALE = 1.64
MEDIAN_K = 3
DIR_RANGE_THR = 10  # paper: intensity range in 1x7 guide → probable hair
BG_CLOSE_R = 2
GRAD_ANGLES_DEG = (0.0, 22.5, 67.5)
GRAD_OTSU_OFFSET = 1.0  # T = Otsu + 1 (on uint8 gradient magnitude)
LINE_SE_LENS = (9, 15, 21, 31)  # multi-width dark-line enhancers
LINE_SE_ANGLES = (0, 30, 60, 90, 120, 150)
INCLUDE_WHITE_HAIR = True
INCLUDE_RULER = False  # forehead crop usually no ruler
WHITE_MAJOR_MIN = 20
WHITE_ECC_MIN = 0.90
# geometric cleanup (stand-in for paper RF)
MIN_AREA = 8
MIN_MAJOR = 12
MIN_ECC = 0.82
MAX_SOLIDITY = 0.95  # very filled blobs → likely not hair
CLOSE_KSIZE = 5
DO_INPAINT = True
INPAINT_DILATE = 1

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "sources" / "exclude_crops"
OUT = Path(__file__).resolve().parent / "output"


def _find_rgba() -> Path:
    matches = sorted(SRC_DIR.glob("*_exclude_rgba.png"))
    if not matches:
        raise FileNotFoundError(f"No *_exclude_rgba.png under {SRC_DIR}")
    return matches[0]


def _load_bgr_roi(rgba_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise RuntimeError(f"Failed to read {rgba_path}")
    if rgba.ndim != 3 or rgba.shape[2] < 4:
        raise ValueError(f"Expected BGRA, got shape {rgba.shape}")
    bgr = rgba[:, :, :3].copy()
    roi = rgba[:, :, 3] > 0
    if not np.any(roi):
        stem = rgba_path.name.replace("_exclude_rgba.png", "")
        alt = SRC_DIR / f"{stem}_exclude_mask.png"
        if alt.is_file():
            m = cv2.imread(str(alt), cv2.IMREAD_GRAYSCALE)
            roi = m > 0
    return bgr, roi.astype(bool)


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), dtype=np.uint8)
    c = length // 2
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    for t in range(-c, c + 1):
        x = int(round(c + t * dx))
        y = int(round(c + t * dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1
    return k


def _prep_red(bgr: np.ndarray, roi: np.ndarray) -> np.ndarray:
    # OpenCV BGR → paper uses red channel
    red = bgr[:, :, 2].astype(np.float32)
    if MEDIAN_K > 1:
        red = cv2.medianBlur(red.astype(np.uint8), MEDIAN_K | 1).astype(np.float32)
    scaled = np.clip(red * RED_SCALE, 0, 255)
    # fill outside so directional ops don't see hard black edge
    if np.any(roi):
        fill = float(np.median(scaled[roi]))
        scaled = np.where(roi, scaled, fill)
    return scaled.astype(np.float32)


def _directional_background(gray: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Approx paper H/V panel filter via multi-orientation morphological closing."""
    g = gray.astype(np.uint8)
    bg = np.zeros_like(g, dtype=np.float32)
    for length in LINE_SE_LENS:
        for ang in LINE_SE_ANGLES:
            se = _line_kernel(length, float(ang))
            closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, se)
            bg = np.maximum(bg, closed.astype(np.float32))
    # paper: also morphological closing disk r=2 on max of directional planes
    disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * BG_CLOSE_R + 1, 2 * BG_CLOSE_R + 1)
    )
    bg_u8 = np.clip(bg, 0, 255).astype(np.uint8)
    bg = cv2.morphologyEx(bg_u8, cv2.MORPH_CLOSE, disk).astype(np.float32)
    return np.where(roi, bg, 0.0)


def _enhance_dark(gray: np.ndarray, background: np.ndarray, roi: np.ndarray) -> np.ndarray:
    # Enhanced hair = Background − I_new  (dark hair → positive)
    enh = np.maximum(background - gray, 0.0)
    return np.where(roi, enh, 0.0).astype(np.float32)


def _tri_directional_mask(enhanced: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Gradients at 0 / 22.5 / 67.5°, Otsu+1 on each, OR."""
    img = np.clip(enhanced, 0, 255).astype(np.uint8)
    # blur lightly to stabilize gradients
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    mask = np.zeros(roi.shape, dtype=bool)
    for ang in GRAD_ANGLES_DEG:
        # rotate so Sobel-x aligns with hair-edge normal ≈ angle
        m = cv2.getRotationMatrix2D(
            (img.shape[1] / 2.0, img.shape[0] / 2.0), float(ang), 1.0
        )
        rot = cv2.warpAffine(
            img,
            m,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        gx = cv2.Sobel(rot, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(rot, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        # rotate magnitude back
        m_inv = cv2.getRotationMatrix2D(
            (img.shape[1] / 2.0, img.shape[0] / 2.0), -float(ang), 1.0
        )
        mag = cv2.warpAffine(
            mag,
            m_inv,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mag_u8 = np.clip(mag, 0, 255).astype(np.uint8)
        vals = mag_u8[roi]
        if vals.size == 0 or int(vals.max()) == 0:
            continue
        thr_otsu, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(thr_otsu) + GRAD_OTSU_OFFSET
        mask |= (mag >= thr) & roi
    return mask


def _white_hair_mask(bgr: np.ndarray, roi: np.ndarray) -> np.ndarray:
    b = bgr[:, :, 0].astype(np.float32)
    gm = cv2.medianBlur(b.astype(np.uint8), 3).astype(np.float32)
    wh = np.maximum(b - gm, 0.0)
    binary = (wh > 1.0) & roi
    return _filter_white_objects(binary)


def _filter_white_objects(binary: np.ndarray) -> np.ndarray:
    u8 = binary.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    keep = np.zeros(binary.shape, dtype=bool)
    for i in range(1, n):
        comp = labels == i
        if int(stats[i, cv2.CC_STAT_AREA]) < 4:
            continue
        cnt = np.column_stack(np.where(comp))[:, ::-1].astype(np.float32)
        if cnt.shape[0] < 5:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(cnt)
        major = float(max(w, h))
        minor = float(max(min(w, h), 1e-3))
        ecc = np.sqrt(max(1.0 - (minor / major) ** 2, 0.0))
        x, y, bw, bh, area = stats[i]
        extent = area / max(bw * bh, 1)
        if major >= WHITE_MAJOR_MIN and ecc >= WHITE_ECC_MIN and extent <= 0.9:
            keep[comp] = True
    return keep


def _ruler_mask(red: np.ndarray, roi: np.ndarray) -> np.ndarray:
    r = np.clip(red, 0, 255).astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))
    closed = cv2.morphologyEx(r, cv2.MORPH_CLOSE, se)
    diff = cv2.subtract(closed, r)
    vals = diff[roi]
    if vals.size == 0 or int(vals.max()) == 0:
        return np.zeros(roi.shape, dtype=bool)
    mn, mx = float(vals.min()), float(vals.max())
    thr = (2.0 / 5.0) * (mx - mn) + mn
    binary = (diff.astype(np.float32) > thr) & roi
    # filter 3: length / solidity
    u8 = binary.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    keep = np.zeros(roi.shape, dtype=bool)
    h, w = roi.shape
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 25:
            continue
        comp = (labels == i).astype(np.uint8)
        cnt = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cnt:
            continue
        hull = cv2.convexHull(cnt[0])
        hull_a = float(cv2.contourArea(hull)) or 1.0
        solidity = area / hull_a
        if solidity < 0.6:
            continue
        # filter 4: drop central quadrant of 3x3 grid
        cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
        cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] // 2
        if w // 3 <= cx < 2 * w // 3 and h // 3 <= cy < 2 * h // 3:
            continue
        keep[comp.astype(bool)] = True
    return keep


def _geometric_cleanup(binary: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Stand-in for paper random-forest FP removal."""
    u8 = (binary & roi).astype(np.uint8)
    # break at junctions roughly: skeleton + remove branch-ish via open
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    keep = np.zeros(binary.shape, dtype=bool)
    kept = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA:
            continue
        comp = (labels == i).astype(np.uint8)
        ys, xs = np.where(comp > 0)
        if ys.size < 5:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        major = float(max(w, h))
        minor = float(max(min(w, h), 1e-3))
        ecc = np.sqrt(max(1.0 - (minor / major) ** 2, 0.0))
        cnt = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        solidity = 1.0
        if cnt:
            hull = cv2.convexHull(cnt[0])
            ha = float(cv2.contourArea(hull)) or 1.0
            solidity = area / ha
        if major >= MIN_MAJOR and ecc >= MIN_ECC and solidity <= MAX_SOLIDITY:
            keep[comp.astype(bool)] = True
            kept += 1
    if CLOSE_KSIZE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (CLOSE_KSIZE, CLOSE_KSIZE)
        )
        keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(
            bool
        )
    keep &= roi
    print(f"geometric cleanup: comps={n - 1} kept={kept} px→{int(keep.sum())}")
    return keep


def _directional_inpaint(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Simplified paper inpaint: horizontal then vertical linear fill + edge median."""
    out = bgr.copy()
    m = mask.astype(np.uint8)
    if INPAINT_DILATE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * INPAINT_DILATE + 1, 2 * INPAINT_DILATE + 1)
        )
        m = cv2.dilate(m, k)
    # OpenCV Telea as practical stand-in (full paper AvgL/AvgR ramp is slower)
    out = cv2.inpaint(out, m, 3, cv2.INPAINT_TELEA)
    # smooth only near mask edge
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    if np.any(edge):
        med = cv2.medianBlur(out, 11)
        out = np.where(edge[..., None] > 0, med, out)
    return out


def _overlay(bgr: np.ndarray, mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
    vis = bgr.copy()
    vis[~roi] = 0
    tint = vis.astype(np.float32)
    tint[mask] = tint[mask] * 0.35 + np.array([0, 0, 220], dtype=np.float32)
    out = np.clip(tint, 0, 255).astype(np.uint8)
    out[~roi] = 0
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rgba_path = _find_rgba()
    bgr, roi = _load_bgr_roi(rgba_path)

    roi_vis = bgr.copy()
    roi_vis[~roi] = 0
    cv2.imwrite(str(OUT / "01_roi.png"), roi_vis)

    red = _prep_red(bgr, roi)
    cv2.imwrite(str(OUT / "01b_red_scaled.png"), np.clip(red, 0, 255).astype(np.uint8))

    background = _directional_background(red, roi)
    cv2.imwrite(
        str(OUT / "02_background.png"), np.clip(background, 0, 255).astype(np.uint8)
    )

    enhanced = _enhance_dark(red, background, roi)
    enh_u8 = np.zeros_like(red, dtype=np.uint8)
    if np.any(roi) and float(enhanced[roi].max()) > 0:
        mx = float(enhanced[roi].max())
        enh_u8[roi] = np.clip(enhanced[roi] / mx * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / "02b_enhanced.png"), enh_u8)

    dark_mask = _tri_directional_mask(enhanced, roi)
    cv2.imwrite(str(OUT / "03a_dark_grad.png"), (dark_mask.astype(np.uint8) * 255))

    white_mask = (
        _white_hair_mask(bgr, roi)
        if INCLUDE_WHITE_HAIR
        else np.zeros(roi.shape, dtype=bool)
    )
    cv2.imwrite(str(OUT / "03b_white.png"), (white_mask.astype(np.uint8) * 255))

    ruler = (
        _ruler_mask(red, roi) if INCLUDE_RULER else np.zeros(roi.shape, dtype=bool)
    )
    cv2.imwrite(str(OUT / "03c_ruler.png"), (ruler.astype(np.uint8) * 255))

    raw = (dark_mask | white_mask | ruler) & roi
    cv2.imwrite(str(OUT / "04a_before_cleanup.png"), (raw.astype(np.uint8) * 255))

    mask = _geometric_cleanup(raw, roi)
    cv2.imwrite(str(OUT / "04_mask.png"), (mask.astype(np.uint8) * 255))
    cv2.imwrite(str(OUT / "05_overlay.png"), _overlay(bgr, mask, roi))

    if DO_INPAINT:
        cleaned = _directional_inpaint(bgr, mask)
        cleaned[~roi] = 0
        cv2.imwrite(str(OUT / "06_inpainted.png"), cleaned)

    print(f"input:  {rgba_path.relative_to(ROOT)}")
    print(f"output: {OUT.relative_to(ROOT)}")
    print(
        f"dark={int(dark_mask.sum())} white={int(white_mask.sum())} "
        f"ruler={int(ruler.sum())} raw={int(raw.sum())} mask={int(mask.sum())}"
    )


if __name__ == "__main__":
    main()
