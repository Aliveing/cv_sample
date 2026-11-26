"""
Self-calibrated hair mask using each image's own skin region as baseline.

No external no-hair reference images are used.
For each sample in sources/exclude_crops/:
  1. Estimate this person's own skin pixels from *_exclude_bbox.jpg (outside ROI,
     excluding black/white background and extreme dark/bright tails).
  2. Compute Y_skin = median luma of self-skin pixels, and normalize exposure
     to GAIN_TARGET (128) so bilateral sigma_color is consistent across exposures.
  3. Run bilateral B (d=7, sc=20, ss=5) -> black-hat (7x7) -> 16-angle line integral (len=21).
  4. Measure the line-response distribution on THIS person's own skin pixels
     (p99 and p99.5) and use that directly as the floor threshold inside ROI.

Run : python code/hair_self_skin/main.py   (from repo root)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC_DIR = ROOT / "sources" / "exclude_crops"
OUT_ROOT = HERE / "output"

GAIN_TARGET = 128.0
BIL_D, BIL_SC, BIL_SS = 7, 20.0, 5.0
BLACKHAT_KSIZE = 7
LINE_LEN = 21
N_ANGLES = 16
CELL_W = 1200

STEPS: list[tuple[str, str]] = [
    ("01_roi.png", "1. roi"),
    ("02_self_skin.png", "2. self skin region (green)"),
    ("03_tophat.png", "3. black-hat (gain-norm + bil B)"),
    ("04_line_resp.png", "4. line response (16 angles)"),
    ("05_darkness.png", "5. darkness (Y_skin - Y) / Y_skin"),
    ("06_mask_p99.png", "6. mask: line >= self_skin p99"),
    ("07_mask_p995.png", "7. mask: line >= self_skin p99.5"),
    ("08_overlay_p99.png", "8. overlay (p99)"),
    ("09_overlay_p995.png", "9. overlay (p99.5)"),
]


def _sample_name(stem: str) -> str:
    parts = stem.split("-")
    return f"{parts[0]}-{parts[2]}" if len(parts) >= 3 else stem


def _find_samples() -> list[tuple[str, Path, Path]]:
    samples = []
    for rgba in sorted(SRC_DIR.glob("*_exclude_rgba.png")):
        stem = rgba.name.replace("_exclude_rgba.png", "")
        bbox = SRC_DIR / f"{stem}_exclude_bbox.jpg"
        if not bbox.is_file():
            raise FileNotFoundError(bbox)
        samples.append((_sample_name(stem), rgba, bbox))
    if not samples:
        raise FileNotFoundError(f"No *_exclude_rgba.png under {SRC_DIR}")
    return samples


def _line_kernels(length: int, n: int) -> list[np.ndarray]:
    half = length // 2 + 2
    size = 2 * half + 1
    kernels = []
    for i in range(n):
        theta = np.pi * i / n
        dx, dy = np.cos(theta), np.sin(theta)
        k = np.zeros((size, size), dtype=np.float32)
        for t in np.linspace(-length / 2, length / 2, length * 4):
            k[int(round(half + t * dy)), int(round(half + t * dx))] = 1.0
        kernels.append(k / k.sum())
    return kernels


_KERNELS = _line_kernels(LINE_LEN, N_ANGLES)


def _line_response(tophat: np.ndarray) -> np.ndarray:
    resp = np.stack(
        [cv2.filter2D(tophat, cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT) for k in _KERNELS]
    )
    return resp.max(axis=0).astype(np.float32)


def _estimate_self_skin(luma: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, float]:
    """Pick clean skin pixels from bbox outside ROI, avoiding black/white bg and dark hair."""
    # Exclude border margin so line-kernel boundary artifacts don't pollute skin stats
    h, w = luma.shape
    margin = LINE_LEN // 2 + 4
    interior = np.zeros((h, w), dtype=bool)
    interior[margin : h - margin, margin : w - margin] = True

    # Erode ~roi slightly so ROI-edge hair doesn't leak into skin baseline
    dil_roi = cv2.dilate(roi.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    cand = (~dil_roi) & interior & (luma > 18) & (luma < 245)
    if cand.sum() < 500:
        cand = interior & (luma > 18) & (luma < 245)

    vals = luma[cand]
    lo, hi = float(np.percentile(vals, 30)), float(np.percentile(vals, 90))
    skin = cand & (luma >= lo) & (luma <= hi)
    y_skin = float(np.median(luma[skin]))
    return skin, y_skin


def _to_u8(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hi = float(np.percentile(x[mask], 99.5)) if mask.any() else 1.0
    out = np.clip(x / max(hi, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    out[~mask] = 0
    return out


def _mask_u8(m: np.ndarray) -> np.ndarray:
    return m.astype(np.uint8) * 255


def _frag(m: np.ndarray) -> tuple[int, int]:
    n, _lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return 0, 0
    return int(len(areas)), int(np.median(areas))


def _make_grid(tiles: list[tuple[str, np.ndarray]], out_path: Path, cell_w: int = CELL_W) -> None:
    h0, w0 = tiles[0][1].shape[:2]
    cell_h = max(200, int(round(cell_w * h0 / max(w0, 1))))
    bar_h, pad, cols, rows = 44, 8, 3, 3
    canvas = np.full(
        (rows * (cell_h + bar_h) + (rows + 1) * pad, cols * cell_w + (cols + 1) * pad, 3),
        18,
        dtype=np.uint8,
    )
    for idx, (label, img) in enumerate(tiles[: rows * cols]):
        r, c = divmod(idx, cols)
        y0 = pad + r * (cell_h + bar_h + pad)
        x0 = pad + c * (cell_w + pad)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img[:, :, :3]
        ih, iw = bgr.shape[:2]
        scale = min(cell_w / max(iw, 1), cell_h / max(ih, 1))
        nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(bgr, (nw, nh), interpolation=interp)
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w, y0 + bar_h), (36, 36, 36), -1)
        cv2.putText(
            canvas, label, (x0 + 12, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA,
        )
        ox = x0 + (cell_w - nw) // 2
        oy = y0 + bar_h + (cell_h - nh) // 2
        canvas[oy : oy + nh, ox : ox + nw] = resized
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _run_one(name: str, rgba_path: Path, bbox_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    roi = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)[..., 3] > 0
    bgr = cv2.imread(str(bbox_path))
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    skin_mask, y_skin = _estimate_self_skin(luma, roi)
    gain = np.clip(luma * (GAIN_TARGET / max(y_skin, 1.0)), 0, 255).astype(np.uint8)
    pre = cv2.bilateralFilter(gain, BIL_D, BIL_SC, BIL_SS)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KSIZE, BLACKHAT_KSIZE))
    tophat = cv2.morphologyEx(pre, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    line = _line_response(tophat)

    skin_line = line[skin_mask]
    thr_p99 = float(np.percentile(skin_line, 99.0))
    thr_p995 = float(np.percentile(skin_line, 99.5))

    mask_p99 = (line >= thr_p99) & roi
    mask_p995 = (line >= thr_p995) & roi

    darkness = np.clip((y_skin - luma) / max(y_skin, 1.0), 0.0, 0.6)

    roi_vis = bgr.copy()
    roi_vis[~roi] = 0
    skin_vis = bgr.copy()
    skin_vis[skin_mask] = (skin_vis[skin_mask] * 0.55 + np.array([0, 110, 0])).astype(np.uint8)
    ov_p99 = roi_vis.copy()
    ov_p99[mask_p99] = (0, 0, 255)
    ov_p995 = roi_vis.copy()
    ov_p995[mask_p995] = (0, 0, 255)

    files = {
        "01_roi.png": roi_vis,
        "02_self_skin.png": skin_vis,
        "03_tophat.png": _to_u8(tophat, roi),
        "04_line_resp.png": _to_u8(line, roi),
        "05_darkness.png": _to_u8(darkness, roi),
        "06_mask_p99.png": _mask_u8(mask_p99),
        "07_mask_p995.png": _mask_u8(mask_p995),
        "08_overlay_p99.png": ov_p99,
        "09_overlay_p995.png": ov_p995,
    }
    for fname, img in files.items():
        cv2.imwrite(str(out_dir / fname), img)

    n_roi = int(roi.sum())
    c99, m99 = _frag(mask_p99)
    c995, m995 = _frag(mask_p995)
    lines = [
        f"sample: {name}",
        f"self_skin: px={int(skin_mask.sum())}  Y_skin(median)={y_skin:.1f}  "
        f"line_p99={thr_p99:.2f}  line_p99.5={thr_p995:.2f}",
        f"roi px={n_roi}",
        f"p99  kept={int(mask_p99.sum()):6d} ({100.0 * mask_p99.sum() / max(n_roi, 1):5.2f}%)  comps={c99:4d}  med_area={m99}",
        f"p995 kept={int(mask_p995.sum()):6d} ({100.0 * mask_p995.sum() / max(n_roi, 1):5.2f}%)  comps={c995:4d}  med_area={m995}",
    ]
    (out_dir / "stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _save_grids(sample_dirs: list[tuple[str, Path]]) -> None:
    for name, d in sample_dirs:
        tiles = []
        for fname, label in STEPS:
            img = cv2.imread(str(d / fname), cv2.IMREAD_UNCHANGED)
            if img is not None:
                tiles.append((label, img))
        if tiles:
            _make_grid(tiles, d / "00_grid_3x3.png")
    for fname, _label in STEPS:
        tiles: list[tuple[str, np.ndarray]] = []
        for idx, (name, d) in enumerate(sample_dirs, 1):
            img = cv2.imread(str(d / fname), cv2.IMREAD_UNCHANGED)
            if img is not None:
                tiles.append((f"{idx}. {name}", img))
        if tiles:
            _make_grid(tiles, OUT_ROOT / fname)


def main() -> None:
    samples = _find_samples()
    sample_dirs: list[tuple[str, Path]] = []
    for idx, (name, rgba, bbox) in enumerate(samples, 1):
        print(f"\n[{idx}/{len(samples)}] === {name} ===")
        out_dir = OUT_ROOT / name
        _run_one(name, rgba, bbox, out_dir)
        sample_dirs.append((name, out_dir))
    _save_grids(sample_dirs)
    print(f"\nSaved per-sample and per-step grids → {OUT_ROOT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
