"""
Canny control vs oriented line filter, on the same input (bilateral B + black-hat).

Input : sources/exclude_crops/*_exclude_bbox.jpg  (full face context, aligned with rgba)
        sources/exclude_crops/*_exclude_rgba.png  (alpha = ROI)

Shared input (identical for both branches):
  luma = gray(bbox) -> bilateral(d=7, sigma_color=20, sigma_space=5) -> black-hat(7x7 ellipse)

Branch L (reference, from hair_bilateral B@90):
  line = max over 16 orientations of line kernel (len 21) * tophat
  mask = line >= 90th percentile inside ROI

Branch C (Canny control):
  edges = Canny(tophat, hysteresis from ROI gradient percentiles)
  skel  = skeletonize(dilate(edges))          # collapse the two parallel edges of a dark strand
  mask  = skeleton components with >= MIN_LEN pixels

Run   : python code/hair_canny/main.py   (from repo root)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC_DIR = ROOT / "sources" / "exclude_crops"
OUT_ROOT = HERE / "output"

BIL_D, BIL_SC, BIL_SS = 7, 20.0, 5.0
BLACKHAT_KSIZE = 7
LINE_LEN = 21
N_ANGLES = 16
LINE_PCT = 90.0  # reference branch, user-chosen

CANNY_LO_PCT = 75.0  # hysteresis low, percentile of ROI gradient magnitude
CANNY_HI_PCT = 95.0  # hysteresis high
DILATE_PX = 1  # merge the two edges of a strand before skeletonizing
MIN_LEN = 12  # px, skeleton component length to keep

CELL_W = 1200

STEPS: list[tuple[str, str]] = [
    ("01_roi.png", "1. roi"),
    ("02_luma_bilateral.png", "2. bilateral B (luma)"),
    ("03_tophat.png", "3. black-hat (input)"),
    ("04_gradmag.png", "4. gradient magnitude"),
    ("05_canny_edges.png", "5. canny edges (hysteresis)"),
    ("06_skeleton.png", "6. skeleton after dilate"),
    ("07_canny_mask.png", "7. canny mask (len >= MIN_LEN)"),
    ("08_line_B90.png", "8. line filter B @ 90% (ref)"),
    ("09_compare.png", "9. red=canny only, blue=line only, green=both"),
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


def _line_response(tophat: np.ndarray, roi: np.ndarray) -> np.ndarray:
    resp = np.stack(
        [cv2.filter2D(tophat, cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT) for k in _KERNELS]
    )
    return np.where(roi, resp.max(axis=0), 0.0).astype(np.float32)


def _pct_mask(x: np.ndarray, roi: np.ndarray, pct: float) -> np.ndarray:
    thr = float(np.percentile(x[roi], pct))
    return (x >= thr) & roi


def _mask_u8(m: np.ndarray) -> np.ndarray:
    return m.astype(np.uint8) * 255


def _to_u8(x: np.ndarray, roi: np.ndarray) -> np.ndarray:
    hi = float(np.percentile(x[roi], 99.5)) if roi.any() else 1.0
    out = np.clip(x / max(hi, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    out[~roi] = 0
    return out


def _canny_branch(tophat_u8: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    gx = cv2.Sobel(tophat_u8.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tophat_u8.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    gmag = np.where(roi, gmag, 0.0)
    lo = float(np.percentile(gmag[roi], CANNY_LO_PCT))
    hi = float(np.percentile(gmag[roi], CANNY_HI_PCT))
    edges = cv2.Canny(tophat_u8, lo, hi, L2gradient=True)
    edges = np.where(roi, edges, 0).astype(np.uint8)
    edges_bool = edges > 0

    dil = cv2.dilate(edges_bool.astype(np.uint8), np.ones((2 * DILATE_PX + 1, 2 * DILATE_PX + 1), np.uint8))
    skel = skeletonize(dil > 0)
    skel &= roi

    n, lab, stats, _ = cv2.connectedComponentsWithStats(skel.astype(np.uint8), connectivity=8)
    keep_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_LEN]
    mask = np.isin(lab, keep_ids) & skel
    return edges_bool, skel, mask, lo, hi


def _frag_stats(m: np.ndarray) -> tuple[int, int, int]:
    n, _lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return 0, 0, 0
    return int(len(areas)), int(np.median(areas)), int((areas >= 20).sum())


def _run_one(name: str, rgba_path: Path, bbox_path: Path, out_dir: Path) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    roi = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)[..., 3] > 0
    bgr = cv2.imread(str(bbox_path))
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bil = cv2.bilateralFilter(luma, BIL_D, BIL_SC, BIL_SS)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KSIZE, BLACKHAT_KSIZE))
    tophat_f = cv2.morphologyEx(bil, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    tophat_f[~roi] = 0.0
    tophat_u8 = np.clip(tophat_f, 0, 255).astype(np.uint8)

    edges, skel, canny_mask, lo, hi = _canny_branch(tophat_u8, roi)
    line = _line_response(tophat_f, roi)
    line_mask = _pct_mask(line, roi, LINE_PCT)

    gx = cv2.Sobel(tophat_u8.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tophat_u8.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gmag_vis = np.where(roi, np.sqrt(gx * gx + gy * gy), 0.0)

    bil_vis = bil.copy()
    bil_vis[~roi] = 0
    roi_vis = bgr.copy()
    roi_vis[~roi] = 0
    compare = (bgr * 0.35).astype(np.uint8)
    compare[canny_mask & ~line_mask] = (0, 0, 255)
    compare[line_mask & ~canny_mask] = (255, 160, 0)
    compare[canny_mask & line_mask] = (0, 255, 0)
    compare[~roi] = 0

    files = {
        "01_roi.png": roi_vis,
        "02_luma_bilateral.png": bil_vis,
        "03_tophat.png": _to_u8(tophat_f, roi),
        "04_gradmag.png": _to_u8(gmag_vis, roi),
        "05_canny_edges.png": _mask_u8(edges),
        "06_skeleton.png": _mask_u8(skel),
        "07_canny_mask.png": _mask_u8(canny_mask),
        "08_line_B90.png": _mask_u8(line_mask),
        "09_compare.png": compare,
    }
    for fname, img in files.items():
        cv2.imwrite(str(out_dir / fname), img)

    c_n, c_med, c_big = _frag_stats(canny_mask)
    l_n, l_med, l_big = _frag_stats(line_mask)
    inter = int(np.count_nonzero(canny_mask & line_mask))
    union = int(np.count_nonzero(canny_mask | line_mask))
    lines = [
        f"sample: {name}",
        f"roi px: {int(np.count_nonzero(roi))}",
        f"canny thresholds (gradmag pct {CANNY_LO_PCT}/{CANNY_HI_PCT}): lo={lo:.1f} hi={hi:.1f}",
        f"canny: skeleton px={int(np.count_nonzero(skel))} mask px={int(np.count_nonzero(canny_mask))} "
        f"comps={c_n} median_area={c_med} comps>=20px={c_big}",
        f"line B@{LINE_PCT:.0f}: mask px={int(np.count_nonzero(line_mask))} "
        f"comps={l_n} median_area={l_med} comps>=20px={l_big}",
        f"IoU(canny, line)={inter / max(union, 1):.3f}  canny-only={int(np.count_nonzero(canny_mask & ~line_mask))}  "
        f"line-only={int(np.count_nonzero(line_mask & ~canny_mask))}",
    ]
    (out_dir / "stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return {
        "canny_px": int(np.count_nonzero(canny_mask)),
        "line_px": int(np.count_nonzero(line_mask)),
        "canny_comps": c_n,
        "line_comps": l_n,
    }


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
    print(
        f"bilateral d={BIL_D} sc={BIL_SC} ss={BIL_SS} blackhat={BLACKHAT_KSIZE} "
        f"line_len={LINE_LEN} line_pct={LINE_PCT} canny_pct={CANNY_LO_PCT}/{CANNY_HI_PCT} "
        f"dilate={DILATE_PX} min_len={MIN_LEN}"
    )
    sample_dirs: list[tuple[str, Path]] = []
    totals = {"canny_px": 0, "line_px": 0, "canny_comps": 0, "line_comps": 0}
    for idx, (name, rgba, bbox) in enumerate(samples, 1):
        print(f"\n[{idx}/{len(samples)}] === {name} ===")
        out_dir = OUT_ROOT / name
        s = _run_one(name, rgba, bbox, out_dir)
        for k in totals:
            totals[k] += s[k]
        sample_dirs.append((name, out_dir))
    _save_grids(sample_dirs)
    print(
        f"\nTOTAL canny px={totals['canny_px']} comps={totals['canny_comps']}  |  "
        f"line B@{LINE_PCT:.0f} px={totals['line_px']} comps={totals['line_comps']}"
    )
    print(f"Saved per-sample and per-step grids → {OUT_ROOT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
