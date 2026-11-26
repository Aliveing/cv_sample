"""
Oriented line filter on the black-top-hat response (strand candidates).

Input : code/hair_frangi_mask/output/<sample>/02e_tophat.png  (run that experiment first)
        sources/exclude_crops/*_exclude_rgba.png              (alpha = ROI)
        code/hair_frangi_mask/output/<sample>/04_mask.png     (reference only)

Method
  1. line_resp : max over N orientations of (straight line kernel of length L) * tophat.
                 Long straight bright runs score high; isotropic pore dots do not.
  2. coherence : structure tensor of tophat (Sobel + Gaussian sigma), anisotropy in [0,1].
                 1 = one dominant orientation (strand), 0 = isotropic (dot/blob).
  3. score     : line_resp * coherence, thresholded inside ROI.
  Ablation    : line_resp alone (09_line_only_thresh.png).

Run   : python code/hair_oriented_line/main.py   (from repo root)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC_DIR = ROOT / "sources" / "exclude_crops"
FRANGI_OUT = ROOT / "code" / "hair_frangi_mask" / "output"
OUT_ROOT = HERE / "output"

LINE_LEN = 21  # px, straight line kernel length
N_ANGLES = 16  # orientations over [0, pi)
ST_SIGMA = 8.0  # px, structure-tensor integration scale
PCT = 97.0  # percentile of ROI values kept
LINE_FLOOR = 20.0  # absolute floor on line_resp (0-255 units)
SCORE_FLOOR = 10.0  # absolute floor on score (0-255 units)
CELL_W = 1200

STEPS: list[tuple[str, str]] = [
    ("01_roi.png", "1. roi"),
    ("02_tophat_in.png", "2. tophat in (02e)"),
    ("03_line_resp.png", "3. line_resp (max over angles)"),
    ("04_coherence.png", "4. coherence (structure tensor)"),
    ("05_score.png", "5. score = line * coherence"),
    ("06_new_thresh.png", "6. NEW: score >= thr"),
    ("07_frangi_ref.png", "7. frangi 04_mask (ref)"),
    ("08_compare.png", "8. red=new only, blue=frangi only, green=both"),
    ("09_line_only_thresh.png", "9. ablation: line_resp >= thr"),
]


def _sample_name(stem: str) -> str:
    parts = stem.split("-")
    return f"{parts[0]}-{parts[2]}" if len(parts) >= 3 else stem


def _find_samples() -> list[tuple[str, Path, Path, Path]]:
    samples = []
    for rgba in sorted(SRC_DIR.glob("*_exclude_rgba.png")):
        name = _sample_name(rgba.name.replace("_exclude_rgba.png", ""))
        tophat = FRANGI_OUT / name / "02e_tophat.png"
        frangi = FRANGI_OUT / name / "04_mask.png"
        if not tophat.is_file() or not frangi.is_file():
            raise FileNotFoundError(f"run hair_frangi_mask first: {tophat} / {frangi}")
        samples.append((name, rgba, tophat, frangi))
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
            x = int(round(half + t * dx))
            y = int(round(half + t * dy))
            k[y, x] = 1.0
        kernels.append(k / k.sum())
    return kernels


def _line_response(tophat: np.ndarray, roi: np.ndarray) -> np.ndarray:
    kernels = _line_kernels(LINE_LEN, N_ANGLES)
    resp = np.stack(
        [cv2.filter2D(tophat, cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT) for k in kernels]
    )
    return np.where(roi, resp.max(axis=0), 0.0).astype(np.float32)


def _coherence(tophat: np.ndarray, roi: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(tophat, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tophat, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), ST_SIGMA)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), ST_SIGMA)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), ST_SIGMA)
    root = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2)
    l1 = 0.5 * (jxx + jyy + root)
    l2 = 0.5 * (jxx + jyy - root)
    coh = ((l1 - l2) / (l1 + l2 + 1e-9)) ** 2
    return np.where(roi, coh, 0.0).astype(np.float32)


def _threshold(x: np.ndarray, roi: np.ndarray, pct: float, floor: float) -> tuple[np.ndarray, float]:
    vals = x[roi]
    thr = max(float(np.percentile(vals, pct)), floor)
    return (x >= thr) & roi, thr


def _to_u8(x: np.ndarray, roi: np.ndarray) -> np.ndarray:
    hi = float(np.percentile(x[roi], 99.5)) if roi.any() else 1.0
    out = np.clip(x / max(hi, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    out[~roi] = 0
    return out


def _mask_u8(m: np.ndarray) -> np.ndarray:
    return m.astype(np.uint8) * 255


def _run_one(
    name: str, rgba_path: Path, tophat_path: Path, frangi_path: Path, out_dir: Path
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    roi = rgba[..., 3] > 0
    tophat = cv2.imread(str(tophat_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    tophat[~roi] = 0.0
    frangi = (cv2.imread(str(frangi_path), cv2.IMREAD_GRAYSCALE) > 0) & roi

    line = _line_response(tophat, roi)
    coh = _coherence(tophat, roi)
    score = (line * coh).astype(np.float32)

    new_mask, thr_s = _threshold(score, roi, PCT, SCORE_FLOOR)
    line_mask, thr_l = _threshold(line, roi, PCT, LINE_FLOOR)

    only_new = new_mask & ~frangi
    only_ref = frangi & ~new_mask
    both = new_mask & frangi
    bgr = cv2.cvtColor(cv2.imread(str(tophat_path), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    compare = (bgr * 0.35).astype(np.uint8)
    compare[only_new] = (0, 0, 255)
    compare[only_ref] = (255, 160, 0)
    compare[both] = (0, 255, 0)
    compare[~roi] = 0

    roi_vis = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)[..., :3].copy()
    roi_vis[~roi] = 0

    files = {
        "01_roi.png": roi_vis,
        "02_tophat_in.png": _to_u8(tophat, roi),
        "03_line_resp.png": _to_u8(line, roi),
        "04_coherence.png": _to_u8(coh, roi),
        "05_score.png": _to_u8(score, roi),
        "06_new_thresh.png": _mask_u8(new_mask),
        "07_frangi_ref.png": _mask_u8(frangi),
        "08_compare.png": compare,
        "09_line_only_thresh.png": _mask_u8(line_mask),
    }
    for fname, img in files.items():
        cv2.imwrite(str(out_dir / fname), img)

    n_roi = int(np.count_nonzero(roi))
    n_new = int(np.count_nonzero(new_mask))
    n_ref = int(np.count_nonzero(frangi))
    n_line = int(np.count_nonzero(line_mask))
    inter = int(np.count_nonzero(both))
    union = int(np.count_nonzero(new_mask | frangi))
    iou = inter / max(union, 1)
    stats = {
        "roi": n_roi,
        "new": n_new,
        "ref": n_ref,
        "line_only": n_line,
        "iou": iou,
        "thr_score": thr_s,
        "thr_line": thr_l,
    }
    lines = [
        f"sample: {name}",
        f"roi px: {n_roi}",
        f"thr score={thr_s:.1f} (pct {PCT} floor {SCORE_FLOOR})  thr line={thr_l:.1f} (floor {LINE_FLOOR})",
        f"NEW kept: {n_new} ({100.0 * n_new / max(n_roi, 1):.1f}% of roi)",
        f"line-only kept: {n_line} ({100.0 * n_line / max(n_roi, 1):.1f}% of roi)",
        f"frangi ref kept: {n_ref} ({100.0 * n_ref / max(n_roi, 1):.1f}% of roi)",
        f"IoU(new, frangi)={iou:.3f}  new-only={int(np.count_nonzero(only_new))}  frangi-only={int(np.count_nonzero(only_ref))}",
    ]
    (out_dir / "stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return stats


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
        f"line_len={LINE_LEN} angles={N_ANGLES} st_sigma={ST_SIGMA} pct={PCT} "
        f"line_floor={LINE_FLOOR} score_floor={SCORE_FLOOR}"
    )
    sample_dirs: list[tuple[str, Path]] = []
    totals = {"roi": 0, "new": 0, "ref": 0, "line_only": 0, "inter": 0.0, "union": 0.0}
    for idx, (name, rgba, tophat, frangi) in enumerate(samples, 1):
        print(f"\n[{idx}/{len(samples)}] === {name} ===")
        out_dir = OUT_ROOT / name
        s = _run_one(name, rgba, tophat, frangi, out_dir)
        for k in ("roi", "new", "ref", "line_only"):
            totals[k] += s[k]
        sample_dirs.append((name, out_dir))
    _save_grids(sample_dirs)
    print(
        f"\nTOTAL roi={totals['roi']}  NEW={totals['new']} "
        f"line-only={totals['line_only']}  frangi ref={totals['ref']}"
    )
    print(f"Saved per-sample and per-step grids → {OUT_ROOT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
