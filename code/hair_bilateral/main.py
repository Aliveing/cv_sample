"""
Bilateral pre-smoothing A/B for strand vs pore-texture separation.

Input : sources/exclude_crops/*_exclude_bbox.jpg  (full face context, aligned with rgba)
        sources/exclude_crops/*_exclude_rgba.png  (alpha = ROI)
        code/hair_frangi_mask/output/<sample>/04_mask.png  (reference only)

Pipeline (same for every variant, only the pre-filter differs):
  1. luma   = gray(bbox)
  2. pre    = bilateral(luma)  (variants B/C) or luma (variant A = baseline)
  3. tophat = black-hat(pre, 7x7 ellipse)           (dark strands -> bright)
  4. line   = max over 16 orientations of line kernel (len 21) * tophat
  5. mask   = line >= 97th percentile inside ROI     (same kept fraction per variant)

Variants
  A  none                      (baseline)
  B  bilateral d=7 sc=20 ss=5
  C  bilateral d=9 sc=40 ss=7

Run   : python code/hair_bilateral/main.py   (from repo root)
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

BLACKHAT_KSIZE = 7
LINE_LEN = 21
N_ANGLES = 16
PCT = 97.0
PCT_SWEEP = (90.0, 93.0, 95.0, 97.0)  # variant B only, for choosing a looser threshold

VARIANTS: dict[str, dict | None] = {
    "A": None,
    "B": {"d": 7, "sigma_color": 20.0, "sigma_space": 5.0},
    "C": {"d": 9, "sigma_color": 40.0, "sigma_space": 7.0},
}

CELL_W = 1200

STEPS: list[tuple[str, str]] = [
    ("01_roi.png", "1. roi"),
    ("02_luma.png", "2. luma (raw)"),
    ("03_bilateral_B.png", "3. bilateral B"),
    ("04_tophat_A.png", "4. tophat A (none)"),
    ("05_tophat_B.png", "5. tophat B"),
    ("06_tophat_C.png", "6. tophat C"),
    ("07_mask_A.png", "7. mask A (97%)"),
    ("08_mask_B.png", "8. mask B (97%)"),
    ("09_mask_C.png", "9. mask C (97%)"),
]


def _sample_name(stem: str) -> str:
    parts = stem.split("-")
    return f"{parts[0]}-{parts[2]}" if len(parts) >= 3 else stem


def _find_samples() -> list[tuple[str, Path, Path, Path]]:
    samples = []
    for rgba in sorted(SRC_DIR.glob("*_exclude_rgba.png")):
        stem = rgba.name.replace("_exclude_rgba.png", "")
        bbox = SRC_DIR / f"{stem}_exclude_bbox.jpg"
        frangi = FRANGI_OUT / _sample_name(stem) / "04_mask.png"
        if not bbox.is_file() or not frangi.is_file():
            raise FileNotFoundError(f"missing bbox or frangi ref for {stem}")
        samples.append((_sample_name(stem), rgba, bbox, frangi))
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


_KERNELS = _line_kernels(LINE_LEN, N_ANGLES)


def _tophat_dark(luma_u8: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KSIZE, BLACKHAT_KSIZE))
    return cv2.morphologyEx(luma_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)


def _line_response(tophat: np.ndarray, roi: np.ndarray) -> np.ndarray:
    resp = np.stack(
        [cv2.filter2D(tophat, cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT) for k in _KERNELS]
    )
    return np.where(roi, resp.max(axis=0), 0.0).astype(np.float32)


def _threshold_pct(x: np.ndarray, roi: np.ndarray, pct: float) -> np.ndarray:
    thr = float(np.percentile(x[roi], pct))
    return (x >= thr) & roi


def _to_u8(x: np.ndarray, roi: np.ndarray) -> np.ndarray:
    hi = float(np.percentile(x[roi], 99.5)) if roi.any() else 1.0
    out = np.clip(x / max(hi, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    out[~roi] = 0
    return out


def _mask_u8(m: np.ndarray) -> np.ndarray:
    return m.astype(np.uint8) * 255


def _run_one(
    name: str, rgba_path: Path, bbox_path: Path, frangi_path: Path, out_dir: Path
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    roi = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)[..., 3] > 0
    bgr = cv2.imread(str(bbox_path))
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    frangi = (cv2.imread(str(frangi_path), cv2.IMREAD_GRAYSCALE) > 0) & roi

    bil_B = cv2.bilateralFilter(luma, VARIANTS["B"]["d"], VARIANTS["B"]["sigma_color"], VARIANTS["B"]["sigma_space"])
    bil_C = cv2.bilateralFilter(luma, VARIANTS["C"]["d"], VARIANTS["C"]["sigma_color"], VARIANTS["C"]["sigma_space"])
    pre = {"A": luma, "B": bil_B, "C": bil_C}

    tops = {k: _tophat_dark(v) for k, v in pre.items()}
    lines = {k: _line_response(t, roi) for k, t in tops.items()}
    masks = {k: _threshold_pct(lines[k], roi, PCT) for k in lines}

    roi_vis = bgr.copy()
    roi_vis[~roi] = 0
    luma_vis = luma.copy()
    luma_vis[~roi] = 0
    bil_vis = bil_B.copy()
    bil_vis[~roi] = 0
    files = {
        "01_roi.png": roi_vis,
        "02_luma.png": luma_vis,
        "03_bilateral_B.png": bil_vis,
        "04_tophat_A.png": _to_u8(tops["A"], roi),
        "05_tophat_B.png": _to_u8(tops["B"], roi),
        "06_tophat_C.png": _to_u8(tops["C"], roi),
        "07_mask_A.png": _mask_u8(masks["A"]),
        "08_mask_B.png": _mask_u8(masks["B"]),
        "09_mask_C.png": _mask_u8(masks["C"]),
    }
    for p in PCT_SWEEP:
        sweep_mask = _threshold_pct(lines["B"], roi, p)
        files[f"11_B_p{int(p)}.png"] = _mask_u8(sweep_mask)
    for fname, img in files.items():
        cv2.imwrite(str(out_dir / fname), img)

    # Compare B against Frangi reference (green both, red B-only, blue Frangi-only)
    mb = masks["B"]
    compare = (bgr * 0.35).astype(np.uint8)
    compare[mb & ~frangi] = (0, 0, 255)
    compare[frangi & ~mb] = (255, 160, 0)
    compare[mb & frangi] = (0, 255, 0)
    compare[~roi] = 0
    cv2.imwrite(str(out_dir / "10_compare_B_vs_frangi.png"), compare)

    stats: dict[str, float] = {"roi": int(np.count_nonzero(roi)), "frangi": int(np.count_nonzero(frangi))}
    lines_out = [f"sample: {name}", f"roi px: {stats['roi']}", f"frangi ref px: {stats['frangi']}"]
    for k in ("A", "B", "C"):
        m = masks[k]
        inter = int(np.count_nonzero(m & frangi))
        union = int(np.count_nonzero(m | frangi))
        iou = inter / max(union, 1)
        stats[f"kept_{k}"] = int(np.count_nonzero(m))
        stats[f"iou_{k}"] = iou
        stats[f"mean_line_{k}"] = float(lines[k][roi].mean())
        lines_out.append(
            f"{k}: kept={stats[f'kept_{k}']} IoU(frangi)={iou:.3f} "
            f"tophat_mean_roi={float(tops[k][roi].mean()):.2f}"
        )
    (out_dir / "stats.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print("\n".join(lines_out))
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
    for p in PCT_SWEEP:
        fname = f"11_B_p{int(p)}.png"
        tiles = []
        for idx, (name, d) in enumerate(sample_dirs, 1):
            img = cv2.imread(str(d / fname), cv2.IMREAD_UNCHANGED)
            if img is not None:
                tiles.append((f"{idx}. {name}  B @ {int(p)}%", img))
        if tiles:
            _make_grid(tiles, OUT_ROOT / fname)
    tiles = []
    for idx, (name, d) in enumerate(sample_dirs, 1):
        img = cv2.imread(str(d / "10_compare_B_vs_frangi.png"), cv2.IMREAD_UNCHANGED)
        if img is not None:
            tiles.append((f"{idx}. {name}", img))
    if tiles:
        _make_grid(tiles, OUT_ROOT / "10_compare_B_vs_frangi.png")


def main() -> None:
    samples = _find_samples()
    print(
        f"blackhat={BLACKHAT_KSIZE} line_len={LINE_LEN} angles={N_ANGLES} pct={PCT} "
        f"variants={ {k: v for k, v in VARIANTS.items()} }"
    )
    sample_dirs: list[tuple[str, Path]] = []
    totals = {"frangi": 0, "A": 0, "B": 0, "C": 0}
    for idx, (name, rgba, bbox, frangi) in enumerate(samples, 1):
        print(f"\n[{idx}/{len(samples)}] === {name} ===")
        out_dir = OUT_ROOT / name
        s = _run_one(name, rgba, bbox, frangi, out_dir)
        totals["frangi"] += s["frangi"]
        for k in ("A", "B", "C"):
            totals[k] += s[f"kept_{k}"]
        sample_dirs.append((name, out_dir))
    _save_grids(sample_dirs)
    print(
        f"\nTOTAL frangi ref={totals['frangi']}  kept A={totals['A']}  B={totals['B']}  C={totals['C']}"
    )
    print(f"Saved per-sample and per-step grids → {OUT_ROOT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
