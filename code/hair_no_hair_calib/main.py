"""
Noise calibration with no-hair skin patches, then per-image normalized line-filter threshold.

Inputs:
  sources/no_hair_face_region/{1..7}.png   (no hair, RGBA, alpha=255 everywhere -> ROI = full image)
  sources/exclude_crops/*_exclude_bbox.jpg + *_exclude_rgba.png  (5 hair samples, alpha = ROI)

Per-image pipeline (same for all):
  1. luma   = gray(img)
  2. gain   = luma * (128 / median(luma in ROI))           exposure normalization
  3. pre    = bilateral(gain, d=7, sc=20, ss=5)            (variant B)
  4. tophat = black-hat(pre, 7x7 ellipse)
  5. line   = max over 16 orientations of line kernel (len 21) * tophat
  6. sigma  = 1.4826 * MAD(line in ROI)                     per-image noise scale
  7. z      = line / sigma

Calibration:
  k = max over no-hair images of the 99.9th percentile of z (conservative).
  Hair samples: mask = z >= k inside ROI.

Run : python code/hair_no_hair_calib/main.py   (from repo root)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC_HAIR = ROOT / "sources" / "exclude_crops"
SRC_NOHAIR = ROOT / "sources" / "no_hair_face_region"
OUT_ROOT = HERE / "output"

GAIN_TARGET = 128.0
BIL_D, BIL_SC, BIL_SS = 7, 20.0, 5.0
BLACKHAT_KSIZE = 7
LINE_LEN = 21
N_ANGLES = 16
CALIB_PCT = 99.9
CELL_W = 1200


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


def _normalized_line(bgr: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    med = float(np.median(luma[roi]))
    gain = np.clip(luma * (GAIN_TARGET / max(med, 1.0)), 0, 255).astype(np.uint8)
    pre = cv2.bilateralFilter(gain, BIL_D, BIL_SC, BIL_SS)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KSIZE, BLACKHAT_KSIZE))
    tophat = cv2.morphologyEx(pre, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    tophat[~roi] = 0.0
    line = _line_response(tophat, roi)
    vals = line[roi]
    sigma = 1.4826 * float(np.median(np.abs(vals - np.median(vals))))
    sigma = max(sigma, 1e-6)
    z = np.where(roi, line / sigma, 0.0).astype(np.float32)
    return line, sigma, z


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


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    nohair_paths = sorted(SRC_NOHAIR.glob("*.png"), key=lambda p: int(p.stem))
    if len(nohair_paths) != 7:
        print(f"warning: expected 7 no-hair images, found {len(nohair_paths)}")

    lines_out: list[str] = []
    lines_out.append("== no-hair calibration (ROI = full image) ==")
    lines_out.append(f"gain target={GAIN_TARGET} bilateral=(d{BIL_D},sc{BIL_SC},ss{BIL_SS}) "
                     f"blackhat={BLACKHAT_KSIZE} line_len={LINE_LEN} angles={N_ANGLES}")
    lines_out.append(f"{'image':<12}{'sigma':>10}{'p99':>10}{'p99.9':>10}{'max':>10}{'roi_px':>10}")

    nohair_tiles: list[tuple[str, np.ndarray]] = []
    nohair_p999: list[float] = []
    per_image_z_p999: list[float] = []
    for p in nohair_paths:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        roi = im[..., 3] > 0
        bgr = im[..., :3].copy()
        line, sigma, z = _normalized_line(bgr, roi)
        zv = z[roi]
        p99, p999, zmax = float(np.percentile(zv, 99)), float(np.percentile(zv, CALIB_PCT)), float(zv.max())
        nohair_p999.append(p999)
        per_image_z_p999.append(p999)
        lines_out.append(
            f"{p.stem:<12}{sigma:>10.3f}{p99:>10.2f}{p999:>10.2f}{zmax:>10.2f}{int(roi.sum()):>10d}"
        )
        nohair_dir = OUT_ROOT / "nohair" / p.stem
        nohair_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(nohair_dir / "z.png"), np.clip(z / max(float(np.percentile(zv, 99.5)), 1e-6) * 255, 0, 255).astype(np.uint8))
        nohair_tiles.append((f"nohair {p.stem}", bgr))

    k_conservative = max(nohair_p999)
    lines_out.append("")
    lines_out.append(f"no-hair p99.9 of z: max={k_conservative:.2f} (conservative, used as k)")
    lines_out.append("")
    lines_out.append("== hair samples: mask = z >= k ==")
    lines_out.append(f"{'sample':<22}{'roi_px':>10}{'kept':>10}{'kept%':>8}{'comps':>8}{'med_area':>10}")

    hair_paths = sorted(SRC_HAIR.glob("*_exclude_rgba.png"))
    hair_tiles: list[tuple[str, np.ndarray]] = []
    for idx, rgba_path in enumerate(hair_paths, 1):
        stem = rgba_path.name.replace("_exclude_rgba.png", "")
        parts = stem.split("-")
        name = f"{parts[0]}-{parts[2]}"
        bbox_path = SRC_HAIR / f"{stem}_exclude_bbox.jpg"
        roi = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)[..., 3] > 0
        bgr = cv2.imread(str(bbox_path))
        _line, sigma, z = _normalized_line(bgr, roi)
        mask = (z >= k_conservative) & roi
        kept = int(mask.sum())
        comps, med = _frag(mask)
        lines_out.append(
            f"{name:<22}{int(roi.sum()):>10d}{kept:>10d}{100.0 * kept / max(int(roi.sum()), 1):>7.2f}%"
            f"{comps:>8d}{med:>10d}"
        )
        hair_dir = OUT_ROOT / "hair" / name
        hair_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(hair_dir / "mask_k.png"), _mask_u8(mask))
        overlay = bgr.copy()
        overlay[~roi] = 0
        overlay[mask] = (0, 0, 255)
        cv2.imwrite(str(hair_dir / "overlay_k.png"), overlay)
        hair_tiles.append((f"{idx}. {name} (k={k_conservative:.1f})", overlay))

    stats_text = "\n".join(lines_out) + "\n"
    (OUT_ROOT / "calib_stats.txt").write_text(stats_text, encoding="utf-8")
    print(stats_text)

    _make_grid(nohair_tiles, OUT_ROOT / "nohair_grid_3x3.png")
    _make_grid(hair_tiles, OUT_ROOT / "hair_overlay_k_grid_3x3.png")
    print(f"Saved calibration stats and grids → {OUT_ROOT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
