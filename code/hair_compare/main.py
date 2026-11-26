"""
3x3 comparison grid of the final masks from 4 methods per sample:
  - frangi        : code/hair_frangi_mask/output/<sample>/04_mask.png
  - oriented_line : code/hair_oriented_line/output/<sample>/06_new_thresh.png
  - bilateral B@90: code/hair_bilateral/output/<sample>/11_B_p90.png
  - canny         : code/hair_canny/output/<sample>/07_canny_mask.png

Run : python code/hair_compare/main.py   (from repo root)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC_DIR = ROOT / "sources" / "exclude_crops"
OUT_ROOT = HERE / "output"

CELL_W = 1200


def _sample_name(stem: str) -> str:
    parts = stem.split("-")
    return f"{parts[0]}-{parts[2]}" if len(parts) >= 3 else stem


def _find_samples() -> list[tuple[str, Path]]:
    samples = []
    for rgba in sorted(SRC_DIR.glob("*_exclude_rgba.png")):
        stem = rgba.name.replace("_exclude_rgba.png", "")
        samples.append((_sample_name(stem), rgba))
    if not samples:
        raise FileNotFoundError(f"No *_exclude_rgba.png under {SRC_DIR}")
    return samples


def _overlay(roi_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = roi_bgr.copy()
    out[mask > 0] = (0, 0, 255)
    return out


def _make_grid_3x3(tiles: list[tuple[str, np.ndarray]], out_path: Path, cell_w: int = CELL_W) -> None:
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
    samples = _find_samples()

    for idx, (name, rgba_path) in enumerate(samples, 1):
        rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
        roi_mask = rgba[..., 3] > 0
        roi_bgr = rgba[..., :3].copy()
        roi_bgr[~roi_mask] = 0

        method_paths = [
            ("frangi", ROOT / "code" / "hair_frangi_mask" / "output" / name / "04_mask.png"),
            ("oriented_line", ROOT / "code" / "hair_oriented_line" / "output" / name / "06_new_thresh.png"),
            ("bilateral B@90", ROOT / "code" / "hair_bilateral" / "output" / name / "11_B_p90.png"),
            ("canny", ROOT / "code" / "hair_canny" / "output" / name / "07_canny_mask.png"),
        ]

        masks: list[tuple[str, np.ndarray]] = []
        for label, p in method_paths:
            if not p.is_file():
                raise FileNotFoundError(f"Missing {label} output: {p}")
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            masks.append((label, m))

        tiles: list[tuple[str, np.ndarray]] = [("1. roi", roi_bgr)]
        for i, (label, m) in enumerate(masks, 2):
            tiles.append((f"{i}. {label} (mask)", m))
        for i, (label, m) in enumerate(masks, 6):
            tiles.append((f"{i}. {label} (overlay)", _overlay(roi_bgr, m)))

        out_path = OUT_ROOT / f"{idx:02d}_{name}_compare_3x3.png"
        _make_grid_3x3(tiles, out_path, cell_w=CELL_W)
        print(f"Saved {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
