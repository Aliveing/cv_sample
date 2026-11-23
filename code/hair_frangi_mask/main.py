"""Extract hair mask: visible strands + optional vellus via dual Frangi branches."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import remove_small_objects

# --- tune here ---
INCLUDE_VELLUS = False
# Frangi input: "luma" | "green" | "skin_residual" | "external"
PREP_MODE = "luma"
SKIN_BLUR_SIGMA = 10.0  # local skin luminance estimate
RESIDUAL_GAIN = 2.5  # darken hair vs skin estimate

# Coarse: obvious strands + brows
SIGMAS = (0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0, 6.5, 8.0, 10.0)
VESSEL_PERCENTILE = 90.0  # floor=0 后百分位才真正生效；50 会半图糊满
LOCAL_NORM_FLOOR = 0.0
CLOSE_KSIZE = 0  # 0 = skip close (03f → 04 不做膨胀腐蚀)
OPEN_KSIZE = 0
MIN_OBJECT_PX = 0  # 0 = skip remove_small_objects

# Vellus: finer scales, looser gate (also bright-ridge pass for lit fuzz)
SIGMAS_VELLUS = (0.6, 0.9, 1.2)
VESSEL_PERCENTILE_VELLUS = 86.0
LOCAL_NORM_FLOOR_VELLUS = 0.18
CLOSE_KSIZE_VELLUS = 3
MIN_OBJECT_PX_VELLUS = 8
VELLUS_BRIGHT_RIDGES = True  # catch highlight peach-fuzz

LOCAL_NORM_KSIZE = 101
SPECULAR_Y = 220
SPECULAR_S_MAX = 40
SKIN_CR = (133, 173)
SKIN_CB = (77, 127)
USE_SKIN_WEIGHT = False  # 关掉色度肤色权重，改用细长连通域去噪
SKIN_HARD_CUTOFF = 0.9  # 仅 USE_SKIN_WEIGHT=True 时生效
SKIN_WEIGHT_FLOOR = 0.0
SKIN_Y_MEDIAN_GATE = True
# 细长连通域过滤（用二阶矩，避免斜向细丝被 AABB 误杀）
ELONGATE_FILTER = True
ELONG_MIN_AREA = 4
ELONG_MIN_ASPECT = 3.0  # λ_max/λ_min（矩特征值比）；越大越严
ELONG_MIN_MAJOR = 6  # 近似主轴长度（px）下限
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "sources" / "exclude_crops"
OUT = Path(__file__).resolve().parent / "output"
# PS 可选颜色+黑白结果（用户提供）
EXTERNAL_PREP = OUT / "01_roi_enhancment.png"


def _find_rgba() -> Path:
    matches = sorted(SRC_DIR.glob("*_exclude_rgba.png"))
    if not matches:
        raise FileNotFoundError(f"No *_exclude_rgba.png under {SRC_DIR}")
    return matches[0]


def _load_bgr_roi(rgba_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise RuntimeError(f"Failed to read {rgba_path}")

    if rgba.ndim == 3 and rgba.shape[2] == 4:
        bgr = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3]
        roi = alpha > 0
        if not np.any(roi):
            mask_path = rgba_path.with_name(
                rgba_path.name.replace("_exclude_rgba.png", "_exclude_mask.png")
            )
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError("Alpha empty and mask fallback missing")
            roi = mask > 0
    else:
        bgr = rgba if rgba.ndim == 3 else cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
        mask_path = rgba_path.with_name(
            rgba_path.name.replace("_exclude_rgba.png", "_exclude_mask.png")
        )
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError("Need alpha or *_exclude_mask.png for ROI")
        roi = mask > 0

    return bgr, roi.astype(bool)


def _skin_prob(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    cr0, cr1 = SKIN_CR
    cb0, cb1 = SKIN_CB
    cr_c = 0.5 * (cr0 + cr1)
    cb_c = 0.5 * (cb0 + cb1)
    cr_r = max(1.0, 0.5 * (cr1 - cr0))
    cb_r = max(1.0, 0.5 * (cb1 - cb0))
    d = np.sqrt(((cr - cr_c) / cr_r) ** 2 + ((cb - cb_c) / cb_r) ** 2)
    return np.clip(1.0 - d, 0.0, 1.0).astype(np.float32)


def _skin_soft_weight(bgr: np.ndarray, roi: np.ndarray) -> np.ndarray:
    skin_prob = _skin_prob(bgr)
    weight = 1.0 - (1.0 - SKIN_WEIGHT_FLOOR) * skin_prob
    # 很像皮肤：直接剔除；可选要求 Y >= ROI 中值，避免误杀暗发
    hard = skin_prob >= SKIN_HARD_CUTOFF
    if SKIN_Y_MEDIAN_GATE and np.any(roi):
        y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        y_med = float(np.median(y[roi]))
        hard = hard & (y >= y_med)
    weight = np.where(hard, 0.0, weight)
    return weight.astype(np.float32)


def _specular_mask(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    y = ycrcb[:, :, 0]
    s = hsv[:, :, 1]
    return (y >= SPECULAR_Y) & (s <= SPECULAR_S_MAX)


def _load_external_prep(roi: np.ndarray) -> np.ndarray:
    path = EXTERNAL_PREP
    if path is None or not path.is_file():
        raise FileNotFoundError(f"EXTERNAL_PREP missing: {path}")
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Failed to read {path}")
    if im.ndim == 3:
        gray = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = im
    if gray.shape[:2] != roi.shape[:2]:
        raise ValueError(
            f"External prep size {gray.shape[:2]} != ROI {roi.shape[:2]}"
        )
    return np.where(roi, gray.astype(np.float32), 0.0)


def _prepare_frangi_input(bgr: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Build Frangi input; skin_residual ≈ professional selective-color skin suppress."""
    if PREP_MODE == "external":
        # User PS pipeline already did selective color + B&W — use as-is
        return _load_external_prep(roi)

    if PREP_MODE == "green":
        base = bgr[:, :, 1].astype(np.float32)
    else:
        base = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)

    if PREP_MODE == "skin_residual":
        skin = _skin_prob(bgr)
        local_skin = cv2.GaussianBlur(base, (0, 0), SKIN_BLUR_SIGMA)
        # positive where darker than local skin estimate (hair)
        darkness = np.maximum(local_skin - base, 0.0)
        darkness *= 0.35 + 0.65 * skin
        # bright bg + dark ridges for black_ridges=True
        prep = np.clip(210.0 - RESIDUAL_GAIN * darkness, 0, 255)
    else:
        prep = base

    u8 = np.clip(prep, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    prep_eq = clahe.apply(u8).astype(np.float32)

    spec = _specular_mask(bgr)
    if np.any(spec & roi):
        blurred = cv2.GaussianBlur(prep_eq, (0, 0), 2.0)
        prep_eq = np.where(spec, blurred, prep_eq)
    return np.where(roi, prep_eq, 0.0)


def _vesselness(
    luma: np.ndarray,
    roi: np.ndarray,
    sigmas: tuple[float, ...],
    *,
    black_ridges: bool = True,
) -> np.ndarray:
    img = luma / 255.0
    v = frangi(img, sigmas=sigmas, black_ridges=black_ridges)
    return np.where(roi, v, 0.0).astype(np.float32)


def _local_normalize(v: np.ndarray, roi: np.ndarray) -> np.ndarray:
    k = LOCAL_NORM_KSIZE | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    local_max = cv2.dilate(v, kernel)
    local_max = np.maximum(local_max, 1e-6)
    norm = v / local_max
    return np.where(roi, norm.astype(np.float32), 0.0)


def _threshold(
    v: np.ndarray,
    roi: np.ndarray,
    weight: np.ndarray,
    *,
    percentile: float,
    floor: float,
    debug_prefix: str | None = None,
) -> np.ndarray:
    """Binary threshold. If debug_prefix set, write each sub-step under OUT/."""
    local_norm = _local_normalize(v, roi)
    weighted = local_norm * weight
    scored = np.where(roi, weighted, 0.0)
    # 百分位只在未硬切掉的像素上算，避免大量 0 把 thr 拉崩
    eligible = roi & (weight > 0)
    vals = scored[eligible]
    if vals.size == 0:
        binary = np.zeros(v.shape, dtype=bool)
        thr = 0.0
    else:
        thr = max(float(np.percentile(vals, percentile)), floor)
        binary = (scored >= thr) & eligible

    if debug_prefix is not None:
        # 03a: vesselness already in 02; still dump ROI-cleared vessel for chain
        cv2.imwrite(str(OUT / f"{debug_prefix}a_vessel_in.png"), _to_u8(v, roi))
        cv2.imwrite(
            str(OUT / f"{debug_prefix}b_local_norm.png"), _to_u8(local_norm, roi)
        )
        # skin weight: 0..1 → gray (bright = keep more)
        w_u8 = np.zeros(roi.shape, dtype=np.uint8)
        w_u8[roi] = np.clip(weight[roi] * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(OUT / f"{debug_prefix}c_skin_weight.png"), w_u8)
        cv2.imwrite(
            str(OUT / f"{debug_prefix}d_scored.png"), _to_u8(scored, roi)
        )
        # pixels killed by weight (local_norm high but scored low)
        killed = (local_norm > thr) & (scored < thr) & roi if vals.size else (
            np.zeros(roi.shape, dtype=bool)
        )
        cv2.imwrite(
            str(OUT / f"{debug_prefix}e_killed_by_weight.png"),
            (killed.astype(np.uint8) * 255),
        )
        cv2.imwrite(
            str(OUT / f"{debug_prefix}f_binary.png"),
            (binary.astype(np.uint8) * 255),
        )
        print(
            f"thresh debug {debug_prefix}: pct={percentile} floor={floor} "
            f"thr={thr:.6g} pass={int(binary.sum())}/{int(roi.sum())}"
        )

    return binary


def _morph(
    binary: np.ndarray,
    *,
    close_ksize: int,
    open_ksize: int,
    min_object_px: int,
) -> np.ndarray:
    binary_out = binary.astype(bool)
    if close_ksize >= 2:
        u8 = binary_out.astype(np.uint8) * 255
        close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, close_k)
        binary_out = u8 > 0
    if open_ksize >= 2:
        u8 = binary_out.astype(np.uint8) * 255
        open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, open_k)
        binary_out = u8 > 0
    if min_object_px > 0:
        binary_out = remove_small_objects(
            binary_out, max_size=max(0, min_object_px - 1)
        )
    return binary_out


def _filter_elongated(binary: np.ndarray) -> np.ndarray:
    """Keep thin/long CCs via central-moment eigenvalues (rotation-invariant).

    AABB aspect fails on diagonal strands (box looks square). Moment ratio
    λ_max/λ_min stays high for thin lines at any angle.
    """
    u8 = binary.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    keep = np.zeros(binary.shape, dtype=bool)
    kept = 0
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < ELONG_MIN_AREA:
            continue
        comp = (labels == i).astype(np.uint8)
        m = cv2.moments(comp, binaryImage=True)
        if m["m00"] < 1e-6:
            continue
        mu20 = m["mu20"] / m["m00"]
        mu02 = m["mu02"] / m["m00"]
        mu11 = m["mu11"] / m["m00"]
        # covariance eigenvalues
        disc = np.sqrt(max((mu20 - mu02) ** 2 + 4.0 * mu11 * mu11, 0.0))
        lam_max = 0.5 * (mu20 + mu02 + disc)
        lam_min = 0.5 * (mu20 + mu02 - disc)
        aspect = lam_max / max(lam_min, 1e-12)
        # major ≈ 4*sqrt(lam_max) for a thin line segment scale
        major = 4.0 * np.sqrt(max(lam_max, 0.0))
        if major >= ELONG_MIN_MAJOR and aspect >= ELONG_MIN_ASPECT:
            keep[comp.astype(bool)] = True
            kept += 1
    print(
        f"elongate filter(moments): comps={n_labels - 1} kept={kept} "
        f"px {int(binary.sum())}→{int(keep.sum())}"
    )
    return keep


def _to_u8(v: np.ndarray, roi: np.ndarray) -> np.ndarray:
    out = np.zeros(v.shape, dtype=np.uint8)
    if not np.any(roi):
        return out
    vmax = max(float(v[roi].max()), 1e-8)
    out[roi] = np.clip(v[roi] / vmax * 255.0, 0, 255).astype(np.uint8)
    return out


def _overlay(
    bgr: np.ndarray,
    coarse: np.ndarray,
    vellus: np.ndarray,
    roi: np.ndarray,
) -> np.ndarray:
    """Red = coarse strands, green = vellus-only, yellow = both."""
    out = bgr.copy()
    both = coarse & vellus
    only_c = coarse & ~vellus
    only_v = vellus & ~coarse
    tint = out.copy()
    tint[only_c] = (0, 0, 255)
    tint[only_v] = (0, 255, 0)
    tint[both] = (0, 255, 255)
    blended = cv2.addWeighted(out, 0.45, tint, 0.55, 0)
    blended[~roi] = bgr[~roi]
    return blended


def _branch(
    luma: np.ndarray,
    roi: np.ndarray,
    weight: np.ndarray,
    spec: np.ndarray,
    *,
    sigmas: tuple[float, ...],
    percentile: float,
    floor: float,
    close_ksize: int,
    open_ksize: int,
    min_object_px: int,
    bright_ridges: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # PS selective-color+B&W here is bright skin + dark hair → black_ridges
    vessel = _vesselness(luma, roi, sigmas, black_ridges=True)
    if bright_ridges:
        vessel = np.maximum(
            vessel, _vesselness(luma, roi, sigmas, black_ridges=False)
        )
    vessel = np.where(spec, 0.0, vessel)
    thresh = _threshold(
        vessel,
        roi,
        weight,
        percentile=percentile,
        floor=floor,
        debug_prefix=None,
    )
    mask = _morph(
        thresh,
        close_ksize=close_ksize,
        open_ksize=open_ksize,
        min_object_px=min_object_px,
    )
    if ELONGATE_FILTER:
        mask = _filter_elongated(mask)
    mask &= roi
    return vessel, thresh, mask


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rgba_path = _find_rgba()
    bgr, roi = _load_bgr_roi(rgba_path)

    roi_vis = bgr.copy()
    roi_vis[~roi] = 0
    cv2.imwrite(str(OUT / "01_roi.png"), roi_vis)

    frangi_in = _prepare_frangi_input(bgr, roi)
    cv2.imwrite(
        str(OUT / "01b_frangi_input.png"),
        np.clip(frangi_in, 0, 255).astype(np.uint8),
    )
    if PREP_MODE == "external" or not USE_SKIN_WEIGHT:
        weight = np.where(roi, 1.0, 0.0).astype(np.float32)
    else:
        weight = _skin_soft_weight(bgr, roi)
        weight = np.where(roi, weight, 0.0)
    if PREP_MODE == "external":
        spec = np.zeros(roi.shape, dtype=bool)
    else:
        spec = _specular_mask(bgr)

    vessel_c, thresh_c, mask_c = _branch(
        frangi_in,
        roi,
        weight,
        spec,
        sigmas=SIGMAS,
        percentile=VESSEL_PERCENTILE,
        floor=LOCAL_NORM_FLOOR,
        close_ksize=CLOSE_KSIZE,
        open_ksize=OPEN_KSIZE,
        min_object_px=MIN_OBJECT_PX,
    )
    # Dump 03_* sub-steps on coarse vesselness (what becomes 03_thresh)
    _threshold(
        vessel_c,
        roi,
        weight,
        percentile=VESSEL_PERCENTILE,
        floor=LOCAL_NORM_FLOOR,
        debug_prefix="03",
    )

    if INCLUDE_VELLUS:
        vessel_v, thresh_v, mask_v = _branch(
            frangi_in,
            roi,
            weight,
            spec,
            sigmas=SIGMAS_VELLUS,
            percentile=VESSEL_PERCENTILE_VELLUS,
            floor=LOCAL_NORM_FLOOR_VELLUS,
            close_ksize=CLOSE_KSIZE_VELLUS,
            open_ksize=0,
            min_object_px=MIN_OBJECT_PX_VELLUS,
            bright_ridges=VELLUS_BRIGHT_RIDGES,
        )
    else:
        vessel_v = np.zeros_like(vessel_c)
        thresh_v = np.zeros(roi.shape, dtype=bool)
        mask_v = np.zeros(roi.shape, dtype=bool)

    vessel = np.maximum(vessel_c, vessel_v)
    cv2.imwrite(str(OUT / "02_vesselness.png"), _to_u8(vessel, roi))
    cv2.imwrite(str(OUT / "02b_vesselness_vellus.png"), _to_u8(vessel_v, roi))

    thresh = thresh_c | thresh_v
    cv2.imwrite(str(OUT / "03_thresh.png"), (thresh.astype(np.uint8) * 255))
    cv2.imwrite(
        str(OUT / "03b_thresh_vellus.png"), (thresh_v.astype(np.uint8) * 255)
    )

    # 04a = 阈值后、细长过滤前；04 = 过滤后
    raw_union = (thresh_c | thresh_v) & roi
    cv2.imwrite(str(OUT / "04a_before_elong.png"), (raw_union.astype(np.uint8) * 255))
    mask = mask_c | mask_v
    mask &= roi
    cv2.imwrite(str(OUT / "04_mask.png"), (mask.astype(np.uint8) * 255))
    cv2.imwrite(str(OUT / "04b_mask_coarse.png"), (mask_c.astype(np.uint8) * 255))
    cv2.imwrite(str(OUT / "04c_mask_vellus.png"), (mask_v.astype(np.uint8) * 255))
    cv2.imwrite(str(OUT / "05_overlay.png"), _overlay(bgr, mask_c, mask_v, roi))

    print(f"input:  {rgba_path.relative_to(ROOT)}")
    print(f"output: {OUT.relative_to(ROOT)}")
    print(
        f"prep={PREP_MODE} skin_w={USE_SKIN_WEIGHT} elong={ELONGATE_FILTER} "
        f"pct={VESSEL_PERCENTILE} vellus={INCLUDE_VELLUS} "
        f"raw={int(raw_union.sum())} coarse={int(mask_c.sum())} "
        f"vellus_px={int(mask_v.sum())} union={int(mask.sum())}"
    )


if __name__ == "__main__":
    main()
