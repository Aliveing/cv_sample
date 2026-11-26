"""Extract hair mask: visible strands + optional vellus via dual Frangi branches."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hessian_matrix
from skimage.filters import frangi, meijering, sato
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
# 补梢策略（Meijering/Sato+分尺度 OR 噪大、精度不如纯 Frangi → 默认 off）:
#   "off" | "tangent" | "meijering_sato" | "per_scale" | "both"
TIP_MODE = "tangent"  # 方案3：分尺度阈值再 OR（纯 Frangi，不加 Meijering/Sato）
# 方式2：Meijering / Sato 与 Frangi 取 max（管端更温和，易噪）
# 方式3：小 σ / 大 σ 分百分位阈值后再 OR（易噪）
PER_SCALE_FINE_MAX = 2.0  # σ ≤ 此值用细尺度百分位
PER_SCALE_FINE_PCT = 90.0  # 与主阈值对齐，避免细 σ 灌噪声
PER_SCALE_COARSE_PCT = 92.0
# 方式1（保留开关，默认关）
TANGENT_TIP_EXTEND = False  # 仅 TIP_MODE=="tangent" 时走
TANGENT_SIGMA = 1.5
TANGENT_SEED_PCT = 90.0
TANGENT_MAX_STEPS = 20
TANGENT_DARK_MARGIN = 2.5
TANGENT_MIN_VESSEL = 0.75
TANGENT_HARD_BORDER = 2
TANGENT_SOFT_BORDER = 14
TANGENT_EDGE_ALIGN = 0.75
TANGENT_ONLY_ENDPOINTS = True
CLOSE_KSIZE = 3  # bridge 1–2px Frangi pinch-offs at crossings
OPEN_KSIZE = 0
MIN_OBJECT_PX = 0  # 0 = skip remove_small_objects

# Black-hat parallel path: dark lines without tubular (junction) suppression
USE_TOPHAT_BRIDGE = True
TOPHAT_KSIZE = 7  # ellipse; ~hair width + margin
TOPHAT_PERCENTILE = 88.0  # 原 94.0(thr=34) 切太狠；靠 TOPHAT_NEAR_PX 控噪
TOPHAT_FLOOR = 4.0  # absolute blackhat floor (0–255)
TOPHAT_NEAR_PX = 5  # 0 = global; >0 only keep tophat within dilate(Frangi)

# Vellus: finer scales, looser gate (also bright-ridge pass for lit fuzz)
SIGMAS_VELLUS = (0.6, 0.9, 1.2)
VESSEL_PERCENTILE_VELLUS = 86.0
LOCAL_NORM_FLOOR_VELLUS = 0.18
CLOSE_KSIZE_VELLUS = 3
MIN_OBJECT_PX_VELLUS = 8
VELLUS_BRIGHT_RIDGES = True  # catch highlight peach-fuzz

# 原始 vessel 绝对下限（0–255，相对 ROI max）。百分位仍在全 ROI 上算，此门只删弱纹理
VESSEL_ABS_FLOOR = 10.0  # 参考 03a p90≈10；0 = 关闭
LOCAL_NORM_KSIZE = 31  # 原 101：核太大 → 线旁弱尾被抬满成光晕
LOCAL_NORM_POWER = 1.35  # >1 压中灰，减轻泛白光晕
# 压 ROI 黑边假脊：外面填边缘中值再跑 Hessian；scored 再按距边衰减
ROI_EDGE_FILL = True
BORDER_DAMP_PX = 5  # 仅 Frangi 分支：dist < 此 → 线性压到 0（原 14 太宽，压掉贴边发丝）
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
ELONG_MIN_ASPECT = 2.5  # λ_max/λ_min；成簇弯曲发会把 ratio 压低，略放宽
ELONG_MIN_MAJOR = 6  # 近似主轴长度（px）下限
# 大块发丝/发簇：aspect 不够细长也保留（04a→04b 曾误杀角上明显大毛）
ELONG_KEEP_AREA = 120
ELONG_KEEP_MAJOR = 24
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "sources" / "exclude_crops"
OUT_ROOT = Path(__file__).resolve().parent / "output"
OUT = OUT_ROOT  # updated per sample in _run_one
# PS 可选颜色+黑白结果（用户提供）
EXTERNAL_PREP = OUT_ROOT / "01_roi_enhancment.png"


def _find_all_rgba() -> list[Path]:
    matches = sorted(SRC_DIR.glob("*_exclude_rgba.png"))
    if not matches:
        raise FileNotFoundError(f"No *_exclude_rgba.png under {SRC_DIR}")
    return matches


def _find_rgba() -> Path:
    return _find_all_rgba()[0]


def _sample_name(rgba_path: Path) -> str:
    """Short folder name like '103964-manual6' from '<id>-<ts>-<manual>-...'."""
    stem = rgba_path.name.replace("_exclude_rgba.png", "")
    parts = stem.split("-")
    if len(parts) >= 3:
        return f"{parts[0]}-{parts[2]}"
    return stem


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
    prep_eq = np.where(roi, prep_eq, 0.0)
    if ROI_EDGE_FILL:
        prep_eq = _fill_outside_roi(prep_eq, roi)
    return prep_eq


def _fill_outside_roi(prep: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Replace black outside with ROI-edge median so Hessian doesn't see a hard step."""
    if not np.any(roi):
        return prep
    dist = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 5)
    ring = roi & (dist <= 3.0)
    if not np.any(ring):
        ring = roi
    fill = float(np.median(prep[ring]))
    out = prep.copy()
    out[~roi] = fill
    return out


def _border_damp(roi: np.ndarray) -> np.ndarray:
    """0 at silhouette → 1 by BORDER_DAMP_PX inside."""
    if BORDER_DAMP_PX <= 0:
        return np.where(roi, 1.0, 0.0).astype(np.float32)
    dist = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(dist / float(BORDER_DAMP_PX), 0.0, 1.0).astype(np.float32)


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


def _norm_roi(v: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Scale ROI values to [0,1] so different ridge filters are comparable."""
    out = np.zeros(v.shape, dtype=np.float32)
    if not np.any(roi):
        return out
    vals = v[roi]
    mx = float(vals.max())
    if mx <= 1e-12:
        return out
    out[roi] = (vals / mx).astype(np.float32)
    return out


def _ridge_stack(
    luma: np.ndarray,
    roi: np.ndarray,
    sigmas: tuple[float, ...],
    *,
    black_ridges: bool = True,
    use_meijering_sato: bool = False,
) -> np.ndarray:
    """Frangi (+ optional Meijering/Sato max). Each filter ROI-normalized first."""
    img = (luma / 255.0).astype(np.float64)
    stack = [_norm_roi(
        np.where(roi, frangi(img, sigmas=sigmas, black_ridges=black_ridges), 0.0).astype(
            np.float32
        ),
        roi,
    )]
    if use_meijering_sato:
        stack.append(
            _norm_roi(
                np.where(
                    roi,
                    meijering(img, sigmas=sigmas, black_ridges=black_ridges),
                    0.0,
                ).astype(np.float32),
                roi,
            )
        )
        stack.append(
            _norm_roi(
                np.where(
                    roi, sato(img, sigmas=sigmas, black_ridges=black_ridges), 0.0
                ).astype(np.float32),
                roi,
            )
        )
    out = stack[0]
    for s in stack[1:]:
        out = np.maximum(out, s)
    return out


def _per_scale_thresh(
    luma: np.ndarray,
    roi: np.ndarray,
    weight: np.ndarray,
    spec: np.ndarray,
    sigmas: tuple[float, ...],
    *,
    use_meijering_sato: bool,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold each σ alone (fine→lower pct), OR masks; vessel = max response."""
    vessel_max = np.zeros(roi.shape, dtype=np.float32)
    mask_or = np.zeros(roi.shape, dtype=bool)
    fine_n = coarse_n = 0
    for sigma in sigmas:
        v = _ridge_stack(
            luma,
            roi,
            (float(sigma),),
            black_ridges=True,
            use_meijering_sato=use_meijering_sato,
        )
        v = np.where(spec, 0.0, v)
        vessel_max = np.maximum(vessel_max, v)
        pct = (
            PER_SCALE_FINE_PCT
            if float(sigma) <= PER_SCALE_FINE_MAX
            else PER_SCALE_COARSE_PCT
        )
        if float(sigma) <= PER_SCALE_FINE_MAX:
            fine_n += 1
        else:
            coarse_n += 1
        mask_or |= _threshold(v, roi, weight, percentile=pct, floor=floor)
    print(
        f"per_scale: sigmas={len(sigmas)} fine(≤{PER_SCALE_FINE_MAX})={fine_n} "
        f"@{PER_SCALE_FINE_PCT} coarse={coarse_n} @{PER_SCALE_COARSE_PCT} "
        f"or_px={int(mask_or.sum())}"
    )
    return vessel_max, mask_or


def _ridge_tangent_field(luma: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangent along dark ridge = eigenvector of smaller |Hessian eigenvalue|."""
    img = (luma / 255.0).astype(np.float64)
    # order='rc' → (Hrr, Hrc, Hcc); row=y, col=x
    Hrr, Hrc, Hcc = hessian_matrix(
        img, sigma=sigma, order="rc", use_gaussian_derivatives=True
    )
    a, b, c = Hrr, Hrc, Hcc
    disc = np.sqrt(np.maximum((a - c) ** 2 + 4.0 * b * b, 0.0))
    # eigenvalues; ridge direction ↔ eigenvalue closer to 0
    l1 = 0.5 * (a + c + disc)
    l2 = 0.5 * (a + c - disc)
    use_l2 = np.abs(l2) <= np.abs(l1)
    lam = np.where(use_l2, l2, l1)
    # eigenvector for λ: (b, λ-a); fallback (λ-c, b)
    vx = np.where(np.abs(b) > 1e-12, b, lam - c)
    vy = np.where(np.abs(b) > 1e-12, lam - a, b)
    n = np.hypot(vx, vy)
    n = np.maximum(n, 1e-12)
    # skimage rc: vy along rows, vx along cols
    return (vx / n).astype(np.float32), (vy / n).astype(np.float32)


def _extend_tips_along_tangent(
    vessel: np.ndarray,
    luma: np.ndarray,
    roi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk tangent from ridge endpoints; return (boosted_vessel, lift_mask)."""
    lift_mask = np.zeros(vessel.shape, dtype=bool)
    if not np.any(roi):
        return vessel, lift_mask

    vx, vy = _ridge_tangent_field(luma, TANGENT_SIGMA)
    local_mean = cv2.GaussianBlur(luma, (0, 0), 3.0)

    dist = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 5)
    gy, gx = np.gradient(dist.astype(np.float32))
    gn = np.maximum(np.hypot(gx, gy), 1e-6)
    etx = (-gy / gn).astype(np.float32)
    ety = (gx / gn).astype(np.float32)

    vals = vessel[roi]
    seed_thr = max(float(np.percentile(vals, TANGENT_SEED_PCT)), 1e-8)
    # allow seeds near hairline; only exclude 2px hard border
    strong = (vessel >= seed_thr) & roi & (dist >= float(TANGENT_HARD_BORDER))
    ys, xs = np.where(strong)
    if ys.size == 0:
        return vessel, lift_mask

    walks: list[tuple[int, int, float]] = []
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        tx, ty = float(vx[y0, x0]), float(vy[y0, x0])
        tn = max(np.hypot(tx, ty), 1e-12)
        tx, ty = tx / tn, ty / tn
        for sign in (-1.0, 1.0):
            xi = int(round(x0 + sign * tx))
            yi = int(round(y0 + sign * ty))
            if not (0 <= xi < vessel.shape[1] and 0 <= yi < vessel.shape[0] and roi[yi, xi]):
                ahead_weak = True
            else:
                ahead_weak = float(vessel[yi, xi]) < seed_thr * 0.5
            if TANGENT_ONLY_ENDPOINTS and not ahead_weak:
                continue
            walks.append((y0, x0, sign))

    if len(walks) > 16000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(walks), size=16000, replace=False)
        walks = [walks[i] for i in idx]

    out = vessel.copy()
    h, w = vessel.shape
    lifted = 0
    stopped_edge = 0

    for y0, x0, sign in walks:
        x, y = float(x0), float(y0)
        dx = float(vx[y0, x0]) * sign
        dy = float(vy[y0, x0]) * sign
        sn = max(np.hypot(dx, dy), 1e-12)
        dx, dy = dx / sn, dy / sn
        seed_v = float(vessel[y0, x0])
        for _ in range(TANGENT_MAX_STEPS):
            x += dx
            y += dy
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or yi < 0 or xi >= w or yi >= h or not roi[yi, xi]:
                break
            d = float(dist[yi, xi])
            if d < float(TANGENT_HARD_BORDER):
                stopped_edge += 1
                break
            # soft border: only block silhouette-parallel walks
            if d < float(TANGENT_SOFT_BORDER):
                if abs(dx * float(etx[yi, xi]) + dy * float(ety[yi, xi])) >= TANGENT_EDGE_ALIGN:
                    stopped_edge += 1
                    break
            if float(luma[yi, xi]) > float(local_mean[yi, xi]) - TANGENT_DARK_MARGIN:
                break

            ndx, ndy = float(vx[yi, xi]), float(vy[yi, xi])
            dot = ndx * dx + ndy * dy
            if abs(dot) < 0.5:
                ndx, ndy = -ndx, -ndy
                dot = ndx * dx + ndy * dy
                if abs(dot) < 0.5:
                    break
            dx = 0.65 * dx + 0.35 * ndx * np.sign(dot + 1e-12)
            dy = 0.65 * dy + 0.35 * ndy * np.sign(dot + 1e-12)
            sn = max(np.hypot(dx, dy), 1e-12)
            dx, dy = dx / sn, dy / sn

            target = seed_v * TANGENT_MIN_VESSEL
            if out[yi, xi] < target:
                out[yi, xi] = target
                lift_mask[yi, xi] = True
                lifted += 1

    print(
        f"tangent tip extend: walks={len(walks)} lifts={lifted} "
        f"edge_stops={stopped_edge} hard={TANGENT_HARD_BORDER} soft={TANGENT_SOFT_BORDER}"
    )
    return out, lift_mask


def _local_normalize(v: np.ndarray, roi: np.ndarray) -> np.ndarray:
    k = LOCAL_NORM_KSIZE | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # outside→copy nearest ROI max so dilate near border isn't crushed by zeros
    v_fill = v.copy()
    if np.any(roi):
        # seed outside with local dilate of ROI-only values (zeros outside first)
        v_roi = np.where(roi, v, 0.0).astype(np.float32)
        local_max0 = cv2.dilate(v_roi, kernel)
        v_fill = np.where(roi, v, local_max0)
    local_max = cv2.dilate(v_fill.astype(np.float32), kernel)
    local_max = np.maximum(local_max, 1e-6)
    norm = v / local_max
    if LOCAL_NORM_POWER != 1.0:
        norm = np.power(np.clip(norm, 0.0, 1.0), LOCAL_NORM_POWER)
    return np.where(roi, norm.astype(np.float32), 0.0)


def _abs_vessel_gate(v: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """True where raw vessel ≥ VESSEL_ABS_FLOOR (0–255 scale vs ROI max)."""
    if VESSEL_ABS_FLOOR <= 0 or not np.any(roi):
        return np.ones(v.shape, dtype=bool)
    vmax = max(float(v[roi].max()), 1e-8)
    return (v / vmax * 255.0) >= float(VESSEL_ABS_FLOOR)


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
    damp = _border_damp(roi)
    weighted = local_norm * weight * damp
    scored = np.where(roi, weighted, 0.0)
    # 百分位只在未硬切掉的像素上算，避免大量 0 把 thr 拉崩
    eligible = roi & (weight > 0)
    vals = scored[eligible]
    # 绝对下限：原始 vessel（相对 ROI max，0–255）低于此值不算头发，防止归一化把弱纹理抬过阈值
    abs_gate = _abs_vessel_gate(v, roi)
    if vals.size == 0:
        binary = np.zeros(v.shape, dtype=bool)
        thr = 0.0
    else:
        thr = max(float(np.percentile(vals, percentile)), floor)
        binary = (scored >= thr) & eligible & abs_gate

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


def _tophat_dark_mask(
    luma: np.ndarray,
    roi: np.ndarray,
    weight: np.ndarray,
    spec: np.ndarray,
    *,
    ksize: int,
    percentile: float,
    floor: float,
    debug_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Black-hat dark strokes (no tubular prior → crossings stay bright)."""
    k = max(3, ksize | 1)
    u8 = np.clip(luma, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    # 黑帽输入已在 ROI 外用边缘中值填充，无需边界衰减（否则压掉真实贴边发丝）
    scored = np.where(roi, bh * weight, 0.0)
    scored = np.where(spec, 0.0, scored)
    eligible = roi & (weight > 0) & ~spec
    vals = scored[eligible]
    if vals.size == 0:
        binary = np.zeros(roi.shape, dtype=bool)
        thr = 0.0
    else:
        thr = max(float(np.percentile(vals, percentile)), float(floor))
        binary = (scored >= thr) & eligible
    if debug_prefix is not None:
        cv2.imwrite(str(OUT / f"{debug_prefix}_1_bh_raw.png"), _to_u8(bh, roi))
        cv2.imwrite(str(OUT / f"{debug_prefix}_2_scored.png"), _to_u8(scored, roi))
        cv2.imwrite(
            str(OUT / f"{debug_prefix}_3_eligible.png"),
            (eligible.astype(np.uint8) * 255),
        )
        cv2.imwrite(
            str(OUT / f"{debug_prefix}_4_thresh_before_near.png"),
            (binary.astype(np.uint8) * 255),
        )
        print(
            f"tophat debug {debug_prefix}: pct={percentile} floor={floor} "
            f"thr={thr:.3f} pass={int(binary.sum())}/{int(eligible.sum())}"
        )
    return scored, binary


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
    λ_max/λ_min stays high for thin lines at any angle. Large clumps/curves
    can have low aspect — also keep if area+major clear the keep-* gates.
    """
    u8 = binary.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    keep = np.zeros(binary.shape, dtype=bool)
    killed = np.zeros(binary.shape, dtype=bool)
    kept = 0
    kept_large = 0
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
        elongated = major >= ELONG_MIN_MAJOR and aspect >= ELONG_MIN_ASPECT
        large = area >= ELONG_KEEP_AREA and major >= ELONG_KEEP_MAJOR
        if elongated or large:
            keep[comp.astype(bool)] = True
            kept += 1
            if large and not elongated:
                kept_large += 1
        else:
            killed[comp.astype(bool)] = True
    print(
        f"elongate filter(moments): comps={n_labels - 1} kept={kept} "
        f"(via_large={kept_large}) px {int(binary.sum())}→{int(keep.sum())}"
    )
    cv2.imwrite(str(OUT / "04d_elong_killed.png"), (killed.astype(np.uint8) * 255))
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
    save_tip_debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mode = TIP_MODE
    use_ms = mode in ("meijering_sato", "both")
    use_ps = mode in ("per_scale", "both")
    use_tangent = mode == "tangent" or (mode == "off" and TANGENT_TIP_EXTEND)

    # Frangi-only baseline for debug delta
    frangi_only = _vesselness(luma, roi, sigmas, black_ridges=True)
    if bright_ridges:
        frangi_only = np.maximum(
            frangi_only, _vesselness(luma, roi, sigmas, black_ridges=False)
        )
    frangi_only = np.where(spec, 0.0, frangi_only)

    if use_ps:
        vessel, thresh = _per_scale_thresh(
            luma,
            roi,
            weight,
            spec,
            sigmas,
            use_meijering_sato=use_ms,
            floor=floor,
        )
        if bright_ridges:
            v_bright = _ridge_stack(
                luma, roi, sigmas, black_ridges=False, use_meijering_sato=use_ms
            )
            v_bright = np.where(spec, 0.0, v_bright)
            vessel = np.maximum(vessel, v_bright)
            thresh = thresh | _threshold(
                v_bright, roi, weight, percentile=percentile, floor=floor
            )
    else:
        vessel = _ridge_stack(
            luma,
            roi,
            sigmas,
            black_ridges=True,
            use_meijering_sato=use_ms,
        )
        if bright_ridges:
            vessel = np.maximum(
                vessel,
                _ridge_stack(
                    luma,
                    roi,
                    sigmas,
                    black_ridges=False,
                    use_meijering_sato=use_ms,
                ),
            )
        vessel = np.where(spec, 0.0, vessel)
        if use_tangent:
            vessel, _tip_lift = _extend_tips_along_tangent(vessel, luma, roi)
        thresh = _threshold(
            vessel,
            roi,
            weight,
            percentile=percentile,
            floor=floor,
            debug_prefix=None,
        )

    if save_tip_debug:
        base = _norm_roi(frangi_only, roi)
        enh = _norm_roi(vessel, roi)
        delta = np.maximum(enh - base, 0.0)
        gain = (enh > base * 1.05) & (enh > 0.05) & roi
        cv2.imwrite(str(OUT / "02a_vessel_frangi_only.png"), _to_u8(base, roi))
        cv2.imwrite(str(OUT / "02b_tip_delta.png"), _to_u8(delta, roi))
        tip_vis = np.zeros((*roi.shape, 3), dtype=np.uint8)
        tip_vis[roi] = _to_u8(enh, roi)[roi, None].repeat(3, axis=1)
        tip_vis[gain] = (0, 255, 255)  # cyan = stronger than Frangi-only
        cv2.imwrite(str(OUT / "02c_tip_lift_overlay.png"), tip_vis)
        # Frangi-only thresh vs new thresh difference
        thr_f = _threshold(
            frangi_only, roi, weight, percentile=percentile, floor=floor
        )
        added = thresh & ~thr_f & roi
        lost = thr_f & ~thresh & roi
        cmp = np.zeros((*roi.shape, 3), dtype=np.uint8)
        cmp[roi] = 40
        cmp[thr_f & thresh] = (200, 200, 200)
        cmp[added] = (0, 255, 0)  # green = newly recovered
        cmp[lost] = (0, 0, 255)  # red = lost vs frangi
        cv2.imwrite(str(OUT / "02d_thresh_delta.png"), cmp)
        print(
            f"tip_mode={mode} gain_px={int(gain.sum())} "
            f"thresh_add={int(added.sum())} thresh_lost={int(lost.sum())}"
        )

    # Bridge Frangi junction dropouts with black-hat dark lines
    thresh_frangi = thresh.copy()
    tophat_added = np.zeros(roi.shape, dtype=bool)
    if USE_TOPHAT_BRIDGE:
        bh, th_bin = _tophat_dark_mask(
            luma,
            roi,
            weight,
            spec,
            ksize=TOPHAT_KSIZE,
            percentile=TOPHAT_PERCENTILE,
            floor=TOPHAT_FLOOR,
            debug_prefix="02e" if save_tip_debug else None,
        )
        if TOPHAT_NEAR_PX > 0:
            k = max(3, int(TOPHAT_NEAR_PX) * 2 + 1)
            near = cv2.dilate(
                thresh_frangi.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            ) > 0
            th_bin = th_bin & near
        tophat_added = th_bin & ~thresh_frangi & roi
        thresh = thresh_frangi | th_bin
        if save_tip_debug:
            cv2.imwrite(str(OUT / "02e_tophat.png"), _to_u8(bh, roi))
            cv2.imwrite(
                str(OUT / "02f_tophat_binary.png"),
                (th_bin.astype(np.uint8) * 255),
            )
            cv2.imwrite(
                str(OUT / "02g_tophat_added.png"),
                (tophat_added.astype(np.uint8) * 255),
            )
            print(
                f"tophat bridge: k={TOPHAT_KSIZE} pct={TOPHAT_PERCENTILE} "
                f"near={TOPHAT_NEAR_PX} add_px={int(tophat_added.sum())} "
                f"union={int(thresh.sum())}"
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
    return vessel, thresh, mask, tophat_added


def _run_one(rgba_path: Path, out_dir: Path) -> np.ndarray:
    global OUT
    OUT = out_dir
    OUT.mkdir(parents=True, exist_ok=True)
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

    vessel_c, thresh_c, mask_c, tophat_c = _branch(
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
        save_tip_debug=True,
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
        vessel_v, thresh_v, mask_v, _tophat_v = _branch(
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
    overlay = _overlay(bgr, mask_c, mask_v, roi)
    cv2.imwrite(str(OUT / "05_overlay.png"), overlay)
    # Cyan = survived tophat pixels that Frangi thresh missed (crossing bridge cue)
    if np.any(tophat_c):
        bridge_vis = overlay.copy()
        kept_bridge = tophat_c & mask_c & roi
        bridge_vis[kept_bridge] = (255, 255, 0)  # cyan in BGR
        cv2.imwrite(str(OUT / "05b_tophat_bridge_overlay.png"), bridge_vis)

    print(f"input:  {rgba_path.relative_to(ROOT)}")
    print(f"output: {OUT.relative_to(ROOT)}")
    print(
        f"prep={PREP_MODE} tip={TIP_MODE} skin_w={USE_SKIN_WEIGHT} "
        f"elong={ELONGATE_FILTER} close={CLOSE_KSIZE} "
        f"tophat={USE_TOPHAT_BRIDGE} pct={VESSEL_PERCENTILE} "
        f"vellus={INCLUDE_VELLUS} raw={int(raw_union.sum())} "
        f"coarse={int(mask_c.sum())} vellus_px={int(mask_v.sum())} "
        f"union={int(mask.sum())} tophat_kept={int((tophat_c & mask_c).sum())}"
    )
    _save_sample_grid_3x3(OUT)
    return overlay


def _make_grid_3x3(
    tiles: list[tuple[str, np.ndarray]],
    out_path: Path,
    *,
    cell_w: int = 1200,
) -> None:
    """Pack up to 9 (label, img) tiles into a crisp 3x3 PNG without heavy compression."""
    if not tiles:
        return
    # Match cell aspect to the first tile so images aren't distorted
    h0, w0 = tiles[0][1].shape[:2]
    cell_h = max(200, int(round(cell_w * h0 / max(w0, 1))))
    bar_h = 44
    pad = 8
    cols, rows = 3, 3
    canvas = np.full(
        (rows * (cell_h + bar_h) + (rows + 1) * pad, cols * cell_w + (cols + 1) * pad, 3),
        18,
        dtype=np.uint8,
    )
    for idx, (label, img) in enumerate(tiles[:9]):
        r, c = divmod(idx, cols)
        y0 = pad + r * (cell_h + bar_h + pad)
        x0 = pad + c * (cell_w + pad)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img[:, :, :3]
        ih, iw = bgr.shape[:2]
        scale = min(cell_w / max(iw, 1), cell_h / max(ih, 1))
        nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(bgr, (nw, nh), interpolation=interp)
        # Header bar
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w, y0 + bar_h), (36, 36, 36), -1)
        cv2.putText(
            canvas,
            label,
            (x0 + 12, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        # Center image inside cell
        ox = x0 + (cell_w - nw) // 2
        oy = y0 + bar_h + (cell_h - nh) // 2
        canvas[oy : oy + nh, ox : ox + nw] = resized
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _save_sample_grid_3x3(out_dir: Path) -> None:
    stage_files = [
        ("1. 01_roi", "01_roi.png"),
        ("2. 01b_frangi_in", "01b_frangi_input.png"),
        ("3. 02_vesselness", "02_vesselness.png"),
        ("4. 02e_tophat", "02e_tophat.png"),
        ("5. 02f_tophat_bin", "02f_tophat_binary.png"),
        ("6. 03d_scored", "03d_scored.png"),
        ("7. 03_thresh", "03_thresh.png"),
        ("8. 04_mask", "04_mask.png"),
        ("9. 05_overlay", "05_overlay.png"),
    ]
    tiles: list[tuple[str, np.ndarray]] = []
    for label, fname in stage_files:
        im = cv2.imread(str(out_dir / fname), cv2.IMREAD_UNCHANGED)
        if im is not None:
            tiles.append((label, im))
    if tiles:
        _make_grid_3x3(tiles, out_dir / "00_grid_3x3.png", cell_w=1200)


def _save_per_step_grids(
    sample_dirs: list[tuple[str, Path]],
    out_root: Path,
    *,
    cell_w: int = 1200,
) -> list[str]:
    """For every step PNG produced in per-sample folders, stitch a 3x3 grid at out_root."""
    step_names: set[str] = set()
    for _, d in sample_dirs:
        for f in d.glob("*.png"):
            if f.name != "00_grid_3x3.png":
                step_names.add(f.name)
    written: list[str] = []
    for fname in sorted(step_names):
        tiles: list[tuple[str, np.ndarray]] = []
        for idx, (name, d) in enumerate(sample_dirs, 1):
            p = d / fname
            if not p.is_file():
                continue
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is not None:
                tiles.append((f"{idx}. {name}", im))
        if tiles:
            _make_grid_3x3(tiles, out_root / fname, cell_w=cell_w)
            written.append(fname)
    return written


def main() -> None:
    paths = _find_all_rgba()
    print(f"Batch running {len(paths)} samples → {OUT_ROOT.relative_to(ROOT)}/<sample>/")
    sample_dirs: list[tuple[str, Path]] = []
    for idx, p in enumerate(paths, 1):
        name = _sample_name(p)
        print(f"\n[{idx}/{len(paths)}] === {name} ===")
        out_dir = OUT_ROOT / name
        _run_one(p, out_dir)
        sample_dirs.append((name, out_dir))
    written = _save_per_step_grids(sample_dirs, OUT_ROOT, cell_w=1200)
    print(f"\nSaved {len(written)} per-step 3x3 grids → {OUT_ROOT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
