"""Auto-generate X-AnyLabeling `skin` polygon annotations inside `exclude` regions.

Supports three MagicMirror lighting modalities:
- manual4 (m4): UV light (blue skin + spots, fluorescence, purple-light regions)
- manual6 (m6): Cross-polarized light (normal skin + spots, sensitive redness)
- manual7 (m7): Parallel-polarized light (normal skin + bright specular highlights)

Outputs new X-AnyLabeling JSONs and cropped visualization overlays under `output/`.
"""

from __future__ import annotations

import copy
import glob
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMG_DIR = Path.home() / "Workspace/ymyy-ai/ai/magic-mirror-serverless/tests/_images"
LBL_DIR = Path.home() / "Workspace/ymyy-ai/ai/magic-mirror-serverless/tests/_label_v1"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_LBL_DIR = OUT_DIR / "labels"
OUT_VIS_DIR = OUT_DIR / "vis"
OUT_GRID_DIR = OUT_DIR / "grid9"

CASE_IDS = [
    "107353-1787306528540",
    "107106-1786776854964",
    "107630-1787818776178",
]
MODALITIES = ("manual4", "manual6", "manual7")
EXTRA_PAIRS = [
    ("109572-1792557939327", "manual7"),
    ("109861-1793243741132", "manual7"),
]
NON_SKIN_LABELS = {"eyebrow", "eyelash", "nostril", "lips", "light_leak", "exclude"}


def rasterize_labels(
    shapes: list[dict[str, Any]],
    labels: set[str],
    height: int,
    width: int,
) -> np.ndarray:
    """Fill polygon shapes matching `labels` into a binary uint8 mask (0 or 255)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        if shape.get("label") not in labels:
            continue
        if shape.get("shape_type") != "polygon":
            continue
        pts = np.round(np.asarray(shape["points"], dtype=np.float32)).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
    return mask


def robust_stats(values: np.ndarray) -> dict[str, float]:
    """Return median, MAD (scaled to std), and key percentiles."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) * 1.4826
    p05, p10, p85, p95 = np.percentile(values, [5, 10, 85, 95])
    return {
        "med": med,
        "mad": max(mad, 3.0),
        "p05": float(p05),
        "p10": float(p10),
        "p85": float(p85),
        "p95": float(p95),
    }


def _build_line_kernels(length: int = 21, n_angles: int = 16) -> list[np.ndarray]:
    """Build normalized directional line kernels (reused from hair_bilateral)."""
    half = length // 2 + 2
    size = 2 * half + 1
    kernels: list[np.ndarray] = []
    for i in range(n_angles):
        theta = np.pi * i / n_angles
        dx, dy = np.cos(theta), np.sin(theta)
        k = np.zeros((size, size), dtype=np.float32)
        for t in np.linspace(-length / 2, length / 2, length * 4):
            x = int(round(half + t * dx))
            y = int(round(half + t * dy))
            k[y, x] = 1.0
        kernels.append(k / k.sum())
    return kernels


_LINE_KERNELS = _build_line_kernels(length=21, n_angles=16)


def thin_dark_hair_mask(
    L: np.ndarray, roi: np.ndarray, clean_skin_mask: np.ndarray, modality: str, is_plaster: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """Detect hair strands via bilateral -> 7x7 black-hat -> 16-angle line response.

    Threshold is self-calibrated on `clean_skin_mask` so pores/spots stay untouched.
    """
    # 1. Bilateral pre-smooth: suppress pores while preserving hair strand edges
    pre = cv2.bilateralFilter(L, d=7, sigmaColor=20.0, sigmaSpace=5.0)
    # 2. Black-hat: dark hair strands become bright
    k_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(pre, cv2.MORPH_BLACKHAT, k_bh).astype(np.float32)
    # 3. 16-orientation directional line response (boosts strands, suppresses round spots/pores)
    resp = np.stack(
        [cv2.filter2D(tophat, cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT) for k in _LINE_KERNELS]
    )
    line_resp = resp.max(axis=0)

    # 4. Self-calibrate threshold from clean face skin
    ref_vals = line_resp[clean_skin_mask > 0]
    if is_plaster:
        # Plaster-style m7: keep long strands only; short wrinkle segments are dropped
        thr = max(8.0, float(np.percentile(ref_vals, 97.0)) * 0.95)
        dil_k = 3
    elif modality == "manual7":
        thr = min(5.5, max(2.8, float(np.percentile(ref_vals, 92.0)) * 0.85))
        dil_k = 5
    else:
        thr = max(3.0, float(np.percentile(ref_vals, 97.5)) * 0.95)
        dil_k = 5

    ridges = ((line_resp >= thr) & (roi > 0)).astype(np.uint8) * 255
    if is_plaster:
        num_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(ridges)
        clean_ridges = np.zeros_like(ridges)
        for i in range(1, num_cc):
            w_cc, h_cc = stats_cc[i, cv2.CC_STAT_WIDTH], stats_cc[i, cv2.CC_STAT_HEIGHT]
            if max(w_cc, h_cc) >= 35:
                clean_ridges[labels_cc == i] = 255
        ridges = clean_ridges
    ridges = cv2.dilate(ridges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k)))
    return ridges > 0, line_resp, thr



def segment_skin_in_exclude(
    bgr: np.ndarray,
    exclude_mask: np.ndarray,
    clean_skin_mask: np.ndarray,
    modality: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment pure skin and modality-specific target features inside `exclude_mask`."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    L, A, B = cv2.split(lab)
    H, S, V = cv2.split(hsv)

    ref = clean_skin_mask > 0
    if int(ref.sum()) < 500:
        raise ValueError("Clean face skin reference mask is too small (<500 px).")

    st_L = robust_stats(L[ref].astype(np.float32))
    st_A = robust_stats(A[ref].astype(np.float32))
    st_B = robust_stats(B[ref].astype(np.float32))

    roi = exclude_mask > 0
    Lf = L.astype(np.float32)
    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)

    # Normalized chroma distance in (a, b) from clean skin median
    da = (Af - st_A["med"]) / st_A["mad"]
    db = (Bf - st_B["med"]) / st_B["mad"]
    chroma_dist = np.hypot(da, db)

    # Auto-detect plaster-style pseudo-color m7 (near-grayscale, low saturation)
    is_plaster = modality == "manual7" and float(np.median(S[ref])) < 20.0

    # Tighter luminance cutoffs to reject dark hair and hair shadows
    L_base_cut = max(35.0, st_L["p05"] - 8.0)
    L_feat_cut = max(30.0, st_L["p05"] - 18.0)
    hair_ridges, bh, bh_thresh = thin_dark_hair_mask(
        L, exclude_mask, clean_skin_mask, modality, is_plaster
    )
    not_hair = ~hair_ridges

    # 1) Base pure skin mask
    if is_plaster:
        # Plaster-style m7 has near-zero chroma; rely on L + directional hair mask
        base_skin = roi & not_hair & (Lf >= max(80.0, st_L["p05"] - 15.0))
    else:
        chroma_max = 2.2 if modality == "manual7" else 2.6
        base_skin = roi & not_hair & (chroma_dist <= chroma_max) & (Lf >= L_base_cut)

    # 2) Modality-specific feature union (strictly excluding hair ridges)
    if modality == "manual4":
        fluor = roi & not_hair & (V >= max(110, int(st_L["p85"]))) & (S >= 65) & (Lf >= st_L["p85"])
        purple = roi & not_hair & (H >= 115) & (H <= 165) & (S >= 28) & (Lf >= L_feat_cut)
        uv_spots = (
            roi
            & not_hair
            & (Lf >= L_feat_cut)
            & (Bf <= st_B["p95"] + 8.0)
            & (chroma_dist <= 3.2)
        )
        mod_feat = fluor | purple | uv_spots

    elif modality == "manual6":
        redness = (
            roi
            & not_hair
            & (Af >= max(136.0, st_A["med"] + 1.2 * st_A["mad"]))
            & (Bf >= st_B["p05"] - 10.0)
            & (Lf >= L_feat_cut)
        )
        spots = (
            roi
            & not_hair
            & (Lf >= L_feat_cut)
            & (Af >= st_A["p05"] - 6.0)
            & (Bf >= st_B["p05"] - 6.0)
            & (chroma_dist <= 3.2)
        )
        mod_feat = redness | spots

    elif modality == "manual7":
        mod_feat = roi & not_hair & (Lf >= max(185.0, st_L["p85"])) & (S <= 95)

    else:
        raise ValueError(f"Unsupported modality: {modality}")

    combined = base_skin | mod_feat
    mask_raw = combined.astype(np.uint8) * 255

    # Only light 3x3 open to remove 1px noise; NO closing so thin hair strands stay split
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, k_open)
    mask = cv2.bitwise_and(mask, exclude_mask)

    debug = {
        "L": L,
        "chroma_dist": chroma_dist,
        "bh": bh,
        "bh_thresh": bh_thresh,
        "hair_ridges": hair_ridges,
        "base_skin": base_skin,
        "mod_feat": mod_feat,
        "mask_raw": mask_raw,
    }
    return mask, debug


def mask_to_xanylabeling_shapes(
    mask: np.ndarray,
    min_area: float = 150.0,
    approx_eps: float = 1.5,
) -> list[dict[str, Any]]:
    """Convert binary mask external contours to X-AnyLabeling `skin` polygon shapes."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes: list[dict[str, Any]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, approx_eps, True)
        if len(approx) < 3:
            continue
        points = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
        shapes.append(
            {
                "review_status": "pending",
                "label": "skin",
                "score": None,
                "points": points,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )
    return shapes


def save_visualization(
    bgr: np.ndarray,
    exclude_mask: np.ndarray,
    auto_skin_mask: np.ndarray,
    existing_skin_mask: np.ndarray,
    out_path: Path,
) -> None:
    """Save cropped side-by-side visualization around `exclude` bounding box."""
    ys, xs = np.where(exclude_mask > 0)
    if len(xs) == 0:
        return
    pad = 40
    h, w = bgr.shape[:2]
    x1, x2 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
    y1, y2 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)

    crop_orig = bgr[y1:y2, x1:x2].copy()
    crop_vis = crop_orig.copy()

    # Overlay auto skin in semi-transparent green
    skin_crop = auto_skin_mask[y1:y2, x1:x2] > 0
    green = np.zeros_like(crop_vis)
    green[:, :, 1] = 255
    crop_vis[skin_crop] = cv2.addWeighted(crop_vis[skin_crop], 0.6, green[skin_crop], 0.4, 0)

    # Draw contours: exclude=yellow, auto_skin=bright green, existing_skin=cyan
    ex_cnts, _ = cv2.findContours(exclude_mask[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sk_cnts, _ = cv2.findContours(auto_skin_mask[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop_vis, ex_cnts, -1, (0, 255, 255), 2)
    cv2.drawContours(crop_vis, sk_cnts, -1, (0, 255, 0), 2)

    if int(existing_skin_mask.sum()) > 0:
        old_cnts, _ = cv2.findContours(
            existing_skin_mask[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(crop_vis, old_cnts, -1, (255, 255, 0), 2)

    panel = np.hstack([crop_orig, crop_vis])
    # Downscale wide panels for fast inspection
    max_w = 1600
    if panel.shape[1] > max_w:
        scale = max_w / panel.shape[1]
        panel = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)))
    cv2.imwrite(str(out_path), panel)


def _put_title(tile: np.ndarray, title: str) -> np.ndarray:
    """Draw a compact top-left label bar on a tile copy."""
    out = tile.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 14 * len(title) + 16), 30), (20, 20, 20), -1)
    cv2.putText(out, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _overlay_mask(base: np.ndarray, mask_bool: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    """Alpha-blend a boolean mask onto a BGR image."""
    out = base.copy()
    if mask_bool.any():
        color_layer = np.zeros_like(out)
        color_layer[:] = color_bgr
        out[mask_bool] = cv2.addWeighted(out[mask_bool], 0.5, color_layer[mask_bool], 0.5, 0)
    return out


def save_grid9(
    bgr: np.ndarray,
    exclude_mask: np.ndarray,
    auto_skin_mask: np.ndarray,
    existing_skin_mask: np.ndarray,
    debug: dict[str, Any],
    modality: str,
    out_path: Path,
) -> None:
    """Save a 3x3 grid of intermediate pipeline steps cropped around `exclude` bbox."""
    ys, xs = np.where(exclude_mask > 0)
    if len(xs) == 0:
        return
    pad = 40
    h, w = bgr.shape[:2]
    x1, x2 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
    y1, y2 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)

    crop = bgr[y1:y2, x1:x2]
    ex_c = exclude_mask[y1:y2, x1:x2]
    ex_cnts, _ = cv2.findContours(ex_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 1. Orig + Exclude
    t1 = crop.copy()
    cv2.drawContours(t1, ex_cnts, -1, (0, 255, 255), 2)

    # 2. Lab L
    t2 = cv2.cvtColor(debug["L"][y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)

    # 3. Chroma Dist heatmap (0..6 mapped to 0..255)
    cd = np.clip(debug["chroma_dist"][y1:y2, x1:x2] / 6.0 * 255.0, 0, 255).astype(np.uint8)
    t3 = cv2.applyColorMap(cd, cv2.COLORMAP_TURBO)
    t3[ex_c == 0] = (t3[ex_c == 0] * 0.25).astype(np.uint8)

    # 4. Bilateral + BlackHat + 16-angle Line response
    bh_roi = debug["bh"][y1:y2, x1:x2]
    hi = max(float(np.percentile(bh_roi[ex_c > 0], 99.5)), 1e-9) if (ex_c > 0).any() else 1.0
    bh_c = np.clip(bh_roi / hi * 255.0, 0, 255).astype(np.uint8)
    t4 = cv2.applyColorMap(bh_c, cv2.COLORMAP_INFERNO)

    # 5. Detected Hair Mask (red)
    t5 = _overlay_mask(crop, debug["hair_ridges"][y1:y2, x1:x2], (0, 0, 255))

    # 6. Base Skin (green)
    t6 = _overlay_mask(crop, debug["base_skin"][y1:y2, x1:x2], (0, 255, 0))

    # 7. Modality Features (magenta)
    t7 = _overlay_mask(crop, debug["mod_feat"][y1:y2, x1:x2], (255, 0, 255))

    # 8. Combined Raw before 3x3 open
    t8 = cv2.cvtColor(debug["mask_raw"][y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)

    # 9. Final Skin + contours
    t9 = _overlay_mask(crop, auto_skin_mask[y1:y2, x1:x2] > 0, (0, 255, 0))
    sk_cnts, _ = cv2.findContours(auto_skin_mask[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(t9, ex_cnts, -1, (0, 255, 255), 2)
    cv2.drawContours(t9, sk_cnts, -1, (0, 255, 0), 1)
    if int(existing_skin_mask.sum()) > 0:
        old_cnts, _ = cv2.findContours(
            existing_skin_mask[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(t9, old_cnts, -1, (255, 255, 0), 2)

    titles = [
        "1. Orig + Exclude",
        "2. Lab L",
        "3. Chroma Dist",
        f"4. Bi+Line (th={debug['bh_thresh']:.1f})",
        "5. Hair Mask",
        "6. Base Skin",
        f"7. {modality} Feat",
        "8. Combined Raw",
        "9. Final Skin",
    ]
    raw_tiles = [t1, t2, t3, t4, t5, t6, t7, t8, t9]

    # Resize each tile to a uniform width of 520px
    tile_w = 520
    tile_h = max(120, int(crop.shape[0] * (tile_w / crop.shape[1])))
    tiles = [
        _put_title(cv2.resize(t, (tile_w, tile_h), interpolation=cv2.INTER_AREA), title)
        for t, title in zip(raw_tiles, titles)
    ]

    grid = np.vstack(
        [
            np.hstack(tiles[0:3]),
            np.hstack(tiles[3:6]),
            np.hstack(tiles[6:9]),
        ]
    )
    cv2.imwrite(str(out_path), grid)


def process_one(case_id: str, modality: str) -> dict[str, Any]:
    """Process one (case_id, modality) pair and write updated JSON + visualization."""
    json_matches = sorted(glob.glob(str(LBL_DIR / f"{case_id}-{modality}-*.json")))
    img_matches = sorted(glob.glob(str(IMG_DIR / f"{case_id}-{modality}-*.jpg")))
    if not json_matches or not img_matches:
        raise FileNotFoundError(f"Missing image or json for {case_id} {modality}")

    json_path = Path(json_matches[0])
    img_path = Path(img_matches[0])

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    jw, jh = int(data["imageWidth"]), int(data["imageHeight"])
    bgr_full = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr_full is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    bgr = cv2.resize(bgr_full, (jw, jh), interpolation=cv2.INTER_AREA)

    shapes = data.get("shapes", [])
    exclude_mask = rasterize_labels(shapes, {"exclude"}, jh, jw)
    face_mask = rasterize_labels(shapes, {"face"}, jh, jw)
    non_skin_mask = rasterize_labels(shapes, NON_SKIN_LABELS, jh, jw)
    clean_skin_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(non_skin_mask))
    existing_skin_mask = rasterize_labels(shapes, {"skin"}, jh, jw)

    auto_skin_mask, debug = segment_skin_in_exclude(bgr, exclude_mask, clean_skin_mask, modality)
    new_skin_shapes = mask_to_xanylabeling_shapes(auto_skin_mask)

    # Keep all non-skin shapes + append newly generated skin shapes
    out_data = copy.deepcopy(data)
    out_data["shapes"] = [s for s in shapes if s.get("label") != "skin"] + new_skin_shapes

    out_json_path = OUT_LBL_DIR / json_path.name
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    out_vis_path = OUT_VIS_DIR / f"{case_id}_{modality}_vis.jpg"
    save_visualization(bgr, exclude_mask, auto_skin_mask, existing_skin_mask, out_vis_path)

    out_grid_path = OUT_GRID_DIR / f"{case_id}_{modality}_grid9.jpg"
    save_grid9(bgr, exclude_mask, auto_skin_mask, existing_skin_mask, debug, modality, out_grid_path)

    ex_px = int((exclude_mask > 0).sum())
    sk_px = int((auto_skin_mask > 0).sum())
    ratio = (sk_px / ex_px * 100.0) if ex_px > 0 else 0.0
    return {
        "case_id": case_id,
        "modality": modality,
        "exclude_px": ex_px,
        "skin_px": sk_px,
        "skin_ratio_pct": round(ratio, 1),
        "skin_polygons": len(new_skin_shapes),
        "out_json": str(out_json_path),
        "out_vis": str(out_vis_path),
    }


def main() -> None:
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_GRID_DIR.mkdir(parents=True, exist_ok=True)

    pairs = [(cid, m) for cid in CASE_IDS for m in MODALITIES] + EXTRA_PAIRS
    for case_id, modality in pairs:
        res = process_one(case_id, modality)
        print(
            f"{res['case_id']} {res['modality']}: "
            f"polygons={res['skin_polygons']:2d} | "
            f"skin/exclude={res['skin_ratio_pct']:5.1f}% "
            f"({res['skin_px']}/{res['exclude_px']} px)"
        )

    # Self-check: verify all output JSONs are valid and contain polygon points
    out_jsons = list(OUT_LBL_DIR.glob("*.json"))
    assert len(out_jsons) == len(pairs), f"Expected {len(pairs)} JSONs, got {len(out_jsons)}"
    for jp in out_jsons:
        with jp.open("r", encoding="utf-8") as f:
            d = json.load(f)
        assert "shapes" in d and d["imageWidth"] == 2000 and d["imageHeight"] == 3000
    print(f"OK: {len(out_jsons)} JSONs -> {OUT_LBL_DIR}")
    print(f"OK: visuals -> {OUT_VIS_DIR}")


if __name__ == "__main__":
    main()
