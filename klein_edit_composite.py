"""
Klein Edit Composite Node
==============================================
"""

import numpy as np
import torch
import cv2
from PIL import Image
import math


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def _create_color_wheel():
    """Create a color wheel for flow visualization."""
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)
    col = 0

    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY

    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG

    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC

    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB

    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM

    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


COLORWHEEL = _create_color_wheel()


def _flow_to_color(flow, max_flow=None):
    """Convert optical flow to RGB color image."""
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(-v, -u) / np.pi

    if max_flow is None:
        max_flow = np.percentile(mag, 99)

    if max_flow > 0:
        mag = np.clip(mag * 8 / max_flow, 0, 8)

    angle = (angle + 1) / 2
    fk = (angle * (COLORWHEEL.shape[0] - 1) + 0.5).astype(np.int32)
    fk = np.clip(fk, 0, COLORWHEEL.shape[0] - 1)

    color = COLORWHEEL[fk]

    # Saturation based on magnitude
    mag = np.clip(mag, 0, 1)
    color = (1 - mag[..., np.newaxis]) * 255 + mag[..., np.newaxis] * color

    return color.astype(np.uint8)


def _apply_heatmap(img_float, mask_float, colormap=cv2.COLORMAP_JET):
    """Overlay a heatmap on top of an image."""
    mask_u8 = np.clip(mask_float * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_u8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    alpha = 0.6
    return (1 - alpha) * img_float + alpha * heatmap


def _draw_sift_matches(gray_orig, gray_gen, kp1, kp2, matches, inlier_mask=None):
    """Draw SIFT feature matches between images."""
    if len(kp1) == 0 or len(kp2) == 0 or len(matches) == 0:
        h = max(gray_orig.shape[0], gray_gen.shape[0])
        w = gray_orig.shape[1] + gray_gen.shape[1]
        canvas = np.zeros((h, w), dtype=np.uint8)
        canvas[:gray_orig.shape[0], :gray_orig.shape[1]] = gray_orig
        canvas[:gray_gen.shape[0], gray_orig.shape[1]:] = gray_gen
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    h1, w1 = gray_orig.shape[:2]
    h2, w2 = gray_gen.shape[:2]
    h_max = max(h1, h2)
    canvas = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1, :] = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2RGB)
    canvas[:h2, w1:w1+w2, :] = cv2.cvtColor(gray_gen, cv2.COLOR_GRAY2RGB)

    kp2_shifted = [cv2.KeyPoint(p.pt[0] + w1, p.pt[1], p.size) for p in kp2]

    inlier_set = set()
    if inlier_mask is not None:
        inlier_set = set(np.where(inlier_mask.ravel())[0])

    for i, m in enumerate(matches):
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
        pt2 = (int(kp2_shifted[m.trainIdx].pt[0]), int(kp2_shifted[m.trainIdx].pt[1]))

        if inlier_mask is None:
            color = (128, 128, 128)
        else:
            color = (0, 255, 0) if i in inlier_set else (0, 0, 255)

        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)

    if inlier_mask is not None:
        n_inliers = int(inlier_mask.sum())
        text = f"Matches: {len(matches)} | Inliers: {n_inliers}"
        cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return canvas.astype(np.float32) / 255.0


def _create_side_by_side(img1, img2, labels=("Original", "Generated")):
    """Create side-by-side comparison with labels."""
    h, w = img1.shape[:2]
    canvas = np.zeros((h + 40, w * 2, 3), dtype=np.float32)
    canvas[40:, :w] = img1
    canvas[40:, w:] = img2

    cv2.putText(canvas, labels[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (1, 1, 1), 2)
    cv2.putText(canvas, labels[1], (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (1, 1, 1), 2)
    return canvas


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def _diag(H: int, W: int) -> float:
    return math.sqrt(H * H + W * W)


def _pct_to_px(pct: float, diag: float) -> int:
    return max(0, round(abs(pct) * diag / 100.0))


def _blur_kernel_for_diag(diag: float) -> tuple:
    k = max(3, int(round(diag / 724.0 * 3)))
    if k % 2 == 0:
        k += 1
    return (k, k)


# ---------------------------------------------------------------------------
# LAB conversion
# ---------------------------------------------------------------------------

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ M.T / np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    def f(t):
        return np.where(t > (6 / 29) ** 3, t ** (1 / 3), t / (3 * (6 / 29) ** 2) + 4 / 29)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# SIFT pre-alignment — WITH DEBUG OUTPUTS
# ---------------------------------------------------------------------------

def _sift_prealign(gray_orig: np.ndarray, gray_gen: np.ndarray,
                   gen_float: np.ndarray, orig_mask: np.ndarray = None,
                   debug: bool = False) -> tuple:
    """Full Perspective (Homography) alignment of gen → orig using SIFT.

    Returns: (aligned_gen_float, H_mat, n_inliers, validity_mask, debug_dict)
    """
    H, W = gray_orig.shape[:2]
    debug_dict = {} if debug else None

    sift = cv2.SIFT_create(5000)
    kp1, des1 = sift.detectAndCompute(gray_orig, mask=orig_mask)
    kp2, des2 = sift.detectAndCompute(gray_gen, mask=None)

    if debug:
        debug_dict['orig_kp'] = len(kp1) if kp1 else 0
        debug_dict['gen_kp'] = len(kp2) if kp2 else 0

    fallback_valid = np.ones((H, W), dtype=np.float32)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        if debug:
            debug_dict['match_viz'] = _draw_sift_matches(gray_orig, gray_gen, kp1 or [], kp2 or [], [])
        return gen_float, None, 0, fallback_valid, debug_dict

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=80),
    )
    raw_matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]

    if debug:
        debug_dict['matches_before_ransac'] = _draw_sift_matches(gray_orig, gray_gen, kp1, kp2, good, None)

    if len(good) < 10:
        if debug:
            debug_dict['match_viz'] = _draw_sift_matches(gray_orig, gray_gen, kp1, kp2, good, None)
        return gen_float, None, 0, fallback_valid, debug_dict

    pts_orig = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_gen  = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H_mat, inlier_mask = cv2.findHomography(pts_gen, pts_orig, cv2.RANSAC, 4.0)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    if debug:
        debug_dict['match_viz'] = _draw_sift_matches(gray_orig, gray_gen, kp1, kp2, good, inlier_mask)
        debug_dict['n_matches'] = len(good)
        debug_dict['n_inliers'] = n_inliers

    if H_mat is None:
        return gen_float, None, 0, fallback_valid, debug_dict

    det = H_mat[0, 0] * H_mat[1, 1] - H_mat[0, 1] * H_mat[1, 0]
    if not (0.3 < det < 3.0):
        if debug:
            debug_dict['failure_reason'] = f"Bad determinant: {det:.2f} (needs 0.3-3.0)"
        return gen_float, None, 0, fallback_valid, debug_dict

    aligned = cv2.warpPerspective(
        gen_float, H_mat, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    valid_mask = cv2.warpPerspective(
        np.ones((H, W), dtype=np.float32), H_mat, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if debug:
        debug_dict['homography'] = H_mat
        debug_dict['determinant'] = det

    return aligned, H_mat, n_inliers, valid_mask, debug_dict


# ---------------------------------------------------------------------------
# Optical flow & helpers
# ---------------------------------------------------------------------------

def _dis_flow(gray_a: np.ndarray, gray_b: np.ndarray, preset: int) -> np.ndarray:
    return cv2.DISOpticalFlow_create(preset).calc(gray_a, gray_b, None)


def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    H, W = flow.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (xx + flow[..., 0]).astype(np.float32)
    map_y = (yy + flow[..., 1]).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)


def _fwd_bwd_error(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> np.ndarray:
    H, W = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    bwd_y = cv2.remap(flow_bwd[..., 1], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    return np.sqrt((flow_fwd[..., 0] + bwd_x)**2 + (flow_fwd[..., 1] + bwd_y)**2)


def _occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, threshold: float) -> np.ndarray:
    return (_fwd_bwd_error(flow_fwd, flow_bwd) > threshold).astype(np.float32)


def _grow_mask(mask: np.ndarray, grow_px: int) -> np.ndarray:
    if grow_px == 0:
        return mask
    radius = abs(grow_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    op = cv2.MORPH_DILATE if grow_px > 0 else cv2.MORPH_ERODE
    return cv2.morphologyEx(mask.astype(np.uint8), op, k).astype(np.float32)


def _open_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Morphological opening (erode then re-dilate) to destroy small noise blobs.

    Any connected region whose radius is smaller than *radius_px* will be
    removed.  Larger regions are restored to approximately their original size
    by the subsequent re-dilation, so real edges are largely preserved.
    """
    if radius_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px * 2 + 1, radius_px * 2 + 1))
    eroded  = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_ERODE,  k)
    reopened = cv2.morphologyEx(eroded,               cv2.MORPH_DILATE, k)
    return reopened.astype(np.float32)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes enclosed within each connected white island independently.

    Each white island is processed separately: a hole is only filled if it is
    fully enclosed by *that island alone*.  This prevents gaps that connect to
    the image exterior (e.g. ladder rungs open to the background) from being
    flooded, while still filling genuine interior voids.
    """
    binary = (mask > 0.5).astype(np.uint8)
    h, w = binary.shape
    n, labeled = cv2.connectedComponents(binary, connectivity=8)
    result = binary.copy()
    for island_id in range(1, n):
        # Isolate this island in a zero-padded canvas (exterior = 0)
        island = (labeled == island_id).astype(np.uint8)
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:h+1, 1:w+1] = island
        inv = 1 - padded
        cv2.floodFill(inv, None, (0, 0), 0)   # erase exterior background
        interior = inv[1:h+1, 1:w+1]          # what remains = enclosed holes
        result = np.clip(result + interior, 0, 1).astype(np.uint8)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Auto-tuning
# ---------------------------------------------------------------------------

def _auto_delta_e_threshold(delta_e: np.ndarray) -> float:
    p75 = float(np.percentile(delta_e, 75))
    p90 = float(np.percentile(delta_e, 90))
    spread = p90 - p75
    threshold = p75 + max(spread * 0.4, 3.0) if spread > 5.0 else p75 + max(spread * 0.6, 4.0)
    return float(np.clip(threshold, 4.0, 75.0))


def _auto_occlusion_threshold(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> float:
    err = _fwd_bwd_error(flow_fwd, flow_bwd)
    p85 = float(np.percentile(err, 85))
    p95 = float(np.percentile(err, 95))
    threshold = p95 + max((p95 - p85) * 0.5, 0.5)
    return float(np.clip(threshold, 1.0, 50.0))


# ---------------------------------------------------------------------------
# Core detection + Composite (V6.1 with Debug)
# ---------------------------------------------------------------------------

FLOW_PRESETS = {
    "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "fast":      cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "medium":    cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}


def _composite(original_np: np.ndarray,
               generated_np: np.ndarray,
               delta_e_threshold: float,
               flow_preset: int,
               occlusion_threshold: float,
               grow_px: int,
               close_radius: int,
               feather_px: float,
               noise_removal_px: int = 0,
               max_islands: int = 0,
               fill_holes: bool = False,
               use_occlusion: bool = False,
               custom_mask: np.ndarray = None,
               debug: bool = False) -> tuple:

    H, W = original_np.shape[:2]
    diag = _diag(H, W)
    debug_images = {} if debug else None

    orig_u8   = (np.clip(original_np,  0, 1) * 255).astype(np.uint8)
    gen_u8    = (np.clip(generated_np, 0, 1) * 255).astype(np.uint8)
    gray_orig = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY)
    gray_gen  = cv2.cvtColor(gen_u8,  cv2.COLOR_RGB2GRAY)

    blur_kernel = _blur_kernel_for_diag(diag)
    sk = max(_blur_kernel_for_diag(diag)[0], 5)
    if sk % 2 == 0:
        sk += 1

    orig_blur = cv2.GaussianBlur(original_np, blur_kernel, 0)
    orig_lab  = _rgb_to_lab(orig_blur.reshape(-1, 3)).reshape(H, W, 3)

    auto_report = {}

    # ------------------------------------------------------------------
    # Custom mask override: skip internal detection entirely
    # ------------------------------------------------------------------
    if custom_mask is not None:
        if debug:
            debug_images['custom_mask_override'] = np.stack([custom_mask] * 3, axis=-1)
            debug_images['mask_overlay'] = original_np.copy()
            debug_images['mask_overlay'][custom_mask > 0.5] = (
                debug_images['mask_overlay'][custom_mask > 0.5] * 0.5
                + np.array([0, 0.5, 0])
            )

        # We still run alignment so the composite uses a properly warped gen
        gen_pre, _, inliers, valid, debug_sift = _sift_prealign(
            gray_orig, gray_gen, generated_np, orig_mask=None, debug=debug
        )
        if debug:
            debug_images['pass1_sift_matches'] = debug_sift.get('match_viz', np.zeros((H, W, 3)))
            debug_images['pass1_validity_mask'] = valid.copy()

        sharp_mask    = np.clip(custom_mask, 0.0, 1.0) * valid
        composite_mask = sharp_mask

        if feather_px > 0:
            inv_mask = (sharp_mask < 0.5).astype(np.uint8)
            if inv_mask.min() == 0:
                dist      = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
                fade_dist = feather_px * 3.0
                t         = np.clip(1.0 - (dist / fade_dist), 0.0, 1.0)
                composite_mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)

        composite_mask *= valid
        m3     = composite_mask[..., np.newaxis]
        result = np.clip(original_np * (1.0 - m3) + gen_pre * m3, 0, 1)

        flow_fwd_final = _dis_flow(
            gray_orig,
            cv2.cvtColor((np.clip(gen_pre, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
            flow_preset,
        )
        if use_occlusion:
            flow_bwd_final = _dis_flow(
                cv2.cvtColor((np.clip(gen_pre, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                gray_orig,
                flow_preset,
            )
            occ_thresh = (
                occlusion_threshold if occlusion_threshold >= 0
                else _auto_occlusion_threshold(flow_fwd_final, flow_bwd_final)
            )
            if occlusion_threshold < 0:
                auto_report['auto_occlusion'] = occ_thresh
        else:
            flow_bwd_final = None
            occ_thresh     = 0

        flow_mag  = np.sqrt((flow_fwd_final**2).sum(axis=2))
        n_changed = int((sharp_mask > 0.5).sum())
        stats = {
            "changed_pct":    100 * n_changed / (H * W),
            "occluded_px":    int((_fwd_bwd_error(flow_fwd_final, flow_bwd_final) > occ_thresh).sum()) if use_occlusion else 0,
            "flow_mean_px":   float(flow_mag.mean()),
            "flow_p99_px":    float(np.percentile(flow_mag, 99)),
            "median_de":      float(np.median(np.zeros((H, W)))),  # N/A in override mode
            "resolution":     f"{W}x{H}",
            "diagonal_px":    round(diag),
            "pass1_inliers":  inliers,
            "pass2_used":     False,
            "custom_mask":    True,
        }
        stats.update(auto_report)

        if debug:
            debug_images['final_flow']      = _flow_to_color(flow_fwd_final)
            debug_images['final_alignment'] = _create_side_by_side(
                original_np, gen_pre, ("Original", "Aligned (custom mask)")
            )
            debug_images['composite_breakdown'] = np.hstack([
                original_np,
                gen_pre,
                result,
                np.stack([composite_mask] * 3, axis=-1),
            ])

        return result, composite_mask, stats, debug_images

    # ------------------------------------------------------------------
    # PASS 1: Blind Alignment → Coarse Mask
    # ------------------------------------------------------------------
    gen_pre_1, H_sift1, inliers_1, valid_1, debug_sift1 = _sift_prealign(
        gray_orig, gray_gen, generated_np, orig_mask=None, debug=debug
    )

    if debug:
        debug_images['pass1_sift_matches'] = debug_sift1.get('match_viz', np.zeros((H, W, 3)))
        debug_images['pass1_validity_mask'] = valid_1.copy()
        valid_overlay = original_np.copy()
        valid_overlay[valid_1 < 0.5] = [1, 0, 0]
        debug_images['pass1_validity_overlay'] = valid_overlay

    gray_gen_pre_1 = cv2.cvtColor(
        (np.clip(gen_pre_1, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    flow_fwd_1 = _dis_flow(gray_orig, gray_gen_pre_1, flow_preset)
    flow_bwd_1 = _dis_flow(gray_gen_pre_1, gray_orig, flow_preset) if use_occlusion else None

    warped_gen_1 = _warp(gen_pre_1, flow_fwd_1)
    wgen_lab_1   = _rgb_to_lab(
        cv2.GaussianBlur(warped_gen_1, blur_kernel, 0).reshape(-1, 3)
    ).reshape(H, W, 3)

    lab_diff_1 = orig_lab - wgen_lab_1
    lab_diff_1[..., 0] *= 0.7
    delta_e_1_warped = cv2.GaussianBlur(np.sqrt((lab_diff_1**2).sum(axis=2)), (sk, sk), 0)

    # Direct (unwarped) comparison — immune to warp-induced holes inside edit regions.
    gen_lab_1_direct = _rgb_to_lab(
        cv2.GaussianBlur(gen_pre_1, blur_kernel, 0).reshape(-1, 3)
    ).reshape(H, W, 3)
    lab_diff_1_direct = orig_lab - gen_lab_1_direct
    lab_diff_1_direct[..., 0] *= 0.7
    delta_e_1_direct = cv2.GaussianBlur(np.sqrt((lab_diff_1_direct**2).sum(axis=2)), (sk, sk), 0)

    # Blend: direct leads where the direct signal is already strong (edit regions);
    # warped leads in the background (better at suppressing camera-shake noise).
    de_thresh  = delta_e_threshold if delta_e_threshold >= 0 else _auto_delta_e_threshold(delta_e_1_direct)
    if use_occlusion:
        occ_thresh = occlusion_threshold if occlusion_threshold >= 0 else _auto_occlusion_threshold(flow_fwd_1, flow_bwd_1)
    blend_w_1  = np.clip(delta_e_1_direct / (de_thresh + 1e-6), 0.0, 1.0)
    delta_e_1  = blend_w_1 * delta_e_1_direct + (1.0 - blend_w_1) * delta_e_1_warped

    coarse_mask = (delta_e_1 > de_thresh).astype(np.float32)
    if use_occlusion:
        coarse_mask = np.maximum(coarse_mask, _occlusion_mask(flow_fwd_1, flow_bwd_1, occ_thresh))
    coarse_mask *= valid_1

    if debug:
        de_normalized = np.clip(delta_e_1 / 50.0, 0, 1)
        debug_images['pass1_delta_e']    = _apply_heatmap(original_np, de_normalized)
        debug_images['pass1_coarse_mask'] = coarse_mask.copy()
        debug_images['pass1_flow']       = _flow_to_color(flow_fwd_1)
        debug_images['pass1_alignment']  = _create_side_by_side(
            original_np, warped_gen_1, ("Original", "Pass1 Warped")
        )

    # ------------------------------------------------------------------
    # PASS 2: Masked SIFT Alignment (Background Only)
    # ------------------------------------------------------------------
    bg_mask_u8   = (coarse_mask < 0.1).astype(np.uint8) * 255
    safe_k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg_mask_safe = cv2.erode(bg_mask_u8, safe_k)

    pass2_used = False
    if (bg_mask_safe > 0).sum() > (H * W * 0.05):
        final_aligned_gen, H_sift2, inliers_2, valid_2, debug_sift2 = _sift_prealign(
            gray_orig, gray_gen, generated_np, orig_mask=bg_mask_safe, debug=debug
        )
        if H_sift2 is not None:
            pass2_used = True
            auto_report["pass2_inliers"] = inliers_2
            if debug:
                debug_images['pass2_sift_matches'] = debug_sift2.get('match_viz', np.zeros((H, W, 3)))
                debug_images['pass2_validity_mask'] = valid_2.copy()
                valid_overlay2 = original_np.copy()
                valid_overlay2[valid_2 < 0.5] = [1, 0, 0]
                debug_images['pass2_validity_overlay'] = valid_overlay2
        else:
            final_aligned_gen, valid_2 = gen_pre_1, valid_1
            auto_report["pass2_inliers"] = 0
    else:
        final_aligned_gen, valid_2 = gen_pre_1, valid_1
        auto_report["pass2_inliers"] = 0
        if debug:
            debug_images['pass2_skip_reason'] = f"Background too small ({(bg_mask_safe > 0).sum()} px)"

    # ------------------------------------------------------------------
    # PASS 3: Final Precision Mask
    # ------------------------------------------------------------------
    gray_gen_pre_2 = cv2.cvtColor(
        (np.clip(final_aligned_gen, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    flow_fwd_2 = _dis_flow(gray_orig, gray_gen_pre_2, flow_preset)
    flow_bwd_2 = _dis_flow(gray_gen_pre_2, gray_orig, flow_preset) if use_occlusion else None

    warped_gen_final = _warp(final_aligned_gen, flow_fwd_2)
    wgen_lab_2       = _rgb_to_lab(
        cv2.GaussianBlur(warped_gen_final, blur_kernel, 0).reshape(-1, 3)
    ).reshape(H, W, 3)

    lab_diff_2 = orig_lab - wgen_lab_2
    lab_diff_2[..., 0] *= 0.7
    delta_e_2_warped = cv2.GaussianBlur(np.sqrt((lab_diff_2**2).sum(axis=2)), (sk, sk), 0)

    # Direct (unwarped) comparison — immune to warp-induced holes inside edit regions.
    gen_lab_2_direct = _rgb_to_lab(
        cv2.GaussianBlur(final_aligned_gen, blur_kernel, 0).reshape(-1, 3)
    ).reshape(H, W, 3)
    lab_diff_2_direct = orig_lab - gen_lab_2_direct
    lab_diff_2_direct[..., 0] *= 0.7
    delta_e_2_direct = cv2.GaussianBlur(np.sqrt((lab_diff_2_direct**2).sum(axis=2)), (sk, sk), 0)

    # Blend: direct leads where the direct signal is already strong (edit regions);
    # warped leads in the background (better at suppressing camera-shake noise).
    blend_w_2  = np.clip(delta_e_2_direct / (de_thresh + 1e-6), 0.0, 1.0)
    delta_e_2  = blend_w_2 * delta_e_2_direct + (1.0 - blend_w_2) * delta_e_2_warped

    sharp_mask = (delta_e_2 > de_thresh).astype(np.float32)
    if use_occlusion:
        sharp_mask = np.maximum(sharp_mask, _occlusion_mask(flow_fwd_2, flow_bwd_2, occ_thresh))
    sharp_mask *= valid_2

    if delta_e_threshold  < 0: auto_report["auto_delta_e"]   = de_thresh
    if use_occlusion and occlusion_threshold < 0: auto_report["auto_occlusion"] = occ_thresh

    # Morphology
    if noise_removal_px > 0:
        sharp_mask = _open_mask(sharp_mask, noise_removal_px)
        sharp_mask *= valid_2
    if grow_px != 0:
        sharp_mask = _grow_mask(sharp_mask, grow_px)
        sharp_mask *= valid_2
    if close_radius > 0:
        k          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
        sharp_mask = cv2.morphologyEx(sharp_mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(np.float32)
    if max_islands > 0:
        n, labeled, stats_cc, _ = cv2.connectedComponentsWithStats(
            (sharp_mask > 0.5).astype(np.uint8), connectivity=8
        )
        if n - 1 > max_islands:  # n includes background label 0
            areas = [(stats_cc[i, cv2.CC_STAT_AREA], i) for i in range(1, n)]
            areas.sort(reverse=True)
            keep = {i for _, i in areas[:max_islands]}
            sharp_mask[np.isin(labeled, list(keep), invert=True) & (labeled > 0)] = 0

    if fill_holes:
        sharp_mask = _fill_holes(sharp_mask)
        sharp_mask *= valid_2

    # Feathered mask for compositing
    if feather_px > 0:
        inv_mask = (sharp_mask < 0.5).astype(np.uint8)
        if inv_mask.min() == 0:
            dist      = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
            fade_dist = feather_px * 3.0
            t         = np.clip(1.0 - (dist / fade_dist), 0.0, 1.0)
            composite_mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        else:
            composite_mask = sharp_mask
    else:
        composite_mask = sharp_mask

    composite_mask *= valid_2

    # Composite
    m3     = composite_mask[..., np.newaxis]
    result = np.clip(original_np * (1.0 - m3) + final_aligned_gen * m3, 0, 1)

    # Stats
    flow_mag  = np.sqrt((flow_fwd_2**2).sum(axis=2))
    n_changed = int((sharp_mask > 0.5).sum())
    stats = {
        "changed_pct":    100 * n_changed / (H * W),
        "occluded_px":    int((_fwd_bwd_error(flow_fwd_2, flow_bwd_2) > occ_thresh).sum()) if use_occlusion else 0,
        "flow_mean_px":   float(flow_mag.mean()),
        "flow_p99_px":    float(np.percentile(flow_mag, 99)),
        "median_de":      float(np.median(delta_e_2)),
        "resolution":     f"{W}x{H}",
        "diagonal_px":    round(diag),
        "pass1_inliers":  inliers_1,
        "pass2_used":     pass2_used,
    }
    stats.update(auto_report)

    if debug:
        debug_images['final_flow']           = _flow_to_color(flow_fwd_2)
        debug_images['final_flow_magnitude'] = flow_mag / (flow_mag.max() + 1e-6)

        de_norm = np.clip(delta_e_2 / 30.0, 0, 1)
        debug_images['final_delta_e'] = _apply_heatmap(original_np, de_norm)

        debug_images['final_sharp_mask']     = sharp_mask.copy()
        debug_images['final_composite_mask'] = composite_mask.copy()

        mask_overlay = original_np.copy()
        mask_overlay[sharp_mask > 0.5] = (
            mask_overlay[sharp_mask > 0.5] * 0.5 + np.array([0, 0.5, 0])
        )
        debug_images['mask_overlay'] = mask_overlay

        debug_images['final_alignment'] = _create_side_by_side(
            original_np, warped_gen_final, ("Original", "Final Warped")
        )
        debug_images['composite_breakdown'] = np.hstack([
            original_np,
            final_aligned_gen,
            result,
            np.stack([composite_mask] * 3, axis=-1),
        ])

    return result, composite_mask, stats, debug_images


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class KleinEditComposite:
    """
    Composites a Klein edit onto the original image with full debug visualization.
    
    """

    CATEGORY = "image/Klein"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image":  ("IMAGE",),
                "generated_image": ("IMAGE",),
                # --- Detection ---
                "delta_e_threshold": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Set to -1 for automatic tuning.",
                }),
                "flow_quality": (["medium", "fast", "ultrafast"], {
                    "default": "medium",
                }),
                "use_occlusion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Add occlusion detection to the change mask. "
                        "Useful for camera-motion edits but causes bleed on heavy color edits. "
                        "Disabled by default."
                    ),
                }),
                "occlusion_threshold": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 50.0, "step": 0.5,
                    "tooltip": "Only used when use_occlusion is enabled. -1 = auto.",
                }),
                # --- Mask cleanup ---
                "noise_removal_pct": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": (
                        "Morphological opening radius as % of image diagonal. "
                        "Erodes then re-dilates the raw mask to destroy speckle noise "
                        "smaller than this radius. 0 = disabled."
                    ),
                }),
                "close_radius_pct": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                }),
                "fill_holes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Flood-fill enclosed interior holes in the mask. "
                        "Only fills regions fully surrounded by mask; outer boundary is unchanged."
                    ),
                }),
                "max_islands": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": (
                        "Keep only the N largest connected regions in the mask, discard the rest. "
                        "0 = disabled (keep all)."
                    ),
                }),
                "grow_mask_pct": ("FLOAT", {
                    "default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1,
                }),
                # --- Output ---
                "feather_pct": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.25,
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Output debug visualization images.",
                }),
            },
            "optional": {
                "custom_mask": ("MASK", {
                    "tooltip": (
                        "When connected, skips internal change detection entirely. "
                        "The supplied mask is used directly as the composite mask "
                        "(feathering still applies). Useful for manual overrides or "
                        "upstream segmentation results."
                    ),
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "MASK", "STRING", "IMAGE")
    RETURN_NAMES  = ("composited_image", "change_mask", "report", "debug_gallery")
    FUNCTION      = "run"
    OUTPUT_NODE   = True

    def run(self, original_image, generated_image,
            delta_e_threshold=-1.0, grow_mask_pct=0.0, feather_pct=2.0,
            flow_quality="medium", occlusion_threshold=-1.0,
            close_radius_pct=0.5, noise_removal_pct=0.0, max_islands=0,
            fill_holes=False, use_occlusion=False, enable_debug=False, custom_mask=None):

        orig_np = original_image[0].cpu().float().numpy()
        gen_np  = generated_image[0].cpu().float().numpy()

        if orig_np.shape != gen_np.shape:
            H, W    = gen_np.shape[:2]
            pil     = Image.fromarray((orig_np * 255).astype(np.uint8))
            orig_np = np.array(pil.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0

        H, W       = gen_np.shape[:2]
        diag       = _diag(H, W)

        grow_px          = round(grow_mask_pct * diag / 100.0)
        feather_px       = abs(feather_pct) * diag / 100.0
        close_px         = _pct_to_px(close_radius_pct, diag)
        noise_removal_px = _pct_to_px(noise_removal_pct, diag)

        # Unpack optional custom mask (ComfyUI passes MASK as [B, H, W])
        custom_mask_np = None
        if custom_mask is not None:
            custom_mask_np = custom_mask[0].cpu().float().numpy()
            if custom_mask_np.shape != (H, W):
                pil_mask       = Image.fromarray((custom_mask_np * 255).astype(np.uint8))
                custom_mask_np = np.array(pil_mask.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0

        result, change_mask, stats, debug_images = _composite(
            orig_np, gen_np,
            delta_e_threshold   = delta_e_threshold,
            flow_preset         = FLOW_PRESETS[flow_quality],
            occlusion_threshold = occlusion_threshold,
            grow_px             = grow_px,
            close_radius        = close_px,
            noise_removal_px    = noise_removal_px,
            max_islands         = max_islands,
            fill_holes          = fill_holes,
            use_occlusion       = use_occlusion,
            feather_px          = feather_px,
            custom_mask         = custom_mask_np,
            debug               = enable_debug,
        )

        report_lines = [
            "=== Klein Edit Composite v6.1 (Debug Edition) ===",
            f"Resolution:       {stats['resolution']}  (diag {stats['diagonal_px']}px)",
            "",
        ]

        if stats.get("custom_mask"):
            report_lines.append("Mask source:      CUSTOM (internal detection bypassed)")
        elif stats.get("pass2_inliers", 0) > 0:
            report_lines.append("Alignment:        Two-Pass Success! Background rigidly locked.")
            report_lines.append(f"Pass 1 Inliers:   {stats['pass1_inliers']} (Blind)")
            report_lines.append(f"Pass 2 Inliers:   {stats['pass2_inliers']} (Masked background only)")
        else:
            report_lines.append(
                f"Alignment:        {'Pass 2 executed' if stats['pass2_used'] else 'Pass 2 skipped'} "
                f"(fallback to Pass 1)"
            )

        if enable_debug and debug_images:
            report_lines += ["", "=== Debug Info ==="]
            if 'pass1_sift_matches' in debug_images:
                h, w = debug_images['pass1_sift_matches'].shape[:2]
                report_lines.append(f"SIFT Match Viz:   {w}x{h} (Green=Inliers, Red=Outliers)")
            if 'pass1_validity_mask' in debug_images:
                valid_pct = 100 * debug_images['pass1_validity_mask'].mean()
                report_lines.append(f"Validity Coverage: {valid_pct:.1f}% (black areas = void after warp)")
            if 'final_flow_magnitude' in debug_images:
                report_lines.append("Flow visualization available in debug gallery")
            if 'custom_mask_override' in debug_images:
                report_lines.append("Custom mask override active — debug shows alignment only")

        report_lines.append("")

        if "auto_delta_e" in stats:
            report_lines.append(f"ΔE threshold:     AUTO → {stats['auto_delta_e']:.1f}")
        else:
            report_lines.append(f"ΔE threshold:     {delta_e_threshold:.1f}")

        if "auto_occlusion" in stats:
            report_lines.append(f"Occlusion thresh: AUTO → {stats['auto_occlusion']:.1f}")
        else:
            report_lines.append(f"Occlusion thresh: {occlusion_threshold:.1f}")

        report_lines += [
            f"Grow mask:        {grow_mask_pct:+.1f}% → {grow_px:+d}px",
            f"Noise removal:    {noise_removal_pct:.2f}% → {noise_removal_px}px (opening radius)",
            f"Max islands:      {max_islands if max_islands > 0 else 'disabled'}",
            f"Fill holes:       {'yes' if fill_holes else 'no'}",
            f"Occlusion:        {'enabled' if use_occlusion else 'disabled'}",
            f"Feather:          {feather_pct:.1f}% → {feather_px:.0f}px",
            f"Flow quality:     {flow_quality}",
            "",
            f"Changed region:   {stats['changed_pct']:.1f}% of image",
            f"Flow mean shift:  {stats['flow_mean_px']:.2f}px  (Residual after Homography)",
            f"Median ΔE:        {stats['median_de']:.2f}",
        ]

        # Build debug gallery
        debug_gallery = None
        if enable_debug:
            if not debug_images:
                warning = np.zeros((512, 512, 3), dtype=np.uint8)
                cv2.putText(warning, "DEBUG ENABLED BUT NO DATA",  (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,   (255, 255, 255), 2)
                cv2.putText(warning, "Check that enable_debug=True", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                debug_gallery = torch.from_numpy(warning.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                order = [
                    ('custom_mask_override',  'Custom Mask Override'),
                    ('pass1_sift_matches',     'SIFT Matches (Pass 1)'),
                    ('pass1_alignment',        'Pass 1 Alignment'),
                    ('pass1_delta_e',          'Delta E (Pass 1)'),
                    ('pass1_validity_overlay', 'Validity Mask (Red=Void)'),
                    ('final_alignment',        'Final Alignment'),
                    ('final_delta_e',          'Delta E (Final)'),
                    ('final_flow',             'Optical Flow'),
                    ('mask_overlay',           'Detected Changes'),
                    ('composite_breakdown',    'Original | Aligned | Result | Mask'),
                ]

                viz_images = []
                for key, _label in order:
                    if key in debug_images and debug_images[key] is not None:
                        img = debug_images[key]
                        if len(img.shape) == 2:
                            img = np.stack([img] * 3, axis=-1)
                        if img.dtype != np.uint8:
                            img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
                        viz_images.append(img)

                if not viz_images:
                    debug_gallery = torch.zeros((1, 512, 512, 3))
                else:
                    target_h = 400
                    rows = []

                    for i in range(0, len(viz_images), 3):
                        row_imgs = viz_images[i:i+3]
                        resized  = []
                        for img in row_imgs:
                            if img.shape[0] != target_h:
                                scale = target_h / img.shape[0]
                                img   = cv2.resize(img, (int(img.shape[1] * scale), target_h))
                            resized.append(img)

                        # Pad to 3 columns with gray if needed
                        while len(resized) < 3:
                            h, w = resized[0].shape[:2] if resized else (target_h, target_h)
                            resized.append(np.full((h, w, 3), 64, dtype=np.uint8))

                        rows.append(np.hstack(resized))

                    # Equalise row widths before stacking
                    if len(rows) > 1:
                        max_width = max(row.shape[1] for row in rows)
                        rows = [
                            np.hstack([row, np.full((target_h, max_width - row.shape[1], 3), 64, dtype=np.uint8)])
                            if row.shape[1] < max_width else row
                            for row in rows
                        ]

                    gallery       = np.vstack(rows)
                    debug_gallery = torch.from_numpy(gallery.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            debug_gallery = torch.zeros((1, 64, 64, 3))

        return (
            torch.from_numpy(result).unsqueeze(0),
            torch.from_numpy(change_mask).unsqueeze(0),
            "\n".join(report_lines),
            debug_gallery,
        )


NODE_CLASS_MAPPINGS      = {"KleinEditComposite": KleinEditComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"KleinEditComposite": "Klein Edit Composite"}