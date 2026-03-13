"""
Klein Edit Composite Node
==========================
Composites a Klein edit back onto the original image using DIS optical flow
change detection. Eliminates color drift by restoring original pixels
everywhere Klein didn't intentionally change anything.

v2.2 — Global Rigid Alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Decouples mask generation from image alignment:
1. Uses dense optical flow to accurately isolate the edited area.
2. Calculates a single global rigid transformation (Translation/Scale/Rotation) 
   based purely on the unedited background.
3. Shifts the entire generated image uniformly, eliminating seam distortion.

Install: drop in ComfyUI/custom_nodes/klein_edit_composite/
         with an __init__.py that imports NODE_CLASS_MAPPINGS.
"""

import numpy as np
import torch
import cv2
from PIL import Image
import math


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def _diag(H: int, W: int) -> float:
    return math.sqrt(H * H + W * W)

def _pct_to_px(pct: float, diag: float) -> int:
    return max(0, round(abs(pct) * diag / 100.0))

def _blur_kernel_for_diag(diag: float) -> tuple:
    k = max(3, int(round(diag / 724.0 * 3)))
    if k % 2 == 0: k += 1
    return (k, k)


# ---------------------------------------------------------------------------
# LAB conversion
# ---------------------------------------------------------------------------

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb <= 0.04045,
                   rgb / 12.92,
                   ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],[0.2126729, 0.7151522, 0.0721750],[0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ M.T / np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    def f(t):
        return np.where(t > (6/29)**3,
                        t ** (1/3),
                        t / (3 * (6/29)**2) + 4/29)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    return np.stack([116*fy - 16, 500*(fx - fy), 200*(fy - fz)], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Optical flow & Math
# ---------------------------------------------------------------------------

def _dis_flow(gray_a: np.ndarray, gray_b: np.ndarray, preset: int) -> np.ndarray:
    return cv2.DISOpticalFlow_create(preset).calc(gray_a, gray_b, None)

def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    H, W = flow.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (xx + flow[..., 0]).astype(np.float32)
    map_y = (yy + flow[..., 1]).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

def _occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, threshold: float) -> np.ndarray:
    H, W = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    bwd_y = cv2.remap(flow_bwd[..., 1], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    err = np.sqrt((flow_fwd[..., 0] + bwd_x)**2 + (flow_fwd[..., 1] + bwd_y)**2)
    return (err > threshold).astype(np.float32)

def _grow_mask(mask: np.ndarray, grow_px: int) -> np.ndarray:
    if grow_px == 0: return mask
    radius = abs(grow_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    op = cv2.MORPH_DILATE if grow_px > 0 else cv2.MORPH_ERODE
    return cv2.morphologyEx(mask.astype(np.uint8), op, k).astype(np.float32)


# ---------------------------------------------------------------------------
# Auto-tuning
# ---------------------------------------------------------------------------

def _auto_delta_e_threshold(delta_e: np.ndarray) -> float:
    p75 = float(np.percentile(delta_e, 75))
    p90 = float(np.percentile(delta_e, 90))
    spread = p90 - p75
    threshold = p75 + max(spread * 0.4, 3.0) if spread > 5.0 else p75 + max(spread * 0.6, 4.0)
    return float(np.clip(threshold, 4.0, 60.0))

def _auto_occlusion_threshold(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> float:
    H, W = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    bwd_y = cv2.remap(flow_bwd[..., 1], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    err = np.sqrt((flow_fwd[..., 0] + bwd_x)**2 + (flow_fwd[..., 1] + bwd_y)**2)
    p85 = float(np.percentile(err, 85))
    p95 = float(np.percentile(err, 95))
    threshold = p95 + max((p95 - p85) * 0.5, 0.5)
    return float(np.clip(threshold, 1.0, 15.0))


# ---------------------------------------------------------------------------
# Core detection + Global Alignment Composite
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
               min_region_px: int,
               feather_px: float) -> tuple:

    H, W = original_np.shape[:2]
    diag = _diag(H, W)
    
    orig_u8 = (np.clip(original_np, 0, 1) * 255).astype(np.uint8)
    gen_u8  = (np.clip(generated_np, 0, 1) * 255).astype(np.uint8)
    gray_orig = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY)
    gray_gen  = cv2.cvtColor(gen_u8,  cv2.COLOR_RGB2GRAY)

    # 1. Coordinate Space Mapping (Orig -> Gen) for Mask Detection
    flow_fwd = _dis_flow(gray_orig, gray_gen, flow_preset)
    flow_bwd = _dis_flow(gray_gen, gray_orig, flow_preset)

    # 2. Extract Change Mask
    warped_gen_dense = _warp(generated_np.astype(np.float32), flow_fwd)
    
    blur_kernel = _blur_kernel_for_diag(diag)
    orig_blur = cv2.GaussianBlur(original_np, blur_kernel, 0)
    wgen_blur = cv2.GaussianBlur(warped_gen_dense, blur_kernel, 0)

    orig_lab = _rgb_to_lab(orig_blur.reshape(-1, 3)).reshape(H, W, 3)
    wgen_lab = _rgb_to_lab(wgen_blur.reshape(-1, 3)).reshape(H, W, 3)

    lab_diff = orig_lab - wgen_lab
    lab_diff[..., 0] *= 0.7
    delta_e = np.sqrt((lab_diff**2).sum(axis=2))

    sk = max(_blur_kernel_for_diag(diag)[0], 5)
    if sk % 2 == 0: sk += 1
    delta_e_smooth = cv2.GaussianBlur(delta_e, (sk, sk), 0)

    auto_report = {}
    if delta_e_threshold < 0:
        delta_e_threshold = _auto_delta_e_threshold(delta_e_smooth)
        auto_report["auto_delta_e"] = delta_e_threshold

    if occlusion_threshold < 0:
        occlusion_threshold = _auto_occlusion_threshold(flow_fwd, flow_bwd)
        auto_report["auto_occlusion"] = occlusion_threshold

    occluded = _occlusion_mask(flow_fwd, flow_bwd, occlusion_threshold)

    changed = np.maximum((delta_e_smooth > delta_e_threshold).astype(np.float32), occluded)

    # Morphology on mask
    if grow_px != 0:
        changed = _grow_mask(changed, grow_px)
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(np.float32)
    if min_region_px > 0:
        n, labeled, stats_cc, _ = cv2.connectedComponentsWithStats((changed > 0.5).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats_cc[i, cv2.CC_STAT_AREA] < min_region_px:
                changed[labeled == i] = 0

    sharp_mask = changed.copy()

    # Feather edge mask for compositing
    if feather_px > 0:
        inv_mask = (sharp_mask < 0.5).astype(np.uint8)
        if inv_mask.min() == 0:
            dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
            fade_dist = feather_px * 3.0
            t = np.clip(1.0 - (dist / fade_dist), 0.0, 1.0)
            composite_mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        else:
            composite_mask = sharp_mask
    else:
        composite_mask = sharp_mask


    # ---------------------------------------------------------------------------
    # 3. GLOBAL RIGID ALIGNMENT: Align Gen -> Orig using ONLY background points
    # ---------------------------------------------------------------------------
    # Sample points on a grid to find global transform
    y_grid, x_grid = np.mgrid[0:H:10, 0:W:10]
    pts_orig = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2).astype(np.float32)
    
    flow_sub = flow_fwd[0:H:10, 0:W:10].reshape(-1, 2)
    mask_sub = sharp_mask[0:H:10, 0:W:10].reshape(-1)

    # Filter for points exclusively in the unedited area
    bg_idx = mask_sub < 0.1
    M = None
    if bg_idx.sum() > 10:
        src_pts = pts_orig[bg_idx]
        dst_pts = src_pts + flow_sub[bg_idx]
        
        # Calculate a single Affine transform (Translation + Rotation + Uniform Scale). 
        # This maps Original space -> Generated space.
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    if M is not None:
        # Warp the entire Generated image uniformly as a single rigid plate.
        # WARP_INVERSE_MAP uses our Orig->Gen matrix to fetch pixels back to Orig space.
        final_aligned_gen = cv2.warpAffine(
            generated_np.astype(np.float32), 
            M.astype(np.float64), 
            (W, H), 
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, 
            borderMode=cv2.BORDER_REFLECT
        )
    else:
        final_aligned_gen = generated_np

    # 4. Final Composite
    m3 = composite_mask[..., np.newaxis]
    result = np.clip(original_np * (1.0 - m3) + final_aligned_gen * m3, 0, 1)

    # Stats
    flow_mag = np.sqrt((flow_fwd**2).sum(axis=2))
    n_changed = int((sharp_mask > 0.5).sum())
    stats = {
        "changed_pct":    100 * n_changed / (H * W),
        "occluded_px":    int(occluded.sum()),
        "flow_mean_px":   float(flow_mag.mean()),
        "flow_p99_px":    float(np.percentile(flow_mag, 99)),
        "median_de":      float(np.median(delta_e)),
        "resolution":     f"{W}x{H}",
        "diagonal_px":    round(diag),
    }
    stats.update(auto_report)

    return result, composite_mask, stats


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class KleinEditComposite:
    """
    Composites a Klein edit onto the original image.

    v2.2: Global Rigid Alignment. Calculates a single global camera shift from 
    unchanged background pixels and translates the entire generated image rigidly. 
    Eliminates seam distortion while fixing AI background drift.
    """

    CATEGORY = "image/Klein"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image":  ("IMAGE",),
                "generated_image": ("IMAGE",),
                "delta_e_threshold": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 100.0, "step": 1.0,
                    "tooltip": "How different a pixel's color must be to count as 'edited'. Higher values = only obvious edits are detected (smaller mask, more original preserved). Lower values = subtle changes are also captured (larger mask, more of the generated image used). Set to -1 for automatic tuning."
                }),
                "grow_mask_pct": ("FLOAT", {
                    "default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Expands or shrinks the detected edit region. Positive values grow the mask outward, capturing more of the surrounding area (useful if edges of the edit are being clipped). Negative values erode the mask inward, trimming the edges (useful if too much background is being pulled in)."
                }),
                "feather_pct": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.25,
                    "tooltip": "How gradually the edit blends into the original at the mask boundary. Higher values create a wider, softer transition (smoother blending, but may wash out fine edges). Lower values create a sharper, more abrupt cutover (crisper edges, but seams may be more visible)."
                }),
                "flow_quality": (["medium", "fast", "ultrafast"], {
                    "default": "medium",
                    "tooltip": "Accuracy of the optical flow alignment between original and generated images. Higher quality = more precise change detection and alignment (slower). Lower quality = faster processing but may miss subtle shifts or produce noisier masks."
                }),
                "occlusion_threshold": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Sensitivity to pixels that moved so much they can't be reliably matched between images. Higher values ignore more motion discrepancies (fewer false positives from camera jitter, but may miss real edits). Lower values flag more pixels as changed (catches more edits, but may over-detect in noisy areas). Set to -1 for automatic tuning."
                }),
                "close_radius_pct": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Fills small holes and gaps inside the detected edit region. Higher values close larger gaps (creates a more solid, continuous mask). Lower values leave small holes intact (preserves finer mask detail but may leave speckled artifacts inside the edit)."
                }),
                "min_region_pct": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Removes small isolated blobs from the mask that are likely false positives. Higher values filter out larger stray regions (cleaner mask, but may discard small intentional edits). Lower values keep smaller regions (preserves tiny edits, but may let through noise)."
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES  = ("composited_image", "change_mask", "report")
    FUNCTION      = "run"

    def run(self, original_image, generated_image,
            delta_e_threshold=-1.0, grow_mask_pct=0.0, feather_pct=2.0,
            flow_quality="medium", occlusion_threshold=-1.0,
            close_radius_pct=0.5, min_region_pct=0.05):

        orig_np = original_image[0].cpu().float().numpy()
        gen_np  = generated_image[0].cpu().float().numpy()

        if orig_np.shape != gen_np.shape:
            H, W = gen_np.shape[:2]
            pil  = Image.fromarray((orig_np * 255).astype(np.uint8))
            orig_np = np.array(pil.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0

        H, W = gen_np.shape[:2]
        diag = _diag(H, W)
        total_area = H * W

        grow_px    = round(grow_mask_pct * diag / 100.0)
        feather_px = abs(feather_pct) * diag / 100.0
        close_px   = _pct_to_px(close_radius_pct, diag)
        min_px     = max(0, round(min_region_pct * total_area / 100.0))

        result, change_mask, stats = _composite(
            orig_np, gen_np,
            delta_e_threshold   = delta_e_threshold,
            flow_preset         = FLOW_PRESETS[flow_quality],
            occlusion_threshold = occlusion_threshold,
            grow_px             = grow_px,
            close_radius        = close_px,
            min_region_px       = min_px,
            feather_px          = feather_px,
        )

        report_lines =[
            "=== Klein Edit Composite v2.2 (Global Align) ===",
            f"Resolution:       {stats['resolution']}  (diag {stats['diagonal_px']}px)",
            f"",
        ]

        if "auto_delta_e" in stats:
            report_lines.append(f"ΔE threshold:     AUTO → {stats['auto_delta_e']:.1f}")
        else:
            report_lines.append(f"ΔE threshold:     {delta_e_threshold:.1f}")

        if "auto_occlusion" in stats:
            report_lines.append(f"Occlusion thresh: AUTO → {stats['auto_occlusion']:.1f}")
        else:
            report_lines.append(f"Occlusion thresh: {occlusion_threshold:.1f}")

        report_lines +=[
            f"Grow mask:        {grow_mask_pct:+.1f}% → {grow_px:+d}px",
            f"Feather:          {feather_pct:.1f}% → {feather_px:.0f}px",
            f"Close radius:     {close_radius_pct:.1f}% → {close_px}px",
            f"Min region:       {min_region_pct:.2f}% → {min_px}px",
            f"Flow quality:     {flow_quality}",
            f"",
            f"Changed region:   {stats['changed_pct']:.1f}% of image",
            f"Occluded pixels:  {stats['occluded_px']:,}",
            f"Flow mean shift:  {stats['flow_mean_px']:.2f}px",
            f"Flow p99 shift:   {stats['flow_p99_px']:.2f}px",
            f"Median ΔE:        {stats['median_de']:.2f}",
        ]
        
        return (torch.from_numpy(result).unsqueeze(0), 
                torch.from_numpy(change_mask).unsqueeze(0), 
                "\n".join(report_lines))

NODE_CLASS_MAPPINGS = {"KleinEditComposite": KleinEditComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"KleinEditComposite": "Klein Edit Composite"}