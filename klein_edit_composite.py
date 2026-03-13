"""
Klein Edit Composite Node
==========================
Composites a Klein edit back onto the original image using DIS optical flow
change detection. Eliminates color drift by restoring original pixels
everywhere Klein didn't intentionally change anything.

Grow mask (+/-):
  Positive values expand the detected change region before compositing —
  useful when Klein's edit bleeds slightly outside the detected boundary.
  Negative values shrink it — useful when the detector is over-sensitive
  and restoring too much of the original.

Install: drop in ComfyUI/custom_nodes/klein_edit_composite/
         with an __init__.py that imports NODE_CLASS_MAPPINGS.
"""

import numpy as np
import torch
import cv2
from PIL import Image


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
    return np.stack([116*fy - 16, 500*(fx - fy), 200*(fy - fz)],
                    axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------

def _dis_flow(gray_a: np.ndarray, gray_b: np.ndarray,
              preset: int) -> np.ndarray:
    return cv2.DISOpticalFlow_create(preset).calc(gray_a, gray_b, None)


def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    H, W = flow.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(image,
                     xx + flow[..., 0], yy + flow[..., 1],
                     cv2.INTER_LINEAR, cv2.BORDER_REFLECT)


def _occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray,
                    threshold: float) -> np.ndarray:
    H, W = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0],
                      xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    bwd_y = cv2.remap(flow_bwd[..., 1],
                      xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    err = np.sqrt((flow_fwd[..., 0] + bwd_x)**2 +
                  (flow_fwd[..., 1] + bwd_y)**2)
    return (err > threshold).astype(np.float32)


# ---------------------------------------------------------------------------
# Grow / shrink mask
# ---------------------------------------------------------------------------

def _grow_mask(mask: np.ndarray, grow_px: int) -> np.ndarray:
    """
    Expand (grow_px > 0) or contract (grow_px < 0) binary mask.
    Uses an elliptical structuring element for natural-looking edges.
    """
    if grow_px == 0:
        return mask
    radius = abs(grow_px)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    op = cv2.MORPH_DILATE if grow_px > 0 else cv2.MORPH_ERODE
    return cv2.morphologyEx(
        mask.astype(np.uint8), op, k).astype(np.float32)


# ---------------------------------------------------------------------------
# Core detection + composite
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
    orig_u8 = (np.clip(original_np, 0, 1) * 255).astype(np.uint8)
    gen_u8  = (np.clip(generated_np, 0, 1) * 255).astype(np.uint8)

    gray_orig = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY)
    gray_gen  = cv2.cvtColor(gen_u8,  cv2.COLOR_RGB2GRAY)

    # Bidirectional DIS flow
    flow_fwd = _dis_flow(gray_orig, gray_gen, flow_preset)
    flow_bwd = _dis_flow(gray_gen, gray_orig, flow_preset)

    # Warp original into generated's coordinate space
    warped = _warp(original_np.astype(np.float32), flow_fwd)

    # 1. PRE-BLUR TRICK: Apply a subtle blur to absorb sub-pixel misalignments 
    # that DIS flow missed. This eliminates false-positive edge halos.
    blur_kernel = (3, 3) # 3x3 or 5x5 works best
    warped_blur = cv2.GaussianBlur(warped, blur_kernel, 0)
    gen_blur    = cv2.GaussianBlur(generated_np, blur_kernel, 0)

    # Convert to LAB space
    warped_lab = _rgb_to_lab(warped_blur.reshape(-1, 3)).reshape(H, W, 3)
    gen_lab    = _rgb_to_lab(gen_blur.reshape(-1, 3)).reshape(H, W, 3)

    # 2. LUMA-WEIGHTING: Calculate the difference, but reduce sensitivity 
    # to pure Lightness (L*) changes by 30%. AI often shifts global contrast.
    lab_diff = warped_lab - gen_lab
    lab_diff[..., 0] *= 0.7  # L channel is index 0. Drop to 0.7 or 0.5 to forgive lighting changes
    
    # Calculate Delta E
    delta_e = np.sqrt((lab_diff**2).sum(axis=2))

    # 3. PRE-THRESHOLD SMOOTHING: Smooth the continuous difference map 
    # before converting it to a harsh binary mask. This kills speckle noise instantly.
    delta_e_smooth = cv2.GaussianBlur(delta_e, (5, 5), 0)

    # Occlusion: new content flow can't account for
    occluded = _occlusion_mask(flow_fwd, flow_bwd, occlusion_threshold)

    # Change = color shifted OR new content appeared
    # Use the smoothed delta_e here!
    changed = np.maximum(
        (delta_e_smooth > delta_e_threshold).astype(np.float32),
        occluded
    )

    # Grow / shrink
    if grow_px != 0:
        changed = _grow_mask(changed, grow_px)

    # Close small holes
    if close_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
        changed = cv2.morphologyEx(
            changed.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(np.float32)

    # Remove tiny isolated blobs
    if min_region_px > 0:
        n, labeled, stats_cc, _ = cv2.connectedComponentsWithStats(
            (changed > 0.5).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats_cc[i, cv2.CC_STAT_AREA] < min_region_px:
                changed[labeled == i] = 0

    # Feather edges (Outward fade / Outer glow)
    # Preserves the solid white interior, fading smoothly only into the black areas.
    if feather_px > 0:
        inv_mask = (changed < 0.5).astype(np.uint8)
        if inv_mask.min() == 0:  # Only apply if there's at least one white pixel
            dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
            # Multiply by 3 to roughly match the visual reach of a 3-sigma Gaussian blur
            fade_dist = feather_px * 3.0
            
            # Linear fade from 1.0 (at dist=0) to 0.0 (at dist=fade_dist)
            t = np.clip(1.0 - (dist / fade_dist), 0.0, 1.0)
            
            # Apply smoothstep easing for a natural, glow-like falloff
            changed = (t * t * (3.0 - 2.0 * t)).astype(np.float32)

    # Composite
    m3 = changed[..., np.newaxis]
    result = np.clip(original_np * (1 - m3) + generated_np * m3, 0, 1)

    # Stats for report
    flow_mag = np.sqrt((flow_fwd**2).sum(axis=2))
    n_changed = int((changed > 0.5).sum())
    stats = {
        "changed_pct":    100 * n_changed / (H * W),
        "occluded_px":    int(occluded.sum()),
        "flow_mean_px":   float(flow_mag.mean()),
        "flow_p99_px":    float(np.percentile(flow_mag, 99)),
        "median_de":      float(np.median(delta_e)),
    }

    return result, changed, stats


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class KleinEditComposite:
    """
    Composites a Klein edit onto the original image.

    Uses DIS optical flow to detect exactly what changed, then blends:
      - Original pixels everywhere Klein drifted color but kept content
      - Generated pixels only where Klein made intentional edits

    The result has the edit applied with the original's color fidelity.

    grow_mask:
      +N px  expands the edit region (keeps more of the generated image)
      -N px  shrinks it (keeps more of the original)
      0      use detected boundary as-is
    """

    CATEGORY = "image/Klein"

    @classmethod
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image":  ("IMAGE",),
                "generated_image": ("IMAGE",),
                "delta_e_threshold": ("FLOAT", {
                    "default": 20.0, "min": 0.5, "max": 100.0, "step": 1,
                    "tooltip": (
                        "How different a pixel has to look before it's treated as "
                        "intentionally changed by the edit. "
                        "LOWER: More sensitive — catches subtle colour shifts "
                        "but may over-restore areas the edit only touched lightly, "
                        "making the composite boundary tighter than intended. "
                        "HIGHER: Less sensitive — only flags obvious changes, "
                        "but lets mild colour drift from the AI bleed through into the "
                        "final result."
                    ),
                }),
                "grow_mask": ("INT", {
                    "default": 0, "min": -50, "max": 50, "step": 1,
                    "tooltip": (
                        "Expands or shrinks the detected edit region after detection. "
                        "POSITIVE: Pushes the boundary outward so more of the generated "
                        "image is kept around the edit edges. Use this when the edit "
                        "bleeds slightly past what was detected and you're seeing a hard "
                        "seam or halo of original pixels cutting into it. "
                        "NEGATIVE: Pulls the boundary inward so more of the original "
                        "image is restored around the edges. Use this when the detector "
                        "is grabbing too much background and you want a tighter cutout. "
                        "ZERO: Use the detected boundary exactly as-is."
                    ),
                }),
                "feather_px": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": (
                        "How far the edit bleeds outward into the surrounding original "
                        "image. The edit interior stays fully solid; only the area just "
                        "outside the detected boundary fades. "
                        "LOWER: Tight fade — the edit stops close to its detected edge. "
                        "Can look clean on hard edges but may show a seam on smooth "
                        "areas like skin or sky. "
                        "HIGHER: Wide outward glow — the edit dissolves gradually into "
                        "the original over a larger area. Hides seams naturally but may "
                        "bleed the edit colour further than intended."
                    ),
                }),
                "flow_quality": (["medium", "fast", "ultrafast"], {
                    "default": "medium",
                    "tooltip": (
                        "How much effort the motion-tracking step puts in before "
                        "comparing the two images. Higher quality = more accurate "
                        "alignment = fewer false positives in the change mask, but slower. "
                        "MEDIUM: Best quality, recommended for final results. "
                        "FAST: Roughly 2x faster, slightly less accurate alignment. "
                        "ULTRAFAST: Fastest, noticeably less accurate — use for quick "
                        "previews or large batch jobs where speed matters more than precision."
                    ),
                }),
                "occlusion_threshold": ("FLOAT", {
                    "default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5,
                    "tooltip": (
                        "Controls how the node detects pixels that are genuinely new "
                        "content rather than just colour shifts — for example, an object "
                        "the edit added from nothing that wasn't in the original at all. "
                        "LOWER: More aggressive — flags more pixels as new content. Can "
                        "cause over-detection on busy or textured areas where the motion "
                        "tracking isn't perfectly consistent. "
                        "HIGHER: More lenient — only flags obviously new content. May "
                        "miss the edges of added objects, leaving a fringe of original "
                        "pixels around them."
                    ),
                }),
                "close_radius": ("INT", {
                    "default": 4, "min": 0, "max": 50, "step": 1,
                    "tooltip": (
                        "Fills small holes and gaps inside the detected edit region. "
                        "Useful when the change detector leaves swiss-cheese gaps inside "
                        "an area that should be solid — e.g. a recoloured object where "
                        "a few pixels happened to stay the same colour by coincidence. "
                        "LOWER: Little or no hole-filling — edit region stays exactly as "
                        "detected, gaps remain. "
                        "HIGHER: Aggressively fills gaps — good for large flat edits but "
                        "may join up separate edit regions that should stay independent."
                    ),
                }),
                "min_region_px": ("INT", {
                    "default": 64, "min": 0, "max": 2048, "step": 16,
                    "tooltip": (
                        "Any isolated blob of changed pixels smaller than this area "
                        "is deleted from the mask and treated as noise. "
                        "LOWER: Keeps very small detections — useful if the edit includes "
                        "fine detail like thin lines or small dots, but will also keep "
                        "specks and artefacts from the detector. "
                        "HIGHER: Removes anything that isn't a substantial region — gives "
                        "a cleaner mask on noisy images but risks dropping small "
                        "intentional edits like eyelashes or text characters."
                    ),
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES  = ("composited_image", "change_mask", "report")
    FUNCTION      = "run"

    def run(self, original_image, generated_image,
            delta_e_threshold=4.0, grow_mask=0, feather_px=8.0,
            flow_quality="medium", occlusion_threshold=3.0,
            close_radius=4, min_region_px=64):

        orig_np = original_image[0].cpu().float().numpy()
        gen_np  = generated_image[0].cpu().float().numpy()

        # Resize original to match generated if needed
        if orig_np.shape != gen_np.shape:
            H, W = gen_np.shape[:2]
            pil  = Image.fromarray((orig_np * 255).astype(np.uint8))
            orig_np = np.array(
                pil.resize((W, H), Image.LANCZOS)
            ).astype(np.float32) / 255.0

        result, change_mask, stats = _composite(
            orig_np, gen_np,
            delta_e_threshold   = delta_e_threshold,
            flow_preset         = FLOW_PRESETS[flow_quality],
            occlusion_threshold = occlusion_threshold,
            grow_px             = grow_mask,
            close_radius        = close_radius,
            min_region_px       = min_region_px,
            feather_px          = feather_px,
        )

        report_lines =[
            "=== Klein Edit Composite ===",
            f"ΔE threshold:     {delta_e_threshold}",
            f"Grow mask:        {grow_mask:+d}px",
            f"Feather (Glow):   {feather_px}px",
            f"Flow quality:     {flow_quality}",
            f"",
            f"Changed region:   {stats['changed_pct']:.1f}% of image",
            f"Occluded pixels:  {stats['occluded_px']:,} (new content detected by flow)",
            f"Flow mean shift:  {stats['flow_mean_px']:.2f}px",
            f"Flow p99 shift:   {stats['flow_p99_px']:.2f}px",
            f"Median ΔE:        {stats['median_de']:.2f}",
        ]
        report = "\n".join(report_lines)

        result_t = torch.from_numpy(result).unsqueeze(0)
        mask_t   = torch.from_numpy(change_mask).unsqueeze(0)

        return (result_t, mask_t, report)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "KleinEditComposite": KleinEditComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KleinEditComposite": "Klein Edit Composite",
}