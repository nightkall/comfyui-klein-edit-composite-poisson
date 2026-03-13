# Klein Edit Composite

A ComfyUI custom node that composites a Klein edit back onto the original image using DIS optical flow change detection. Eliminates color drift by restoring original pixels everywhere Klein didn't intentionally change anything.

## How It Works

The node uses bidirectional DIS optical flow to align the original and generated images, then detects which pixels genuinely changed using perceptual color distance (ΔE) in LAB color space. Only those changed pixels are taken from the generated image — everything else is restored from the original, preserving its color fidelity.

A global rigid alignment step (v2.2) calculates a single camera-shift correction from unchanged background pixels, then applies it uniformly to the entire generated image. This fixes AI-induced background drift without introducing seam distortion around the edit boundary.

Several refinements prevent false positives:
- A pre-blur step absorbs sub-pixel misalignment halos
- Luma weighting reduces sensitivity to AI-induced global contrast shifts
- Smoothing of the difference map before thresholding eliminates speckle noise
- Morphological closing fills holes in the change mask
- Small isolated blobs are removed automatically

## Installation

Drop the folder into `ComfyUI/custom_nodes/klein_edit_composite/` with an `__init__.py` that imports `NODE_CLASS_MAPPINGS`.

## Settings

### delta_e_threshold
How different a pixel's color must be to count as "edited." Higher values mean only obvious edits are detected, resulting in a smaller mask and more of the original preserved. Lower values capture subtler changes, producing a larger mask that uses more of the generated image. Set to **-1** for automatic tuning.

### grow_mask_pct
Expands or shrinks the detected edit region. Positive values grow the mask outward, capturing more of the surrounding area — useful if edges of the edit are being clipped. Negative values erode the mask inward, trimming the boundary — useful if too much background is being pulled in.

### feather_pct
How gradually the edit blends into the original at the mask boundary. Higher values create a wider, softer transition for smoother blending, but may wash out fine edges. Lower values create a sharper cutover with crisper edges, but seams may become more visible.

### flow_quality
Accuracy of the optical flow alignment between original and generated images. Higher quality means more precise change detection and alignment but slower processing. Lower quality is faster but may miss subtle shifts or produce noisier masks.

### occlusion_threshold
Sensitivity to pixels that moved so much they can't be reliably matched between images. Higher values ignore more motion discrepancies, reducing false positives from camera jitter but potentially missing real edits. Lower values flag more pixels as changed, catching more edits but possibly over-detecting in noisy areas. Set to **-1** for automatic tuning.

### close_radius_pct
Fills small holes and gaps inside the detected edit region. Higher values close larger gaps, creating a more solid and continuous mask. Lower values leave small holes intact, preserving finer mask detail but potentially leaving speckled artifacts inside the edit.

### min_region_pct
Removes small isolated blobs from the mask that are likely false positives. Higher values filter out larger stray regions for a cleaner mask, but may discard small intentional edits. Lower values keep smaller regions, preserving tiny edits but potentially letting through noise.

## Outputs

- **composited_image** — The final result with original pixels restored in unchanged areas.
- **change_mask** — The detected edit region used for compositing. Can be fed to other nodes for further processing.
- **report** — A text summary of the settings used (including auto-tuned values) and detection statistics.

---

NOTE: AI coded. I am not a developer.
