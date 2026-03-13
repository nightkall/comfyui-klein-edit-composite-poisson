<img width="1331" height="759" alt="image" src="https://github.com/user-attachments/assets/1ed2e7da-2196-418f-97d7-8bb8cec84dd3" />

# Klein Edit Composite

A ComfyUI custom node that composites a Klein edit back onto the original image using DIS optical flow change detection. Eliminates color drift by restoring original pixels everywhere Klein didn't intentionally change anything.

## How It Works

The node uses bidirectional DIS optical flow to align the original and generated images, then detects which pixels genuinely changed using perceptual color distance (ΔE) in LAB color space. Only those changed pixels are taken from the generated image — everything else is restored from the original, preserving its color fidelity.

Several refinements prevent false positives:
- A pre-blur step absorbs sub-pixel misalignment halos
- Luma weighting reduces sensitivity to AI-induced global contrast shifts
- Smoothing of the difference map before thresholding eliminates speckle noise
- Morphological closing fills holes in the change mask
- Small isolated blobs are removed automatically

NOTE: AI coded. I am not a developer.
