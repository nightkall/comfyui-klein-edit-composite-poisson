# Klein Edit Composite

A ComfyUI custom node that composites a Klein edit back onto the original image using DIS optical flow change detection. Eliminates color drift by restoring original pixels everywhere Klein didn't intentionally change anything.

<img width="337" height="367" alt="image" src="https://github.com/user-attachments/assets/0705ca6b-68e2-432a-ba97-4aacb8a0d47e" />

Series of edits without the node:

![nonode](https://github.com/user-attachments/assets/4287b4a8-f58b-4b08-9e80-a41c5b165cd7)

Series of edits with the node:

![with-node](https://github.com/user-attachments/assets/6e29809a-e248-4c3f-a0e2-32aea222790a)

## How It Works

The node uses bidirectional DIS optical flow to align the original and generated images, then detects which pixels genuinely changed using perceptual color distance (ΔE) in LAB color space. Only those changed pixels are taken from the generated image — everything else is restored from the original, preserving its color fidelity.
A global rigid alignment step calculates a single camera-shift correction from unchanged background pixels, then applies it uniformly to the entire generated image. This fixes AI-induced background drift without introducing seam distortion around the edit boundary.
Several refinements prevent false positives:

-A pre-blur step absorbs sub-pixel misalignment halos

-Luma weighting reduces sensitivity to AI-induced global contrast shifts

-Smoothing of the difference map before thresholding eliminates speckle noise

-Morphological closing fills holes in the change mask

-Small isolated blobs are removed automatically


## NOTE: AI coded. I am not a developer.
