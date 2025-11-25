import numpy as np

def compute_multi_roi_ncc(ref_image, moving_image, roi_masks):
    num_roi = len(roi_masks)
    roi_cc = np.zeros(num_roi)

    for i, mask in enumerate(roi_masks):
        ref_pixels = ref_image[mask]
        mov_pixels = moving_image[mask]

        if len(ref_pixels) > 0 and len(mov_pixels) > 0:
            roi_cc[i] = _compute_ncc(ref_pixels, mov_pixels)
        else:
            roi_cc[i] = np.nan

    return roi_cc

def _compute_ncc(ref_pixels, mov_pixels):
    ref_mean = ref_pixels.mean()
    mov_mean = mov_pixels.mean()
    ref_std = ref_pixels.std()
    mov_std = mov_pixels.std()

    if ref_std > 0 and mov_std > 0:
        return np.mean((ref_pixels - ref_mean) * (mov_pixels - mov_mean)) / (ref_std * mov_std)
    else:
        return np.nan
