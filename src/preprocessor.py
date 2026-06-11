"""
src/preprocessor.py
--------------------
Seasonal-aware image preprocessing to reduce appearance gap before feature matching.

CLAHE (Contrast Limited Adaptive Histogram Equalization) normalises local contrast —
especially effective when snow or heavy shadow changes the global histogram drastically.
Histogram matching further aligns the tonal distribution of the two images.
"""

import cv2
import numpy as np


def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE to a grayscale or BGR image.

    For grayscale: applied directly.
    For BGR: applied to L-channel in LAB space so hue/saturation are preserved.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if len(img.shape) == 2:
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def histogram_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match the histogram of `src` to that of `ref` (per channel).

    Useful when summer/winter images have very different global brightness.
    """
    if src.shape != ref.shape:
        ref = cv2.resize(ref, (src.shape[1], src.shape[0]))

    if len(src.shape) == 2:
        return _match_channel(src, ref)

    channels = []
    for c in range(src.shape[2]):
        channels.append(_match_channel(src[:, :, c], ref[:, :, c]))
    return cv2.merge(channels)


def _match_channel(src_ch: np.ndarray, ref_ch: np.ndarray) -> np.ndarray:
    src_hist, _ = np.histogram(src_ch.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref_ch.flatten(), 256, [0, 256])
    src_cdf = src_hist.cumsum() / src_hist.sum()
    ref_cdf = ref_hist.cumsum() / ref_hist.sum()
    # build lookup table: for each src value, find the ref value with closest CDF
    lut = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
    return lut[src_ch]


def preprocess_pair(img1: np.ndarray, img2: np.ndarray,
                    use_clahe: bool = True,
                    use_hist_match: bool = False) -> tuple:
    """Apply seasonal preprocessing to an image pair.

    Args:
        img1: Summer/reference image.
        img2: Winter/query image.
        use_clahe: Apply CLAHE to both images.
        use_hist_match: Match img1's histogram to img2's (experimental).

    Returns:
        (processed_img1, processed_img2)
    """
    if use_hist_match:
        img1 = histogram_match(img1, img2)
    if use_clahe:
        img1 = clahe_enhance(img1)
        img2 = clahe_enhance(img2)
    return img1, img2
