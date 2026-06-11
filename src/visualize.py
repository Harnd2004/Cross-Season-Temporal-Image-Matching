from pathlib import Path
import cv2
import numpy as np


def _to_color(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()


def draw_matches(img1, img2, kpts1, kpts2, matches, inlier_mask=None, max_draw=150):
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    if inlier_mask is not None and len(inlier_mask) == len(matches):
        filtered = [m for m, keep in zip(matches, inlier_mask) if keep]
    else:
        filtered = matches
    filtered = filtered[:max_draw]
    return cv2.drawMatches(_to_color(img1), kpts1, _to_color(img2), kpts2, filtered, None, flags=flags)


def overlay_warped(warped, target, alpha=0.5):
    warped_c = _to_color(warped)
    target_c = _to_color(target)
    return cv2.addWeighted(warped_c, alpha, target_c, 1 - alpha, 0)


def save_image(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
