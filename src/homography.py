import cv2
import numpy as np


def estimate_homography(pts1, pts2, reproj_threshold=3.0, min_matches=10, min_inliers=8):
    if len(pts1) < max(4, min_matches) or len(pts2) < max(4, min_matches):
        return None, None, {"status": "insufficient_matches", "inliers": 0, "inlier_ratio": 0.0}

    H, mask = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, reproj_threshold)

    if H is None or mask is None:
        return None, None, {"status": "homography_failed", "inliers": 0, "inlier_ratio": 0.0}

    mask = mask.ravel().astype(bool)
    inliers = int(mask.sum())

    # Degenerate homography check: condition number of the 3x3 matrix
    if not _is_valid_homography(H):
        return None, mask, {
            "status": "degenerate_homography",
            "inliers": inliers,
            "inlier_ratio": inliers / max(len(mask), 1),
        }

    if inliers < min_inliers:
        return H, mask, {
            "status": "too_few_inliers",
            "inliers": inliers,
            "inlier_ratio": inliers / max(len(mask), 1),
        }

    return H, mask, {
        "status": "ok",
        "inliers": inliers,
        "inlier_ratio": inliers / max(len(mask), 1),
    }


def _is_valid_homography(H: np.ndarray, max_cond: float = 1e6, max_det_ratio: float = 100.0) -> bool:
    """Check if a homography is geometrically reasonable.

    Rejects:
    - Ill-conditioned matrices (large condition number → degenerate configuration)
    - Matrices with extreme scale change (|det| >> 1 or << 1)
    """
    try:
        cond = np.linalg.cond(H)
        if cond > max_cond:
            return False
        det = abs(np.linalg.det(H))
        if det < 1.0 / max_det_ratio or det > max_det_ratio:
            return False
        return True
    except Exception:
        return False


def warp_image(img, H, target_shape):
    h, w = target_shape[:2]
    return cv2.warpPerspective(img, H, (w, h))


def reprojection_rmse(pts1, pts2, H, mask=None):
    if H is None or len(pts1) == 0:
        return None
    pts1h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)
    proj = (H @ pts1h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - pts2, axis=1)
    if mask is not None and len(mask) == len(err):
        err = err[mask]
    if len(err) == 0:
        return None
    return float(np.sqrt(np.mean(err ** 2)))
