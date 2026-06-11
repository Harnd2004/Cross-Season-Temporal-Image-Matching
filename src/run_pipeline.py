from pathlib import Path
from time import perf_counter
import logging

import cv2

from .data_io import read_image, resize_max
from .matcher import build_matcher, OpenCVFeatureMatcher
from .homography import estimate_homography, warp_image, reprojection_rmse
from .visualize import draw_matches, overlay_warped, save_image
from .preprocessor import preprocess_pair

logger = logging.getLogger(__name__)

# Module-level matcher cache so the GPU model is loaded once across all pairs
_matcher_cache = {}


def get_matcher(cfg):
    key = cfg.get("processing", {}).get("matcher", "auto")
    if key not in _matcher_cache:
        _matcher_cache[key] = build_matcher(cfg)
    return _matcher_cache[key]


def process_pair(summer_path, winter_path, cfg, root: Path, save_outputs=True):
    p = cfg["processing"]
    o = cfg["output"]

    # ── 1. Data loading ────────────────────────────────────────────────────
    t0 = perf_counter()
    img1 = read_image(summer_path, grayscale=p.get("grayscale", True))
    img2 = read_image(winter_path, grayscale=p.get("grayscale", True))
    img1, scale1 = resize_max(img1, p.get("max_size", 1280))
    img2, scale2 = resize_max(img2, p.get("max_size", 1280))

    # ── 2. Seasonal preprocessing (CLAHE) ─────────────────────────────────
    use_clahe = p.get("clahe", True)
    use_hist_match = p.get("hist_match", False)
    if use_clahe or use_hist_match:
        img1, img2 = preprocess_pair(img1, img2,
                                     use_clahe=use_clahe,
                                     use_hist_match=use_hist_match)
    t1 = perf_counter()

    # ── 3. Feature extraction + matching ───────────────────────────────────
    matcher = get_matcher(cfg)
    mres = matcher.extract_and_match(img1, img2)
    t2 = perf_counter()

    # ── 4. Homography estimation ───────────────────────────────────────────
    H, inlier_mask, meta = estimate_homography(
        mres.pts1, mres.pts2,
        reproj_threshold=p.get("ransac_reproj_threshold", 3.0),
        min_matches=p.get("min_matches", 10),
        min_inliers=p.get("min_inliers", 8),
    )
    rmse = reprojection_rmse(mres.pts1, mres.pts2, H, inlier_mask)
    t3 = perf_counter()

    # ── 5. Save visualizations ─────────────────────────────────────────────
    basename = Path(summer_path).stem
    if save_outputs:
        match_vis = draw_matches(
            img1, img2,
            mres.keypoints1, mres.keypoints2,
            mres.good_matches, inlier_mask,
        )
        save_image(root / o["matches_dir"] / f"{basename}_matches.jpg", match_vis)
        if H is not None:
            warped = warp_image(img1, H, img2.shape)
            overlay = overlay_warped(warped, img2)
            save_image(root / o["warped_dir"] / f"{basename}_warped.jpg", overlay)

    matcher_name = type(matcher).__name__

    return {
        "pair_name": basename,
        "summer_path": str(summer_path),
        "winter_path": str(winter_path),
        "matcher": matcher_name,
        "data_loading_ms": round((t1 - t0) * 1000, 3),
        "feature_matching_ms": round((t2 - t1) * 1000, 3),
        "matrix_estimation_ms": round((t3 - t2) * 1000, 3),
        "total_ms": round((t3 - t0) * 1000, 3),
        "num_keypoints_1": len(mres.keypoints1),
        "num_keypoints_2": len(mres.keypoints2),
        "num_good_matches": len(mres.good_matches),
        "inliers": meta["inliers"],
        "inlier_ratio": round(float(meta["inlier_ratio"]), 4),
        "rmse": None if rmse is None else round(float(rmse), 4),
        "status": meta["status"],
        "scale1": scale1,
        "scale2": scale2,
    }
