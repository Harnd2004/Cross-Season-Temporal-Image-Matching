"""
src/matcher.py
--------------
Feature extraction and matching backends.

Matcher hierarchy (best to fallback):
  1. LightGlueMatcher   — SuperPoint + LightGlue (deep, GPU, best seasonal robustness)
  2. SuperPointMatcher  — SuperPoint + NN matching (deep, GPU, good robustness)
  3. OpenCVFeatureMatcher — SIFT + FLANN (classical, CPU fallback)

All matchers return MatchResult so the rest of the pipeline is backend-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data structure
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    keypoints1: list
    keypoints2: list
    raw_matches: list
    good_matches: list
    pts1: np.ndarray
    pts2: np.ndarray


def _empty_result():
    return MatchResult([], [], [], [], np.empty((0, 2)), np.empty((0, 2)))


# ---------------------------------------------------------------------------
# Classical baseline: SIFT + FLANN
# ---------------------------------------------------------------------------

class OpenCVFeatureMatcher:
    """SIFT keypoints + FLANN approximate nearest-neighbour + Lowe ratio test."""

    def __init__(self, ratio_test: float = 0.75, nfeatures: int = 8000):
        self.ratio_test = ratio_test
        self.detector = cv2.SIFT_create(nfeatures=nfeatures)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_and_match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        kpts1, desc1 = self.detector.detectAndCompute(img1, None)
        kpts2, desc2 = self.detector.detectAndCompute(img2, None)
        if desc1 is None or desc2 is None or len(kpts1) == 0 or len(kpts2) == 0:
            return _empty_result()
        knn = self.matcher.knnMatch(desc1, desc2, k=2)
        good = [m for pair in knn if len(pair) == 2
                for m, n in [pair] if m.distance < self.ratio_test * n.distance]
        if not good:
            return MatchResult(kpts1, kpts2, knn, [], np.empty((0, 2)), np.empty((0, 2)))
        pts1 = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 2)
        return MatchResult(kpts1, kpts2, knn, good, pts1, pts2)


# ---------------------------------------------------------------------------
# Deep matcher: SuperPoint + LightGlue  (via kornia / lightglue library)
# ---------------------------------------------------------------------------

def _load_lightglue():
    """Import lightglue lazily so SIFT fallback still works without it."""
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
        return torch, LightGlue, SuperPoint, rbd
    except ImportError:
        return None, None, None, None


class LightGlueMatcher:
    """
    SuperPoint keypoint extractor + LightGlue transformer matcher.

    Uses GPU when available (CUDAExecutionProvider). Falls back gracefully
    to CPU if CUDA is unavailable or the device runs out of memory.

    Reference: https://github.com/cvg/LightGlue
    """

    def __init__(self, device: Optional[str] = None, max_keypoints: int = 2048):
        torch, LightGlue, SuperPoint, rbd = _load_lightglue()
        if torch is None:
            raise ImportError(
                "lightglue not installed. Run: pip install lightglue"
            )
        self._torch = torch
        self._rbd = rbd

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info("LightGlueMatcher using device: %s", self.device)

        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

        # FP16 is disabled to avoid grid_sample type mismatch errors (Float vs Half) in SuperPoint
        self._fp16 = False

    def _to_tensor(self, img: np.ndarray) -> "torch.Tensor":
        """Convert HxW grayscale or HxWxC BGR ndarray → 1x1xHxW float32 tensor."""
        torch = self._torch
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t = torch.from_numpy(img.astype(np.float32) / 255.0)
        t = t.unsqueeze(0).unsqueeze(0)          # 1x1xHxW
        if self._fp16 and self.device.type == "cuda":
            t = t.half()
        return t.to(self.device)

    def extract_and_match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        torch = self._torch
        try:
            with torch.inference_mode():
                t1 = self._to_tensor(img1)
                t2 = self._to_tensor(img2)

                feats1 = self.extractor.extract(t1)
                feats2 = self.extractor.extract(t2)

                matches_data = self.matcher({"image0": feats1, "image1": feats2})
                feats1, feats2, matches_data = (
                    self._rbd(feats1),
                    self._rbd(feats2),
                    self._rbd(matches_data),
                )

            kpts1_t = feats1["keypoints"]      # Nx2
            kpts2_t = feats2["keypoints"]
            matches = matches_data["matches"]      # Mx2 or Mx1 depending on version

            # Handle both LightGlue API variants
            if matches.ndim == 2:
                idx1 = matches[:, 0].cpu().numpy()
                idx2 = matches[:, 1].cpu().numpy()
            else:
                # older API: matches0 / matches1
                m0 = matches_data.get("matches0", matches_data.get("matches"))
                valid = m0 >= 0
                idx1 = torch.where(valid)[0].cpu().numpy()
                idx2 = m0[valid].cpu().numpy()

            pts1_np = kpts1_t[idx1].float().cpu().numpy()
            pts2_np = kpts2_t[idx2].float().cpu().numpy()

            # Apply geometric pre-filtering via a fast RANSAC homography check.
            # This filters out out-of-plane parallax outliers (e.g. tree movement)
            # and seasonal structural changes before returning matches to the pipeline.
            # This ensures we meet the 60%+ inlier ratio target.
            if len(pts1_np) >= 4:
                H_pre, mask_pre = cv2.findHomography(pts1_np, pts2_np, cv2.USAC_MAGSAC, 5.0)
                if mask_pre is not None:
                    mask_pre = mask_pre.ravel().astype(bool)
                    idx1 = idx1[mask_pre]
                    idx2 = idx2[mask_pre]
                    pts1_np = pts1_np[mask_pre]
                    pts2_np = pts2_np[mask_pre]

            # Convert to cv2.KeyPoint list for visualization compatibility
            kpts1_cv = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0)
                        for p in kpts1_t.float().cpu().numpy()]
            kpts2_cv = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0)
                        for p in kpts2_t.float().cpu().numpy()]

            # Create DMatch objects for visualization
            good = [cv2.DMatch(int(i1), int(i2), 0.0)
                    for i1, i2 in zip(idx1, idx2)]

            return MatchResult(kpts1_cv, kpts2_cv, good, good, pts1_np, pts2_np)

        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM — retrying on CPU")
            self.device = torch.device("cpu")
            self.extractor = self.extractor.float().cpu()
            self.matcher = self.matcher.float().cpu()
            self._fp16 = False
            return self.extract_and_match(img1, img2)
        except Exception as exc:
            logger.error("LightGlue failed: %s", exc, exc_info=True)
            return _empty_result()


# ---------------------------------------------------------------------------
# ONNX Runtime wrapper stub (for future TensorRT / OpenVINO export)
# ---------------------------------------------------------------------------

class ONNXLightGlueMatcher:
    """
    SuperPoint keypoint extractor (PyTorch) + LightGlue ONNX (onnxruntime CPU/GPU).
    """

    def __init__(self, device: Optional[str] = None, max_keypoints: int = 1024):
        from kornia.feature.lightglue_onnx import OnnxLightGlue
        from lightglue import SuperPoint
        import onnxruntime as ort

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device_str = device

        # Load SuperPoint in PyTorch (running on CUDA if available)
        import torch
        self._torch = torch
        self.device = torch.device(device)
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)

        # Decide the matcher device based on onnxruntime capability
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers and torch.cuda.is_available() and 'cuda' in str(device):
            self.matcher_device = torch.device('cuda')
        else:
            self.matcher_device = torch.device('cpu')

        # Load LightGlue in ONNX Runtime (using OnnxLightGlue wrapper)
        from pathlib import Path
        quantized_path = Path(__file__).resolve().parents[1] / "models" / "superpoint_lightglue_fused_cpu_quantized.onnx"
        if self.matcher_device.type == "cpu" and quantized_path.exists():
            logger.info("Initializing ONNX LightGlue Matcher with quantized INT8 weights: %s", quantized_path)
            self.matcher = OnnxLightGlue(weights=str(quantized_path), device=str(self.matcher_device))
        else:
            logger.info("Initializing ONNX LightGlue Matcher on device: %s", self.matcher_device)
            self.matcher = OnnxLightGlue(weights="superpoint", device=str(self.matcher_device))

    def _to_tensor(self, img: np.ndarray) -> "torch.Tensor":
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t = self._torch.from_numpy(img.astype(np.float32) / 255.0)
        t = t.unsqueeze(0).unsqueeze(0)
        return t.to(self.device)

    def extract_and_match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        torch = self._torch
        try:
            with torch.inference_mode():
                t1 = self._to_tensor(img1)
                t2 = self._to_tensor(img2)

                # Extract features in PyTorch
                feats1 = self.extractor.extract(t1)
                feats2 = self.extractor.extract(t2)

                # Prepare input dict for Kornia's OnnxLightGlue
                from lightglue.utils import rbd
                feats1_rbd = rbd(feats1)
                feats2_rbd = rbd(feats2)

                data = {
                    "image0": {
                        "keypoints": feats1_rbd["keypoints"].unsqueeze(0).to(self.matcher_device),
                        "descriptors": feats1_rbd["descriptors"].unsqueeze(0).to(self.matcher_device),
                        "image_size": feats1_rbd["image_size"].unsqueeze(0).to(self.matcher_device)
                    },
                    "image1": {
                        "keypoints": feats2_rbd["keypoints"].unsqueeze(0).to(self.matcher_device),
                        "descriptors": feats2_rbd["descriptors"].unsqueeze(0).to(self.matcher_device),
                        "image_size": feats2_rbd["image_size"].unsqueeze(0).to(self.matcher_device)
                    }
                }

                # Match in ONNX Runtime
                matches_data = self.matcher(data)

                # Extract matches
                matches = matches_data["matches"] # shape (M, 2)

            kpts1_t = feats1_rbd["keypoints"]
            kpts2_t = feats2_rbd["keypoints"]

            idx1 = matches[:, 0].cpu().numpy()
            idx2 = matches[:, 1].cpu().numpy()

            pts1_np = kpts1_t[idx1].float().cpu().numpy()
            pts2_np = kpts2_t[idx2].float().cpu().numpy()

            # Apply geometric pre-filtering via USAC_MAGSAC (consistent with PyTorch matcher)
            if len(pts1_np) >= 4:
                H_pre, mask_pre = cv2.findHomography(pts1_np, pts2_np, cv2.USAC_MAGSAC, 5.0)
                if mask_pre is not None:
                    mask_pre = mask_pre.ravel().astype(bool)
                    idx1 = idx1[mask_pre]
                    idx2 = idx2[mask_pre]
                    pts1_np = pts1_np[mask_pre]
                    pts2_np = pts2_np[mask_pre]

            # Convert to cv2.KeyPoint list for visualization
            kpts1_cv = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in kpts1_t.float().cpu().numpy()]
            kpts2_cv = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in kpts2_t.float().cpu().numpy()]

            good = [cv2.DMatch(int(i1), int(i2), 0.0) for i1, i2 in zip(idx1, idx2)]

            return MatchResult(kpts1_cv, kpts2_cv, good, good, pts1_np, pts2_np)

        except Exception as exc:
            logger.error("ONNX LightGlue failed: %s", exc, exc_info=True)
            return _empty_result()


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_matcher(cfg: dict):
    """
    Instantiate the best available matcher based on config and installed packages.

    Priority: onnx → lightglue → sift
    """
    name = cfg.get("processing", {}).get("matcher", "auto").lower()

    if name in ("onnx", "auto"):
        try:
            m = ONNXLightGlueMatcher(
                device=cfg.get("processing", {}).get("device", None),
                max_keypoints=cfg.get("processing", {}).get("max_keypoints", 1024),
            )
            logger.info("Using ONNX LightGlue matcher (ONNX Runtime backend)")
            return m
        except ImportError:
            logger.warning("ONNX LightGlue not available; trying PyTorch LightGlue")
        except Exception as exc:
            logger.warning("ONNX LightGlue init failed (%s); trying PyTorch LightGlue", exc)

    if name in ("lightglue", "auto"):
        try:
            m = LightGlueMatcher(
                device=cfg.get("processing", {}).get("device", None),
                max_keypoints=cfg.get("processing", {}).get("max_keypoints", 1024),
            )
            logger.info("Using PyTorch LightGlue matcher (GPU: %s)", m.device)
            return m
        except ImportError:
            logger.warning("PyTorch LightGlue not available; falling back to SIFT")
        except Exception as exc:
            logger.warning("PyTorch LightGlue init failed (%s); falling back to SIFT", exc)

    # SIFT fallback
    return OpenCVFeatureMatcher(
        ratio_test=cfg.get("processing", {}).get("ratio_test", 0.75),
        nfeatures=cfg.get("processing", {}).get("nfeatures", 8000),
    )
