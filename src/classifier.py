"""Classification module to distinguish stars/blobs (Class 0) from objects/streaks (Class 1)."""

import math
from typing import Dict, Any, Tuple
import numpy as np
import cv2

class SourceClassifier:
    """Classifies detected connected components into Star/Blob (0) or Object/Streak (1)."""

    def __init__(
        self,
        min_star_area: int = 3,
        min_streak_length: float = 16.0,
        streak_axis_ratio_thresh: float = 2.0,
        streak_eccentricity_thresh: float = 0.82,
        max_star_axis_ratio: float = 1.65,
    ):
        self.min_star_area = min_star_area
        self.min_streak_length = min_streak_length
        self.streak_axis_ratio_thresh = streak_axis_ratio_thresh
        self.streak_eccentricity_thresh = streak_eccentricity_thresh
        self.max_star_axis_ratio = max_star_axis_ratio

    def compute_morphology(
        self, contour: np.ndarray, mask_patch: np.ndarray, intensity_patch: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Computes geometric and moment-based morphology features.
        
        Args:
            contour: OpenCV contour points (N, 1, 2).
            mask_patch: Binary mask of the component.
            intensity_patch: Optional intensity values.
            
        Returns:
            Dictionary of morphological measurements.
        """
        area = float(cv2.contourArea(contour))
        if area == 0.0:
            area = float(np.sum(mask_patch))
            
        perimeter = float(cv2.arcLength(contour, closed=True))
        circularity = (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        
        # Bounding boxes
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox_aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        
        # Rotated min area rectangle
        rect = cv2.minAreaRect(contour)
        (rcx, rcy), (rw, rh), angle = rect
        rect_len = max(rw, rh)
        rect_width = min(rw, rh)
        rect_aspect = rect_len / (rect_width + 1e-5)
        
        # Second order central moments
        moments = cv2.moments(contour)
        m00 = moments["m00"] if moments["m00"] != 0 else 1.0
        mu20 = moments["mu20"] / m00
        mu02 = moments["mu02"] / m00
        mu11 = moments["mu11"] / m00
        
        # Eigenvalues of covariance
        term1 = (mu20 + mu02) / 2.0
        term2 = math.sqrt(max(0.0, ((mu20 - mu02) / 2.0) ** 2 + mu11 ** 2))
        lambda1 = max(0.0, term1 + term2)
        lambda2 = max(0.0, term1 - term2)
        
        semi_major = 2.0 * math.sqrt(lambda1)
        semi_minor = 2.0 * math.sqrt(lambda2)
        axis_ratio = semi_major / (semi_minor + 1e-4)
        eccentricity = math.sqrt(max(0.0, 1.0 - (lambda2 / (lambda1 + 1e-6))))
        
        # Intensity profile along principal axis (if patch provided)
        peak_intensity = float(np.max(intensity_patch)) if intensity_patch is not None and intensity_patch.size > 0 else 0.0
        mean_intensity = float(np.mean(intensity_patch[mask_patch > 0])) if intensity_patch is not None and np.any(mask_patch > 0) else 0.0
        
        return {
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "bbox_aspect": bbox_aspect,
            "rect_len": rect_len,
            "rect_width": rect_width,
            "rect_aspect": rect_aspect,
            "semi_major": semi_major,
            "semi_minor": semi_minor,
            "axis_ratio": axis_ratio,
            "eccentricity": eccentricity,
            "orientation_angle": angle,
            "peak_intensity": peak_intensity,
            "mean_intensity": mean_intensity,
        }

    def classify(
        self, morph: Dict[str, float], local_psf_fwhm: float = 3.5
    ) -> Tuple[int, str, bool, float]:
        """
        Classifies feature into Class 0 (star/blob) or Class 1 (object/streak).
        
        Args:
            morph: Morphology dictionary from compute_morphology.
            local_psf_fwhm: Expected PSF FWHM of point sources in the vicinity.
            
        Returns:
            Tuple of (class_id, class_name, needs_review, confidence_score).
        """
        area = morph["area"]
        axis_ratio = morph["axis_ratio"]
        ecc = morph["eccentricity"]
        rect_len = morph["rect_len"]
        rect_aspect = morph["rect_aspect"]
        circ = morph["circularity"]
        
        # Hot pixel / noise reject
        if area < self.min_star_area:
            return -1, "noise_artifact", False, 0.0
            
        # Decision logic for streak
        # 1. Clear elongated streak
        is_long_streak = (rect_len >= self.min_streak_length) and (axis_ratio >= self.streak_axis_ratio_thresh or rect_aspect >= self.streak_axis_ratio_thresh)
        # 2. Short/fat streak:
        # Length is notably larger than PSF diameter, aspect ratio >= 1.85, low circularity
        is_short_fat_streak = (
            (rect_len >= max(14.0, 2.5 * local_psf_fwhm))
            and (axis_ratio >= 1.85 or rect_aspect >= 1.85)
            and (circ < 0.65)
        )
        
        if is_long_streak or is_short_fat_streak:
            # Check if this could be an unresolved close binary star pair
            # Binary stars have dumbbell shape with two peaks, but streaks are continuous
            confidence = min(0.99, 0.5 + (axis_ratio / 10.0) + (rect_len / 100.0))
            return 1, "object_streak", False, confidence
            
        # Check ambiguous zone
        is_ambiguous = (
            (1.55 <= axis_ratio < self.streak_axis_ratio_thresh)
            and (rect_len >= 12.0)
            and (circ < 0.55)
        )
        
        if is_ambiguous:
            # Flag for review, default to streak if rect_aspect > 1.8 else star
            if rect_aspect >= 1.8:
                return 1, "object_streak", True, 0.60
            else:
                return 0, "star_blob", True, 0.60
                
        # Star/blob criteria
        # Point-source blobs are compact, circularity typically > 0.4 (for discretized small pixels),
        # axis ratio <= max_star_axis_ratio
        confidence = min(0.99, 0.7 + circ * 0.3)
        return 0, "star_blob", False, confidence
