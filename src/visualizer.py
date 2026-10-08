"""Visualizer module to render RGB annotation overlays and comparison figures."""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2

class Visualizer:
    """Renders pixel-level masks and bounding boxes onto display imagery."""

    # Colors in BGR format
    # Class 0 (Star/Blob): Electric Blue / Cyan
    COLOR_STAR = (245, 130, 40)
    # Class 1 (Streak): Bright Magenta / Pink-Red
    COLOR_STREAK = (140, 40, 255)

    @classmethod
    def create_overlay(
        cls,
        display_gray: np.ndarray,
        semantic_mask: np.ndarray,
        alpha: float = 0.55,
        draw_contours: bool = True
    ) -> np.ndarray:
        """
        Creates an RGB overlay of semantic masks on a grayscale display image.
        
        Args:
            display_gray: 8-bit grayscale image (H, W).
            semantic_mask: 2D mask (0=bg, 1=star, 2=streak).
            alpha: Opacity of the mask overlay.
            draw_contours: Whether to outline detections.
            
        Returns:
            RGB/BGR 3-channel uint8 array.
        """
        if len(display_gray.shape) == 2:
            base_rgb = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)
        else:
            base_rgb = display_gray.copy()
            
        colored_layer = base_rgb.copy()
        
        # Apply star color
        star_mask = (semantic_mask == 1)
        if np.any(star_mask):
            colored_layer[star_mask] = cls.COLOR_STAR
            
        # Apply streak color
        streak_mask = (semantic_mask == 2)
        if np.any(streak_mask):
            colored_layer[streak_mask] = cls.COLOR_STREAK
            
        # Blend
        has_detection = star_mask | streak_mask
        overlay = base_rgb.copy()
        overlay[has_detection] = cv2.addWeighted(
            colored_layer, alpha, base_rgb, 1.0 - alpha, 0
        )[has_detection]
        
        # Draw contour boundaries for crisp delineation
        if draw_contours:
            if np.any(star_mask):
                cnts_star, _ = cv2.findContours(
                    star_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, cnts_star, -1, cls.COLOR_STAR, 1)
            if np.any(streak_mask):
                cnts_streak, _ = cv2.findContours(
                    streak_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, cnts_streak, -1, cls.COLOR_STREAK, 1)
                
        return overlay

    @classmethod
    def save_comparison_crop(
        cls,
        raw_crop: np.ndarray,
        preprocessed_crop: np.ndarray,
        overlay_crop: np.ndarray,
        output_path: str,
        labels: Tuple[str, str, str] = ("Raw FITS", "Preprocessed", "Annotated Mask")
    ) -> None:
        """Saves side-by-side comparison image of raw vs preprocessed vs annotated mask."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        h, w = raw_crop.shape[:2]
        
        def to_bgr(img):
            if len(img.shape) == 2:
                # normalize if not uint8
                if img.dtype != np.uint8:
                    p1, p99 = np.percentile(img, [1, 99.8])
                    norm = np.clip((img - p1) / (max(p99, p1 + 1) - p1) * 255.0, 0, 255).astype(np.uint8)
                else:
                    norm = img
                return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
            return img
            
        bgr_raw = to_bgr(raw_crop)
        bgr_prep = to_bgr(preprocessed_crop)
        bgr_over = to_bgr(overlay_crop)
        
        # Put labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for img, lbl in [(bgr_raw, labels[0]), (bgr_prep, labels[1]), (bgr_over, labels[2])]:
            cv2.putText(img, lbl, (15, 30), font, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, lbl, (15, 30), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            
        combined = np.hstack([bgr_raw, bgr_prep, bgr_over])
        cv2.imwrite(output_path, combined)
