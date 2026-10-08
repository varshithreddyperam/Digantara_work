"""Preprocessing module for SSA FITS imagery.

Provides:
- Spatially varying 2D background and noise estimation.
- Scientific background-subtracted intensity data.
- Contrast-enhanced 8-bit display representation using robust non-linear stretching.
- Saturated pixel detection.
"""

from typing import Tuple, Dict, Any
import numpy as np
import cv2

class Preprocessor:
    """Handles photometric background estimation, noise calculation, and visual stretch."""

    def __init__(self, grid_size: int = 128, p_low: float = 0.5, p_high: float = 99.8):
        """
        Args:
            grid_size: Cell size for 2D background mesh estimation.
            p_low: Lower percentile for display stretch.
            p_high: Upper percentile for display stretch.
        """
        self.grid_size = grid_size
        self.p_low = p_low
        self.p_high = p_high

    def estimate_background_and_noise(self, data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Estimates the spatially varying background 2D map and local noise sigma using MAD.
        
        Args:
            data: Raw 2D image array (uint16 or float32).
            
        Returns:
            Tuple of (bg_map, median_bg, noise_sigma).
        """
        h, w = data.shape
        data_f = data.astype(np.float32)
        
        # Downscale grid for fast, robust 2D background estimation
        gh = max(1, h // self.grid_size)
        gw = max(1, w // self.grid_size)
        
        # Block median approximation via area resize
        small = cv2.resize(data_f, (gw, gh), interpolation=cv2.INTER_AREA)
        # Upsample back with bilinear interpolation to obtain smooth 2D background
        bg_map = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate residual for noise estimation
        residual = data_f - bg_map
        # Sample pixels uniformly for speed in calculating MAD
        sample_res = residual[::4, ::4].ravel()
        med_res = float(np.median(sample_res))
        mad = float(np.median(np.abs(sample_res - med_res)))
        noise_sigma = mad * 1.4826
        
        # If noise_sigma is tiny (e.g. low-count quantized image), ensure a positive floor
        if noise_sigma < 0.5:
            std_sub = float(np.std(sample_res))
            noise_sigma = max(0.5, std_sub)
            
        median_bg = float(np.median(small))
        return bg_map, median_bg, noise_sigma

    def prepare_scientific_and_display(
        self, data: np.ndarray, bg_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Separates scientific background-subtracted data from contrast-enhanced display uint8 image.
        
        Args:
            data: Raw uint16 data.
            bg_map: Estimated background map.
            
        Returns:
            Tuple of (scientific_subtracted_data, display_image_uint8, stats_dict).
        """
        data_f = data.astype(np.float32)
        subtracted = np.clip(data_f - bg_map, 0.0, None)
        
        # Determine saturation threshold
        max_val = float(np.max(data))
        if max_val > 4096:
            sat_threshold = 65500.0
        else:
            sat_threshold = 4090.0
        saturated_mask = (data_f >= sat_threshold)
        
        # Contrast-enhanced display image (8-bit)
        # Sample for percentile calculation
        sample_sub = subtracted[::4, ::4].ravel()
        # Exclude exact zeros for percentile if background is zero-dominated
        nonzero_sample = sample_sub[sample_sub > 0]
        if len(nonzero_sample) > 100:
            v_min = float(np.percentile(nonzero_sample, self.p_low))
            v_max = float(np.percentile(nonzero_sample, self.p_high))
        else:
            v_min = 0.0
            v_max = float(np.percentile(sample_sub, 99.5))
            
        if v_max <= v_min:
            v_max = v_min + 10.0
            
        # Non-linear asinh stretch to preserve faint stars while maintaining bright streak details
        # x_norm = (I - v_min) / (v_max - v_min)
        scaled = np.clip((subtracted - v_min) / (v_max - v_min), 0.0, 10.0)
        # Asinh mapping
        stretched = np.arcsinh(scaled * 3.0) / np.arcsinh(30.0)
        display_uint8 = np.clip(stretched * 255.0, 0.0, 255.0).astype(np.uint8)
        
        stats = {
            "v_min": v_min,
            "v_max": v_max,
            "sat_threshold": sat_threshold,
            "num_saturated": int(np.sum(saturated_mask)),
        }
        
        return subtracted, display_uint8, stats
