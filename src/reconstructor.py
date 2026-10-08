"""Reconstructor module to reassemble tiles into full-size imagery and masks."""

import os
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2

class Reconstructor:
    """Reassembles 1024x1024 tiles into full-size 9568x6380 images and verifies reconstruction integrity."""

    @staticmethod
    def reassemble_tiles(
        tile_manifest: List[Dict[str, Any]],
        tile_dir: str,
        file_suffix: str = ".png",
        is_color: bool = False
    ) -> np.ndarray:
        """
        Reassembles tiles into full original image using manifest parameters.
        
        Args:
            tile_manifest: List of tile metadata dictionaries.
            tile_dir: Directory containing individual tile files.
            file_suffix: Suffix of tile files.
            is_color: Whether tiles are 3-channel BGR.
            
        Returns:
            Reassembled 2D or 3D array matching original image dimensions.
        """
        if not tile_manifest:
            raise ValueError("Empty tile manifest provided.")
            
        first = tile_manifest[0]
        orig_w = first["orig_width"]
        orig_h = first["orig_height"]
        pad_w = first["tile_width"] * ((orig_w + first["tile_width"] - 1) // first["tile_width"])
        pad_h = first["tile_height"] * ((orig_h + first["tile_height"] - 1) // first["tile_height"])
        
        if is_color:
            canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((pad_h, pad_w), dtype=np.uint8)
            
        for meta in tile_manifest:
            tile_id = meta["tile_id"]
            tile_path = os.path.join(tile_dir, f"{tile_id}{file_suffix}")
            if not os.path.exists(tile_path):
                raise FileNotFoundError(f"Missing tile file: {tile_path}")
                
            if is_color:
                tile_img = cv2.imread(tile_path, cv2.IMREAD_COLOR)
            else:
                tile_img = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED)
                
            x0 = meta["x_offset"]
            y0 = meta["y_offset"]
            tw = meta["tile_width"]
            th = meta["tile_height"]
            
            canvas[y0 : y0 + th, x0 : x0 + tw] = tile_img
            
        # Crop away padding to exact original dimensions
        reconstructed = canvas[:orig_h, :orig_w]
        return reconstructed

    @staticmethod
    def verify_reconstruction(
        original: np.ndarray, reconstructed: np.ndarray
    ) -> Dict[str, Any]:
        """
        Verifies numerical and spatial fidelity between original and reconstructed arrays.
        """
        is_exact = np.array_equal(original, reconstructed)
        diff = np.abs(original.astype(np.int64) - reconstructed.astype(np.int64))
        max_diff = int(np.max(diff)) if diff.size > 0 else 0
        mean_diff = float(np.mean(diff)) if diff.size > 0 else 0.0
        
        return {
            "is_exact_match": bool(is_exact),
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "original_shape": list(original.shape),
            "reconstructed_shape": list(reconstructed.shape),
        }
