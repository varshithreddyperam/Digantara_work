"""Tiling module for exact 1024x1024 division with boundary padding and lossless reconstruction."""

import math
from typing import List, Dict, Any, Tuple
import numpy as np

class ImageTiler:
    """Handles right/bottom padding, 1024x1024 tiling, manifest generation, and exact reconstruction."""

    def __init__(self, tile_size: int = 1024):
        self.tile_size = tile_size

    def compute_tiling_parameters(self, height: int, width: int) -> Dict[str, int]:
        """
        Computes grid columns, rows, padded dimensions, and padding amounts.
        """
        cols = math.ceil(width / self.tile_size)
        rows = math.ceil(height / self.tile_size)
        padded_w = cols * self.tile_size
        padded_h = rows * self.tile_size
        pad_right = padded_w - width
        pad_bottom = padded_h - height
        
        return {
            "orig_width": width,
            "orig_height": height,
            "tile_size": self.tile_size,
            "cols": cols,
            "rows": rows,
            "num_tiles": cols * rows,
            "padded_width": padded_w,
            "padded_height": padded_h,
            "pad_right": pad_right,
            "pad_bottom": pad_bottom,
        }

    def pad_image(
        self, image: np.ndarray, pad_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Pads image on right and bottom boundaries only.
        
        Args:
            image: 2D array of shape (H, W).
            pad_value: Constant value for padding.
            
        Returns:
            Tuple of (padded_image, valid_mask, tiling_params).
        """
        h, w = image.shape
        params = self.compute_tiling_parameters(h, w)
        pad_r = params["pad_right"]
        pad_b = params["pad_bottom"]
        
        padded_img = np.pad(
            image,
            ((0, pad_b), (0, pad_r)),
            mode="constant",
            constant_values=pad_value
        )
        
        valid_mask = np.zeros(padded_img.shape, dtype=bool)
        valid_mask[:h, :w] = True
        
        return padded_img, valid_mask, params

    def extract_tiles(
        self, padded_image: np.ndarray, image_id: str, orig_height: int, orig_width: int
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Extracts 1024x1024 non-overlapping tiles and their metadata.
        
        Returns:
            List of (tile_array, tile_metadata).
        """
        h, w = padded_image.shape
        params = self.compute_tiling_parameters(orig_height, orig_width)
        tiles = []
        
        for r in range(params["rows"]):
            for c in range(params["cols"]):
                y0 = r * self.tile_size
                x0 = c * self.tile_size
                tile = padded_image[y0 : y0 + self.tile_size, x0 : x0 + self.tile_size]
                
                # Compute valid pixel span inside tile
                valid_w = max(0, min(self.tile_size, orig_width - x0))
                valid_h = max(0, min(self.tile_size, orig_height - y0))
                
                tile_id = f"{image_id}_tile_r{r:02d}_c{c:02d}"
                meta = {
                    "image_id": image_id,
                    "tile_id": tile_id,
                    "row": r,
                    "col": c,
                    "x_offset": x0,
                    "y_offset": y0,
                    "tile_width": self.tile_size,
                    "tile_height": self.tile_size,
                    "orig_width": orig_width,
                    "orig_height": orig_height,
                    "pad_right": params["pad_right"],
                    "pad_bottom": params["pad_bottom"],
                    "valid_w": valid_w,
                    "valid_h": valid_h,
                    "is_boundary_tile": (r == params["rows"] - 1 or c == params["cols"] - 1),
                }
                tiles.append((tile, meta))
                
        return tiles

    def reconstruct_from_tiles(
        self, tiles: List[Tuple[np.ndarray, Dict[str, Any]]], orig_height: int, orig_width: int
    ) -> np.ndarray:
        """
        Reassembles tiles back into full-size array and removes right/bottom padding.
        
        Args:
            tiles: List of (tile_array, tile_metadata).
            orig_height: Original height in pixels.
            orig_width: Original width in pixels.
            
        Returns:
            Exact reconstructed 2D array of shape (orig_height, orig_width).
        """
        params = self.compute_tiling_parameters(orig_height, orig_width)
        canvas = np.zeros(
            (params["padded_height"], params["padded_width"]),
            dtype=tiles[0][0].dtype
        )
        
        for tile, meta in tiles:
            r = meta["row"]
            c = meta["col"]
            y0 = r * self.tile_size
            x0 = c * self.tile_size
            canvas[y0 : y0 + self.tile_size, x0 : x0 + self.tile_size] = tile
            
        # Crop away padding to exactly match original dimensions
        reconstructed = canvas[:orig_height, :orig_width]
        return reconstructed
