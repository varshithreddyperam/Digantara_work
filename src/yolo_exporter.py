"""YOLO Ultralytics segmentation format exporter and semantic mask serializer."""

import os
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import yaml

class YoloExporter:
    """Exports dataset into YOLO Ultralytics segmentation format and lossless masks."""

    CLASS_MAPPING = {
        0: "star_blob",
        1: "object_streak"
    }
    
    MASK_TO_CLASS = {
        1: 0,  # semantic mask 1 -> YOLO class 0 (star_blob)
        2: 1   # semantic mask 2 -> YOLO class 1 (object_streak)
    }

    @staticmethod
    def mask_to_yolo_polygons(
        mask: np.ndarray, epsilon: float = 1.0, min_area: float = 2.0
    ) -> List[Tuple[int, List[float]]]:
        """
        Extracts normalized YOLO segmentation polygon lines from a 2D semantic mask.
        
        Args:
            mask: 2D uint8 mask (0=bg, 1=star, 2=streak).
            epsilon: approxPolyDP tolerance for contour simplification.
            min_area: Minimum contour area.
            
        Returns:
            List of (class_id, [x1, y1, x2, y2, ... normalized in [0, 1]]).
        """
        h, w = mask.shape
        polygons = []
        
        for mask_val, class_id in YoloExporter.MASK_TO_CLASS.items():
            class_binary = (mask == mask_val).astype(np.uint8)
            if not np.any(class_binary):
                continue
                
            contours, _ = cv2.findContours(
                class_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
            )
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area and len(cnt) < 3:
                    continue
                    
                # Simplify polygon
                approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
                # Need at least 3 vertices
                if len(approx) < 3:
                    # Fall back to original contour if it had >=3 points
                    if len(cnt) >= 3:
                        approx = cnt
                    else:
                        # Envelope into small 4-point bounding box
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        approx = np.array([
                            [[bx, by]],
                            [[bx + max(1, bw), by]],
                            [[bx + max(1, bw), by + max(1, bh)]],
                            [[bx, by + max(1, bh)]]
                        ])
                        
                pts = approx.reshape(-1, 2)
                # Normalize coordinates to [0, 1]
                norm_coords = []
                for px, py in pts:
                    nx = float(np.clip(px / w, 0.0, 1.0))
                    ny = float(np.clip(py / h, 0.0, 1.0))
                    norm_coords.extend([round(nx, 6), round(ny, 6)])
                    
                if len(norm_coords) >= 6:
                    polygons.append((class_id, norm_coords))
                    
        return polygons

    @staticmethod
    def write_yolo_label_file(
        filepath: str, polygons: List[Tuple[int, List[float]]]
    ) -> None:
        """Writes YOLO Ultralytics format lines: class_id x1 y1 x2 y2 ..."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as fp:
            for class_id, coords in polygons:
                coord_str = " ".join(f"{c:.6f}" for c in coords)
                fp.write(f"{class_id} {coord_str}\n")

    @staticmethod
    def save_dataset_yaml(output_path: str, data_root_relative: str = ".") -> None:
        """Writes standard Ultralytics dataset.yaml configuration."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        config = {
            "path": data_root_relative,
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "star_blob",
                1: "object_streak"
            }
        }
        with open(output_path, "w") as fp:
            yaml.dump(config, fp, sort_keys=False)
