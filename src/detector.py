"""Astronomical source detector for SSA imagery producing pixel-accurate masks."""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from scipy.ndimage import label, find_objects
from src.classifier import SourceClassifier

class SourceDetector:
    """Detects astronomical point sources (stars/blobs) and tracks (streaks) at pixel level."""

    def __init__(
        self,
        k_core: float = 3.8,
        k_boundary: float = 2.0,
        min_pixels: int = 3,
        max_pixels: int = 50000,
        classifier: SourceClassifier = None,
    ):
        self.k_core = k_core
        self.k_boundary = k_boundary
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.classifier = classifier or SourceClassifier()

    def detect(
        self, subtracted: np.ndarray, noise_sigma: float, valid_mask: np.ndarray = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Runs multi-threshold detection and morphology classification.
        
        Args:
            subtracted: Background-subtracted 2D float32/uint16 array.
            noise_sigma: Estimated local background noise sigma.
            valid_mask: Optional boolean mask of valid image region.
            
        Returns:
            Tuple of (semantic_mask [0=bg, 1=star, 2=streak], list of instance dictionaries).
        """
        h, w = subtracted.shape
        t_core = max(8.0, self.k_core * noise_sigma)
        t_boundary = max(4.0, self.k_boundary * noise_sigma)
        
        # 1. Binary core thresholding to identify true sources (seeds)
        core_binary = (subtracted >= t_core).astype(np.uint8)
        
        # 2. Binary boundary thresholding to capture full isophotal extent
        boundary_binary = (subtracted >= t_boundary).astype(np.uint8)
        
        if valid_mask is not None:
            core_binary = core_binary & valid_mask.astype(np.uint8)
            boundary_binary = boundary_binary & valid_mask.astype(np.uint8)
            
        # Connected components on boundary binary
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            boundary_binary, connectivity=8
        )
        
        # Core presence check per component to eliminate false noise clumps
        # Compute max core value within each component
        # We can identify which labels contain at least one core pixel
        core_labels = np.unique(labels[core_binary > 0])
        core_set = set(core_labels)
        core_set.discard(0)  # remove background
        
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        instances = []
        inst_counter = 1
        
        # Estimate typical stellar PSF FWHM in image from compact bright components
        psf_samples = []
        for lbl in list(core_set)[:1000]:
            area = stats[lbl, cv2.CC_STAT_AREA]
            if 6 <= area <= 50:
                bw = stats[lbl, cv2.CC_STAT_WIDTH]
                bh = stats[lbl, cv2.CC_STAT_HEIGHT]
                if max(bw, bh) <= 10:
                    psf_samples.append((bw + bh) / 2.0)
                    
        local_psf_fwhm = float(np.median(psf_samples)) if psf_samples else 3.5
        
        for lbl in core_set:
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < self.min_pixels or area > self.max_pixels:
                continue
                
            x = stats[lbl, cv2.CC_STAT_LEFT]
            y = stats[lbl, cv2.CC_STAT_TOP]
            bw = stats[lbl, cv2.CC_STAT_WIDTH]
            bh = stats[lbl, cv2.CC_STAT_HEIGHT]
            
            # Extract component patch
            comp_mask = (labels[y : y + bh, x : x + bw] == lbl).astype(np.uint8)
            intensity_patch = subtracted[y : y + bh, x : x + bw]
            
            # Find contours within patch
            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
                
            contour = max(contours, key=cv2.contourArea)
            # Offset contour to global coordinates
            global_contour = contour + np.array([x, y])
            
            # Calculate morphology
            morph = self.classifier.compute_morphology(contour, comp_mask, intensity_patch)
            class_id, class_name, needs_review, confidence = self.classifier.classify(
                morph, local_psf_fwhm=local_psf_fwhm
            )
            
            if class_id < 0:
                # Filtered as noise artifact
                continue
                
            # Semantic mask mapping:
            # Class 0 (star_blob) -> semantic mask 1
            # Class 1 (object_streak) -> semantic mask 2
            mask_val = 1 if class_id == 0 else 2
            
            # Write to semantic mask
            mask_indices = np.where(comp_mask > 0)
            semantic_mask[y + mask_indices[0], x + mask_indices[1]] = mask_val
            
            # Create instance record
            instance_data = {
                "instance_id": inst_counter,
                "class_id": class_id,
                "class_name": class_name,
                "mask_value": mask_val,
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "centroid": [float(centroids[lbl][0]), float(centroids[lbl][1])],
                "area": int(area),
                "morphology": morph,
                "needs_review": needs_review,
                "confidence": round(float(confidence), 3),
                "contour": global_contour.squeeze().tolist(),  # [N, 2]
            }
            instances.append(instance_data)
            inst_counter += 1
            
        return semantic_mask, instances
