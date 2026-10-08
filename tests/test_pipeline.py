"""Unit tests for tiling, padding, reconstruction, and YOLO export."""

import pytest
import numpy as np
from src.tiler import ImageTiler
from src.classifier import SourceClassifier
from src.yolo_exporter import YoloExporter
from src.reconstructor import Reconstructor

def test_tiling_parameters():
    tiler = ImageTiler(tile_size=1024)
    params = tiler.compute_tiling_parameters(height=6380, width=9568)
    
    assert params["cols"] == 10
    assert params["rows"] == 7
    assert params["num_tiles"] == 70
    assert params["padded_width"] == 10240
    assert params["padded_height"] == 7168
    assert params["pad_right"] == 672
    assert params["pad_bottom"] == 788

def test_padding_and_reconstruction_lossless():
    tiler = ImageTiler(tile_size=1024)
    h, w = 6380, 9568
    # Create test synthetic image with gradient and distinct markers
    rng = np.random.RandomState(42)
    original_img = rng.randint(0, 255, size=(h, w), dtype=np.uint8)
    
    # Pad
    padded_img, valid_mask, params = tiler.pad_image(original_img, pad_value=0)
    assert padded_img.shape == (7168, 10240)
    assert np.all(valid_mask[:h, :w] == True)
    assert np.all(valid_mask[h:, :] == False)
    assert np.all(valid_mask[:, w:] == False)
    
    # Extract tiles
    tiles = tiler.extract_tiles(padded_img, "test_img", h, w)
    assert len(tiles) == 70
    for tile_arr, meta in tiles:
        assert tile_arr.shape == (1024, 1024)
        
    # Reconstruct
    reconstructed = tiler.reconstruct_from_tiles(tiles, h, w)
    assert reconstructed.shape == (h, w)
    
    # Verify bit-for-bit exactness
    verif = Reconstructor.verify_reconstruction(original_img, reconstructed)
    assert verif["is_exact_match"] is True
    assert verif["max_difference"] == 0

def test_yolo_polygon_conversion():
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    # Draw star blob (class 0 -> mask 1)
    mask[100:110, 100:110] = 1
    # Draw streak (class 1 -> mask 2)
    mask[200:208, 200:260] = 2
    
    polygons = YoloExporter.mask_to_yolo_polygons(mask, epsilon=1.0)
    assert len(polygons) == 2
    
    classes = [p[0] for p in polygons]
    assert 0 in classes
    assert 1 in classes
    
    for cls_id, coords in polygons:
        assert len(coords) >= 6  # at least 3 vertices (x, y pairs)
        for val in coords:
            assert 0.0 <= val <= 1.0

def test_classifier_streak_vs_blob():
    classifier = SourceClassifier()
    
    # Circular blob contour (Star)
    radius = 5
    blob_pts = []
    for theta in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        blob_pts.append([[int(10 + radius * np.cos(theta)), int(10 + radius * np.sin(theta))]])
    blob_cnt = np.array(blob_pts, dtype=np.int32)
    patch_blob = np.zeros((25, 25), dtype=np.uint8)
    morph_blob = classifier.compute_morphology(blob_cnt, patch_blob)
    cls_id, name, _, _ = classifier.classify(morph_blob, local_psf_fwhm=3.5)
    assert cls_id == 0
    assert name == "star_blob"
    
    # Linear elongated contour (Streak)
    streak_pts = np.array([[[10, 10]], [[60, 12]], [[60, 16]], [[10, 14]]], dtype=np.int32)
    patch_streak = np.zeros((30, 80), dtype=np.uint8)
    morph_streak = classifier.compute_morphology(streak_pts, patch_streak)
    cls_id_s, name_s, _, _ = classifier.classify(morph_streak, local_psf_fwhm=3.5)
    assert cls_id_s == 1
    assert name_s == "object_streak"
