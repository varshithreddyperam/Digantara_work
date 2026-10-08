"""End-to-End Pipeline for Digantara SSA Assessment.

Executes:
1. FITS ingestion & metadata extraction
2. 2D background subtraction & noise estimation
3. Full-resolution hysteresis detection & morphology classification
4. Exact 1024x1024 right/bottom padded tiling
5. YOLO Ultralytics polygon segmentation export & semantic mask generation
6. Exact tile reassembly & reconstruction verification
7. Visual overlay generation & validation summary
"""

import os
import sys
import glob
import json
import csv
import time
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
from src.fits_loader import FITSLoader
from src.preprocessor import Preprocessor
from src.tiler import ImageTiler
from src.detector import SourceDetector
from src.classifier import SourceClassifier
from src.yolo_exporter import YoloExporter
from src.visualizer import Visualizer
from src.reconstructor import Reconstructor

def process_single_image(fits_path: str, config: dict, out_base: str):
    image_id = os.path.splitext(os.path.basename(fits_path))[0]
    is_val = image_id in config["yolo"]["val_images"]
    split = "val" if is_val else "train"
    
    print(f"\n=======================================================")
    print(f"Processing {image_id} (Assigned split: {split})")
    print(f"=======================================================")
    
    start_t = time.time()
    
    # 1. Load FITS
    raw_data, meta = FITSLoader.load(fits_path)
    h, w = raw_data.shape
    print(f"1. Loaded FITS: {h}x{w}, Dtype: {raw_data.dtype}, Range: [{meta['raw_min']}, {meta['raw_max']}]")
    
    # 2. Preprocess
    prep_cfg = config["preprocessing"]
    preprocessor = Preprocessor(
        grid_size=prep_cfg["grid_size"],
        p_low=prep_cfg["p_low"],
        p_high=prep_cfg["p_high"]
    )
    bg_map, med_bg, noise_sigma = preprocessor.estimate_background_and_noise(raw_data)
    subtracted, display_uint8, prep_stats = preprocessor.prepare_scientific_and_display(raw_data, bg_map)
    print(f"2. Preprocessed: Med BG={med_bg:.2f}, Noise Sigma={noise_sigma:.2f}, Saturated Pixels={prep_stats['num_saturated']}")
    
    # 3. Detect and Classify at Full Resolution
    det_cfg = config["detection"]
    cls_cfg = config["classification"]
    classifier = SourceClassifier(
        min_star_area=cls_cfg["min_star_area"],
        min_streak_length=cls_cfg["min_streak_length"],
        streak_axis_ratio_thresh=cls_cfg["streak_axis_ratio_thresh"],
        max_star_axis_ratio=cls_cfg["max_star_axis_ratio"],
    )
    detector = SourceDetector(
        k_core=det_cfg["k_core"],
        k_boundary=det_cfg["k_boundary"],
        min_pixels=det_cfg["min_pixels"],
        max_pixels=det_cfg["max_pixels"],
        classifier=classifier
    )
    semantic_mask, instances = detector.detect(subtracted, noise_sigma)
    
    num_stars = sum(1 for inst in instances if inst["class_id"] == 0)
    num_streaks = sum(1 for inst in instances if inst["class_id"] == 1)
    num_review = sum(1 for inst in instances if inst.get("needs_review", False))
    print(f"3. Detection: {len(instances)} sources ({num_stars} stars/blobs, {num_streaks} objects/streaks, {num_review} review flags)")
    
    # 4. Tiling Setup
    tiler = ImageTiler(tile_size=config["tiling"]["tile_size"])
    padded_display, valid_mask, tile_params = tiler.pad_image(display_uint8, pad_value=0)
    padded_mask, _, _ = tiler.pad_image(semantic_mask, pad_value=0)
    print(f"4. Padded to {tile_params['padded_height']}x{tile_params['padded_width']} "
          f"(pad_right={tile_params['pad_right']}, pad_bottom={tile_params['pad_bottom']}) -> {tile_params['num_tiles']} tiles")
    
    # Extract tiles
    display_tiles = tiler.extract_tiles(padded_display, image_id, h, w)
    mask_tiles = tiler.extract_tiles(padded_mask, image_id, h, w)
    
    # Directories for YOLO output and artifacts
    yolo_img_dir = os.path.join(out_base, "dataset", "images", split)
    yolo_lbl_dir = os.path.join(out_base, "dataset", "labels", split)
    tile_mask_dir = os.path.join(out_base, "tiles", "masks", image_id)
    tile_overlay_dir = os.path.join(out_base, "tiles", "overlays", image_id)
    manifest_dir = os.path.join(out_base, "manifests")
    review_crop_dir = os.path.join(out_base, "review_crops", image_id)
    
    for d in [yolo_img_dir, yolo_lbl_dir, tile_mask_dir, tile_overlay_dir, manifest_dir, review_crop_dir]:
        os.makedirs(d, exist_ok=True)
        
    manifest_rows = []
    tile_tuples_for_recon = []
    tile_mask_tuples_for_recon = []
    
    eps = config["yolo"]["polygon_epsilon"]
    min_poly_area = config["yolo"]["min_polygon_area"]
    
    for idx, ((dtile, meta), (mtile, _)) in enumerate(zip(display_tiles, mask_tiles)):
        tile_id = meta["tile_id"]
        tile_img_path = os.path.join(yolo_img_dir, f"{tile_id}.png")
        tile_lbl_path = os.path.join(yolo_lbl_dir, f"{tile_id}.txt")
        tile_msk_path = os.path.join(tile_mask_dir, f"{tile_id}_mask.png")
        
        # Save tile image and mask
        cv2.imwrite(tile_img_path, dtile)
        cv2.imwrite(tile_msk_path, mtile)
        
        # Extract YOLO polygon labels for this tile
        polys = YoloExporter.mask_to_yolo_polygons(mtile, epsilon=eps, min_area=min_poly_area)
        YoloExporter.write_yolo_label_file(tile_lbl_path, polys)
        
        # Count classes in tile
        stars_in_tile = sum(1 for cid, _ in polys if cid == 0)
        streaks_in_tile = sum(1 for cid, _ in polys if cid == 1)
        
        meta["num_star_annotations"] = stars_in_tile
        meta["num_streak_annotations"] = streaks_in_tile
        meta["num_polygons"] = len(polys)
        manifest_rows.append(meta)
        
        # Tile visual overlay for representative tiles (or boundary tiles with detections)
        if len(polys) > 0 or meta["is_boundary_tile"]:
            toverlay = Visualizer.create_overlay(dtile, mtile)
            cv2.imwrite(os.path.join(tile_overlay_dir, f"{tile_id}_overlay.png"), toverlay)
            
        tile_tuples_for_recon.append((dtile, meta))
        tile_mask_tuples_for_recon.append((mtile, meta))
        
    # Write manifest files
    manifest_json_path = os.path.join(manifest_dir, f"{image_id}_manifest.json")
    with open(manifest_json_path, "w") as fp:
        json.dump(manifest_rows, fp, indent=2)
        
    manifest_csv_path = os.path.join(manifest_dir, f"{image_id}_manifest.csv")
    with open(manifest_csv_path, "w", newline="") as fp:
        fieldnames = list(manifest_rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    print(f"5. Exported 70 tile image/label pairs and manifests.")
    
    # 5. Full-Size Reconstruction & Verification
    recon_img = tiler.reconstruct_from_tiles(tile_tuples_for_recon, h, w)
    recon_mask = tiler.reconstruct_from_tiles(tile_mask_tuples_for_recon, h, w)
    
    recon_img_verif = Reconstructor.verify_reconstruction(display_uint8, recon_img)
    recon_mask_verif = Reconstructor.verify_reconstruction(semantic_mask, recon_mask)
    
    print(f"6. Reconstruction Verification:")
    print(f"   Image bit-for-bit exact match: {recon_img_verif['is_exact_match']} (Max diff: {recon_img_verif['max_difference']})")
    print(f"   Mask bit-for-bit exact match:  {recon_mask_verif['is_exact_match']} (Max diff: {recon_mask_verif['max_difference']})")
    
    # 6. Save Full Reconstructed Imagery and Labels
    recon_dir = os.path.join(out_base, "reconstructed")
    os.makedirs(os.path.join(recon_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(recon_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(recon_dir, "overlays"), exist_ok=True)
    os.makedirs(os.path.join(recon_dir, "labels"), exist_ok=True)
    
    recon_img_path = os.path.join(recon_dir, "images", f"{image_id}.png")
    recon_mask_path = os.path.join(recon_dir, "masks", f"{image_id}_mask.png")
    recon_label_path = os.path.join(recon_dir, "labels", f"{image_id}.txt")
    recon_overlay_path = os.path.join(recon_dir, "overlays", f"{image_id}_overlay.png")
    
    cv2.imwrite(recon_img_path, recon_img)
    cv2.imwrite(recon_mask_path, recon_mask)
    
    # Full image YOLO labels
    full_polys = YoloExporter.mask_to_yolo_polygons(recon_mask, epsilon=eps, min_area=min_poly_area)
    YoloExporter.write_yolo_label_file(recon_label_path, full_polys)
    
    # Full overlay
    full_overlay = Visualizer.create_overlay(recon_img, recon_mask)
    cv2.imwrite(recon_overlay_path, full_overlay)
    
    # 7. Extract Representative Review Crops (Streaks & Ambiguous Cases)
    review_queue = []
    # All streaks
    streak_instances = [inst for inst in instances if inst["class_id"] == 1]
    for idx, inst in enumerate(streak_instances):
        cx, cy = int(inst["centroid"][0]), int(inst["centroid"][1])
        r = 128
        y1, y2 = max(0, cy - r), min(h, cy + r)
        x1, x2 = max(0, cx - r), min(w, cx + r)
        crop_raw = raw_data[y1:y2, x1:x2]
        crop_disp = display_uint8[y1:y2, x1:x2]
        crop_over = full_overlay[y1:y2, x1:x2]
        crop_path = os.path.join(review_crop_dir, f"streak_{idx+1}_inst{inst['instance_id']}.png")
        Visualizer.save_comparison_crop(crop_raw, crop_disp, crop_over, crop_path)
        
        review_entry = {
            "image_id": image_id,
            "instance_id": inst["instance_id"],
            "class_id": 1,
            "class_name": "object_streak",
            "bbox": inst["bbox"],
            "centroid": inst["centroid"],
            "morphology": inst["morphology"],
            "confidence": inst["confidence"],
            "crop_path": crop_path,
        }
        review_queue.append(review_entry)
        
    # Ambiguous review instances
    flagged_instances = [inst for inst in instances if inst.get("needs_review", False)]
    for idx, inst in enumerate(flagged_instances[:10]):
        cx, cy = int(inst["centroid"][0]), int(inst["centroid"][1])
        r = 128
        y1, y2 = max(0, cy - r), min(h, cy + r)
        x1, x2 = max(0, cx - r), min(w, cx + r)
        crop_raw = raw_data[y1:y2, x1:x2]
        crop_disp = display_uint8[y1:y2, x1:x2]
        crop_over = full_overlay[y1:y2, x1:x2]
        crop_path = os.path.join(review_crop_dir, f"review_{idx+1}_inst{inst['instance_id']}.png")
        Visualizer.save_comparison_crop(crop_raw, crop_disp, crop_over, crop_path)
        
    elapsed = time.time() - start_t
    print(f"Completed {image_id} in {elapsed:.1f}s")
    
    return {
        "image_id": image_id,
        "split": split,
        "width": w,
        "height": h,
        "noise_sigma": noise_sigma,
        "bg_median": med_bg,
        "num_stars": num_stars,
        "num_streaks": num_streaks,
        "num_review": num_review,
        "reconstruction_image_exact": recon_img_verif["is_exact_match"],
        "reconstruction_mask_exact": recon_mask_verif["is_exact_match"],
        "num_tiles": len(display_tiles),
        "full_polygons": len(full_polys),
        "review_entries": review_queue,
        "elapsed_sec": round(elapsed, 2),
    }

def main():
    with open("configs/pipeline_config.yaml") as fp:
        config = yaml.safe_load(fp)
        
    out_base = "data/processed"
    os.makedirs(out_base, exist_ok=True)
    
    # Save dataset.yaml
    yolo_yaml_path = os.path.join(out_base, "dataset", "dataset.yaml")
    YoloExporter.save_dataset_yaml(yolo_yaml_path, data_root_relative=".")
    
    fits_files = sorted(glob.glob(os.path.join(config["data"]["fits_dir"], "*.fits")))
    print(f"Starting end-to-end processing of {len(fits_files)} FITS images...")
    
    all_summaries = []
    total_start = time.time()
    
    for f in fits_files:
        summary = process_single_image(f, config, out_base)
        all_summaries.append(summary)
        
    total_elapsed = time.time() - total_start
    print(f"\n=======================================================")
    print(f"PIPELINE COMPLETE in {total_elapsed:.1f}s ({total_elapsed/60.0:.2f} mins)")
    print(f"=======================================================")
    
    # Save validation & pipeline summary
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)
    summary_report = {
        "pipeline_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_images_processed": len(all_summaries),
        "total_tiles_generated": sum(s["num_tiles"] for s in all_summaries),
        "total_stars_detected": sum(s["num_stars"] for s in all_summaries),
        "total_streaks_detected": sum(s["num_streaks"] for s in all_summaries),
        "total_review_flags": sum(s["num_review"] for s in all_summaries),
        "all_reconstructions_lossless": all(s["reconstruction_image_exact"] and s["reconstruction_mask_exact"] for s in all_summaries),
        "image_summaries": all_summaries,
    }
    
    with open(os.path.join(docs_dir, "pipeline_summary.json"), "w") as fp:
        json.dump(summary_report, fp, indent=2)
    print("Saved pipeline summary to docs/pipeline_summary.json")

if __name__ == "__main__":
    main()
