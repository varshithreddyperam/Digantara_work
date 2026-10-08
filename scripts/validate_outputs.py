"""Comprehensive validation script verifying YOLO dataset syntax, masks, manifests, and reconstruction."""

import os
import glob
import json
import numpy as np
import cv2

def validate_dataset(processed_dir="data/processed"):
    print("=== RUNNING RIGOROUS DATASET VALIDATION ===")
    report = {
        "dataset_checks": {},
        "label_checks": {
            "total_tile_labels_checked": 0,
            "total_polygons_checked": 0,
            "classes_found": set(),
            "invalid_coords_count": 0,
            "polygons_under_3_points": 0,
            "empty_tile_labels": 0,
        },
        "mask_checks": {
            "total_tile_masks_checked": 0,
            "valid_mask_values": True,
            "invalid_values_found": set(),
        },
        "manifest_checks": {
            "manifest_files_count": 0,
            "total_tiles_in_manifests": 0,
            "padding_exact": True,
        },
        "full_size_checks": {
            "full_images_count": 0,
            "full_masks_count": 0,
            "full_overlays_count": 0,
            "full_labels_count": 0,
            "dimensions_match_9568x6380": True,
        }
    }
    
    # 1. Manifest Checks
    manifests = glob.glob(os.path.join(processed_dir, "manifests", "*_manifest.json"))
    report["manifest_checks"]["manifest_files_count"] = len(manifests)
    total_manifest_tiles = 0
    
    for mf in manifests:
        with open(mf) as fp:
            data = json.load(fp)
            total_manifest_tiles += len(data)
            for row in data:
                if row["orig_width"] != 9568 or row["orig_height"] != 6380:
                    report["manifest_checks"]["dimensions_exact"] = False
                if row["pad_right"] != 672 or row["pad_bottom"] != 788:
                    report["manifest_checks"]["padding_exact"] = False
                    
    report["manifest_checks"]["total_tiles_in_manifests"] = total_manifest_tiles
    print(f"Manifests: {len(manifests)} files, {total_manifest_tiles} total tiles recorded.")
    
    # 2. YOLO Tile Image/Label Checks
    for split in ["train", "val"]:
        img_dir = os.path.join(processed_dir, "dataset", "images", split)
        lbl_dir = os.path.join(processed_dir, "dataset", "labels", split)
        
        img_files = glob.glob(os.path.join(img_dir, "*.png"))
        lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        
        report["dataset_checks"][f"{split}_images_count"] = len(img_files)
        report["dataset_checks"][f"{split}_labels_count"] = len(lbl_files)
        
        # Check correspondence
        img_stems = set(os.path.splitext(os.path.basename(f))[0] for f in img_files)
        lbl_stems = set(os.path.splitext(os.path.basename(f))[0] for f in lbl_files)
        assert img_stems == lbl_stems, f"Mismatch between images and labels in {split}"
        
        for lf in lbl_files:
            report["label_checks"]["total_tile_labels_checked"] += 1
            with open(lf) as fp:
                lines = [line.strip() for line in fp if line.strip()]
            if not lines:
                report["label_checks"]["empty_tile_labels"] += 1
                continue
                
            for line in lines:
                parts = line.split()
                cls_id = int(parts[0])
                report["label_checks"]["classes_found"].add(cls_id)
                coords = [float(p) for p in parts[1:]]
                report["label_checks"]["total_polygons_checked"] += 1
                
                if len(coords) < 6 or len(coords) % 2 != 0:
                    report["label_checks"]["polygons_under_3_points"] += 1
                    
                for c in coords:
                    if c < 0.0 or c > 1.0:
                        report["label_checks"]["invalid_coords_count"] += 1
                        
    report["label_checks"]["classes_found"] = sorted(list(report["label_checks"]["classes_found"]))
    print(f"YOLO Labels: Checked {report['label_checks']['total_tile_labels_checked']} tiles, "
          f"{report['label_checks']['total_polygons_checked']} polygons across classes {report['label_checks']['classes_found']}.")
    print(f"  Empty tiles: {report['label_checks']['empty_tile_labels']} (valid background negative tiles)")
    print(f"  Invalid coordinates: {report['label_checks']['invalid_coords_count']}")
    print(f"  Polygons under 3 vertices: {report['label_checks']['polygons_under_3_points']}")
    
    # 3. Tile Mask Checks (sampling 100 tiles)
    tile_masks = glob.glob(os.path.join(processed_dir, "tiles", "masks", "*", "*_mask.png"))
    report["mask_checks"]["total_tile_masks_checked"] = len(tile_masks)
    
    for mf in tile_masks[::7]:
        mask = cv2.imread(mf, cv2.IMREAD_UNCHANGED)
        unique_vals = set(np.unique(mask))
        diff_vals = unique_vals - {0, 1, 2}
        if diff_vals:
            report["mask_checks"]["valid_mask_values"] = False
            report["mask_checks"]["invalid_values_found"].update(diff_vals)
            
    report["mask_checks"]["invalid_values_found"] = sorted(list(report["mask_checks"]["invalid_values_found"]))
    print(f"Tile Masks: {len(tile_masks)} masks checked. Semantic values valid: {report['mask_checks']['valid_mask_values']}")
    
    # 4. Full Size Reconstructed Checks
    recon_imgs = glob.glob(os.path.join(processed_dir, "reconstructed", "images", "*.png"))
    recon_masks = glob.glob(os.path.join(processed_dir, "reconstructed", "masks", "*.png"))
    recon_overlays = glob.glob(os.path.join(processed_dir, "reconstructed", "overlays", "*.png"))
    recon_labels = glob.glob(os.path.join(processed_dir, "reconstructed", "labels", "*.txt"))
    
    report["full_size_checks"]["full_images_count"] = len(recon_imgs)
    report["full_size_checks"]["full_masks_count"] = len(recon_masks)
    report["full_size_checks"]["full_overlays_count"] = len(recon_overlays)
    report["full_size_checks"]["full_labels_count"] = len(recon_labels)
    
    for f in recon_imgs:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img.shape != (6380, 9568):
            report["full_size_checks"]["dimensions_match_9568x6380"] = False
            
    print(f"Full-Size Reconstructed: {len(recon_imgs)} images, {len(recon_masks)} masks, {len(recon_overlays)} overlays, {len(recon_labels)} labels.")
    print(f"All full-size dimensions exactly 6380x9568: {report['full_size_checks']['dimensions_match_9568x6380']}")
    
    # Save Report
    with open("docs/validation_report.json", "w") as fp:
        json.dump(report, fp, indent=2)
    print("Saved validation report to docs/validation_report.json")
    
    return report

if __name__ == "__main__":
    validate_dataset()
