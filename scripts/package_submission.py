"""Packages assessment deliverables into a standalone submission ZIP archive."""

import os
import zipfile
import json
import time

def create_submission_readme(dest_path="SUBMISSION_README.md"):
    content = """# Digantara SSA Annotation Intern Assessment Submission

**Candidate:** Varshith Reddy Peram  
**Email:** varshithreddy13@gmail.com  
**GitHub Repository:** https://github.com/varshithreddyperam/Digantara_work  
**Date:** October 8, 2026  

---

## 📁 Submission Contents Structure

```
Digantara_Assessment_Submission/
├── ASSESSMENT_REPORT.pdf            # Official 4-page technical assessment report
├── ASSESSMENT_REPORT.md             # Complete report in Markdown format
├── SUBMISSION_README.md             # This guide and inventory
├── dataset.yaml                     # Ultralytics YOLO segmentation dataset config
├── dataset/                         # Exact 1024x1024 YOLO segmentation dataset
│   ├── images/
│   │   ├── train/                   # 560 training tiles (8 images)
│   │   └── val/                     # 140 validation tiles (2 CAM_B images)
│   └── labels/
│       ├── train/                   # 560 YOLO polygon segmentation label files (.txt)
│       └── val/                     # 140 YOLO polygon segmentation label files (.txt)
├── manifests/                       # Tile manifests with coordinate mapping & padding info
│   ├── *_manifest.csv               # Machine-readable tabular manifests (70 tiles each)
│   └── *_manifest.json              # Detailed metadata manifests
├── reconstructed/                   # Full-size repatched imagery (9568x6380)
│   ├── images/                      # Reconstructed 8-bit display imagery (bit-for-bit lossless)
│   ├── masks/                       # Reconstructed lossless semantic masks (0=bg, 1=star, 2=streak)
│   ├── overlays/                    # Reconstructed visual inspection overlays
│   └── labels/                      # Full-size YOLO normalized segmentation labels
├── review_crops/                    # Inspection crops (Raw vs Preprocessed vs Mask)
│   └── <image_id>/                  # Streak candidates and flagged review instances
└── validation/                      # Quantitative reports and test logs
    ├── validation_report.json       # Formal dataset syntax & boundary verification
    ├── pipeline_summary.json        # Detection counts and execution timing
    └── dataset_inspection.json      # Header metadata and statistical analysis
```

---

## 🎯 Questions Answered in Report (Summary)

1. **Preprocessing (Question 2a):**
   - 2D mesh grid background estimation ($128 \\times 128$ blocks) with bilinear interpolation.
   - Robust noise floor estimation via Median Absolute Deviation (MAD): $\\sigma = 1.4826 \\times \\text{median}(|R - \\text{median}(R)|)$.
   - Strict separation of scientific intensity data ($I_{sub}$) from contrast-enhanced display images (non-linear $\\text{asinh}$ stretch).
   - Zero destructive filtering to protect faint point sources and streak boundaries.

2. **Lossless Tiling (Question 2b):**
   - Exact mathematical derivation: $10 \\text{ columns} \\times 7 \\text{ rows} = 70 \\text{ tiles}$ of $1024 \\times 1024$ px per image.
   - Padded footprint: $10,240 \\times 7,168$ px (Right pad: 672 px, Bottom pad: 788 px).
   - Detection performed on unified full-size image to eliminate tile seam splitting.
   - Automated reassembly verified: **Max Absolute Difference = 0** across all 10 images.

3. **Distinguishing Short/Fat Streaks from Blobs (Question 2c):**
   - Second-order central moments and covariance matrix eigenvalues ($\lambda_1, \lambda_2$).
   - Intrinsic axis ratio $R_{axis} = a / b$ and eccentricity $e = \\sqrt{1 - \\lambda_2/\\lambda_1}$.
   - Rotated bounding box length $L$ compared against local stellar PSF FWHM ($L \\ge 2.5 \\times \\text{FWHM}$).
   - Isoperimetric compactness $C = 4\\pi A / P^2$ and linear plateau intensity profiles.

4. **Criteria for Faint/Small Blobs as Star Class (Question 2d):**
   - Dual-threshold hysteresis ($T_{core} = 3.8\\sigma, T_{boundary} = 2.0\\sigma$).
   - Minimum spatial support $\\text{Area} \\ge 3 \\text{ px}$ to filter out 1-2 px cosmic rays and hot pixels.
   - Centroid peak centrality in $3 \\times 3$ neighborhood.
   - Ambiguous boundary cases routed to human review queue.

---

## 🚀 Reproduction Quickstart

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run unit tests:
   ```bash
   pytest tests/test_pipeline.py
   ```
3. Run the end-to-end pipeline:
   ```bash
   python scripts/run_pipeline.py
   ```
4. Validate outputs:
   ```bash
   python scripts/validate_outputs.py
   ```
"""
    with open(dest_path, "w", encoding="utf-8") as fp:
        fp.write(content.strip() + "\n")
    print(f"Created {dest_path}")

def package_submission(zip_path="submission/Digantara_Assessment_Submission.zip"):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    create_submission_readme("docs/SUBMISSION_README.md")
    
    print(f"Building submission zip: {zip_path}...")
    start_t = time.time()
    
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # Add Reports and README
        z.write("docs/ASSESSMENT_REPORT.pdf", "ASSESSMENT_REPORT.pdf")
        z.write("docs/ASSESSMENT_REPORT.md", "ASSESSMENT_REPORT.md")
        z.write("docs/SUBMISSION_README.md", "SUBMISSION_README.md")
        
        # Add dataset.yaml
        z.write("data/processed/dataset/dataset.yaml", "dataset.yaml")
        
        # Add validation documents
        z.write("docs/validation_report.json", "validation/validation_report.json")
        z.write("docs/pipeline_summary.json", "validation/pipeline_summary.json")
        z.write("docs/dataset_inspection.json", "validation/dataset_inspection.json")
        
        # Add manifests
        for mf in sorted(os.listdir("data/processed/manifests")):
            z.write(os.path.join("data/processed/manifests", mf), f"manifests/{mf}")
            
        # Add review crops
        for root, _, files in os.walk("data/processed/review_crops"):
            for f in files:
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, "data/processed")
                z.write(full_p, rel_p)
                
        # Add YOLO tile labels
        for split in ["train", "val"]:
            lbl_dir = f"data/processed/dataset/labels/{split}"
            for f in os.listdir(lbl_dir):
                z.write(os.path.join(lbl_dir, f), f"dataset/labels/{split}/{f}")
                
        # Add YOLO tile images
        for split in ["train", "val"]:
            img_dir = f"data/processed/dataset/images/{split}"
            for f in os.listdir(img_dir):
                z.write(os.path.join(img_dir, f), f"dataset/images/{split}/{f}")
                
        # Add reconstructed full-size images, masks, overlays, and labels
        for cat in ["images", "masks", "overlays", "labels"]:
            cat_dir = f"data/processed/reconstructed/{cat}"
            for f in os.listdir(cat_dir):
                z.write(os.path.join(cat_dir, f), f"reconstructed/{cat}/{f}")
                
    sz_bytes = os.path.getsize(zip_path)
    sz_mb = sz_bytes / (1024 * 1024)
    elapsed = time.time() - start_t
    print(f"Submission zip packaged successfully in {elapsed:.1f}s!")
    print(f"Archive file: {zip_path}")
    print(f"Archive size: {sz_mb:.2f} MB ({sz_bytes} bytes)")

if __name__ == "__main__":
    package_submission()
