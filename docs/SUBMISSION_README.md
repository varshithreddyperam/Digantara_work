# Digantara SSA Annotation Intern Assessment Submission

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
   - 2D mesh grid background estimation ($128 \times 128$ blocks) with bilinear interpolation.
   - Robust noise floor estimation via Median Absolute Deviation (MAD): $\sigma = 1.4826 \times \text{median}(|R - \text{median}(R)|)$.
   - Strict separation of scientific intensity data ($I_{sub}$) from contrast-enhanced display images (non-linear $\text{asinh}$ stretch).
   - Zero destructive filtering to protect faint point sources and streak boundaries.

2. **Lossless Tiling (Question 2b):**
   - Exact mathematical derivation: $10 \text{ columns} \times 7 \text{ rows} = 70 \text{ tiles}$ of $1024 \times 1024$ px per image.
   - Padded footprint: $10,240 \times 7,168$ px (Right pad: 672 px, Bottom pad: 788 px).
   - Detection performed on unified full-size image to eliminate tile seam splitting.
   - Automated reassembly verified: **Max Absolute Difference = 0** across all 10 images.

3. **Distinguishing Short/Fat Streaks from Blobs (Question 2c):**
   - Second-order central moments and covariance matrix eigenvalues ($\lambda_1, \lambda_2$).
   - Intrinsic axis ratio $R_{axis} = a / b$ and eccentricity $e = \sqrt{1 - \lambda_2/\lambda_1}$.
   - Rotated bounding box length $L$ compared against local stellar PSF FWHM ($L \ge 2.5 \times \text{FWHM}$).
   - Isoperimetric compactness $C = 4\pi A / P^2$ and linear plateau intensity profiles.

4. **Criteria for Faint/Small Blobs as Star Class (Question 2d):**
   - Dual-threshold hysteresis ($T_{core} = 3.8\sigma, T_{boundary} = 2.0\sigma$).
   - Minimum spatial support $\text{Area} \ge 3 \text{ px}$ to filter out 1-2 px cosmic rays and hot pixels.
   - Centroid peak centrality in $3 \times 3$ neighborhood.
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
