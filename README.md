# Digantara SSA AI/ML Data Annotation Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://pytest.org)

An end-to-end, reproducible astronomical pipeline for Space Situational Awareness (SSA) imagery preprocessing, exact 1024×1024 lossless boundary-padded tiling, multi-scale morphology disambiguation (stars/blobs vs space objects/streaks), and YOLO Ultralytics segmentation export.

---

## 📌 Project Architecture

```
Digantara_work/
├── configs/
│   └── pipeline_config.yaml         # Central pipeline configuration parameters
├── src/
│   ├── __init__.py
│   ├── fits_loader.py               # FITS HDU reader, metadata parser, BSCALE/BZERO handling
│   ├── preprocessor.py              # Spatially varying 2D background (MAD noise), asinh contrast stretch
│   ├── tiler.py                     # Exact 1024x1024 tiling, right/bottom padding, manifest generator
│   ├── detector.py                  # Multi-scale hysteresis detection, connected components
│   ├── classifier.py                # Moments, covariance eigenvalues, elongation, circularity, local PSF
│   ├── yolo_exporter.py             # Polygon extraction, normalization, YOLO segmentation format
│   ├── reconstructor.py             # Lossless tile reassembly engine & bit-for-bit assertion
│   └── visualizer.py                # Visual overlays (blue blobs, pink streaks) and comparison crops
├── scripts/
│   ├── inspect_dataset.py           # Analyzes headers, statistics, and sensor regimes across FITS files
│   ├── scan_candidates.py           # Rapid streak candidate scanning and preview crop extraction
│   ├── run_pipeline.py              # Single command executing the full end-to-end pipeline
│   ├── validate_outputs.py          # Rigorous verification of YOLO syntax, masks, and dimensions
│   ├── generate_report_pdf.py       # Renders the official 4-page Assessment Report PDF via ReportLab
│   ├── package_submission.py        # Compiles the full 1.34 GB standalone submission ZIP
│   └── package_lightweight.py       # Compiles a ~29 MB upload-ready core deliverables ZIP
├── tests/
│   └── test_pipeline.py             # Pytest suite verifying tiling math, lossless reconstruction, etc.
├── requirements.txt                 # Tested Python dependencies
└── .gitignore                       # Excludes raw data, reports, large outputs, caches
```

---

## 🚀 Quickstart & Pipeline Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/varshithreddyperam/Digantara_work.git
cd Digantara_work

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python -m pytest tests/test_pipeline.py
```

### 3. Run Dataset Inspection
```bash
python scripts/inspect_dataset.py
```

### 4. Execute Complete End-to-End Pipeline
Run the entire workflow (preprocessing, full-resolution detection, 1024×1024 tiling, YOLO export, and tile reassembly) with a single command:
```bash
python scripts/run_pipeline.py
```

### 5. Validate All Outputs
Verify polygon bounds, label syntax, mask values, and lossless reconstruction:
```bash
python scripts/validate_outputs.py
```

### 6. Build Assessment Report PDF & Submission Archives
```bash
python scripts/generate_report_pdf.py
python scripts/package_submission.py
python scripts/package_lightweight.py
```

---

## 🔬 Core Scientific Methodology

### 1. Preprocessing (Question 2a)
- **Scientific vs. Display Separation:** Linear raw fluxes are preserved for photometry and morphology analysis. Display imagery uses an 8-bit inverse hyperbolic sine ($\text{asinh}$) non-linear stretch to illuminate faint stars while preventing streak blowout.
- **2D Mesh Background Subtraction:** SSA sensor frames exhibit significant vignetting and airglow gradients. A $128 \times 128$ pixel grid computes block medians, upsampled bilinearly to form a continuous surface $I_{bg}$.
- **Noise Floor Estimation:** Local noise $\sigma$ is computed from residual $R = I - I_{bg}$ using Median Absolute Deviation (MAD):
  $$\sigma_{MAD} = 1.4826 \times \text{median}(|R - \text{median}(R)|)$$

### 2. Lossless Tiling & Reassembly (Question 2b)
- **Grid Math:** For native dimensions $9568 \times 6380$ and tile dimension $1024$:
  - Columns: $\lceil 9568 / 1024 \rceil = 10$
  - Rows: $\lceil 6380 / 1024 \rceil = 7$
  - Total tiles per image: $10 \times 7 = 70 \text{ tiles}$ (700 tiles across 10 images)
  - Padded footprint: $10,240 \times 7,168$ px (Right pad: $672$ px, Bottom pad: $788$ px)
- **Zero-Loss Padding:** Appended exclusively to right and bottom boundaries.
- **Seam-Free Detection:** Detection is performed on the unified full-size image, eliminating tile boundary clipping artifacts.
- **Lossless Verification:** Automated assertion confirms:
  $$\max |I_{original} - I_{reconstructed}| = 0 \quad (100.0\% \text{ bit-for-bit lossless identity})$$

### 3. Disambiguating Short/Fat Streaks from Blobs (Question 2c)
- **Second-Order Central Moments & Covariance Eigenvalues:**
  $$\lambda_{1,2} = \frac{\mu_{20} + \mu_{02}}{2} \pm \sqrt{\left(\frac{\mu_{20} - \mu_{02}}{2}\right)^2 + \mu_{11}^2}$$
  Yields orientation-invariant intrinsic axis ratio $R_{axis} = a / b$ and eccentricity $e = \sqrt{1 - \lambda_2/\lambda_1}$.
- **Local Stellar PSF Comparison:** Local optical stars have FWHM $\approx 3.0 - 4.5$ px. A candidate is classified as **Object/Streak (Class 1)** if rotated bounding box length $L \ge 2.5 \times \text{FWHM}_{local}$ ($L \ge 15\text{ px}$), $R_{axis} \ge 1.85$, and circularity $C < 0.65$. Saturated stars retain circular symmetry ($R_{axis} < 1.50$) and are classified as **Star/Blob (Class 0)**.

### 4. Faint/Small Blobs as Star Class (Question 2d)
- **Hysteresis Thresholding:** High core seed ($T_{core} = 3.8\sigma$) and lower boundary threshold ($T_{boundary} = 2.0\sigma$) trace full isophotal contours.
- **Noise & Hot Pixel Rejection:** Single-pixel spikes lack optical diffraction support. Real telescope PSFs span an Airy disk of $\ge 3\text{ px}$. Components with $\text{Area} < 3\text{ px}$ are rejected.
- **Review Queue:** Borderline detections are isolated in `docs/review_queue/` for human verification.

---

## 📊 Summary Results Table

| Image ID | Split | Noise $\sigma_{bg}$ | Star Count | Streak Count | Review Flags | Tiles | Reconstruction Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1a600998...` | Train | 0.85 ADU | 513 | 0 | 0 | 70 | 0 (Exact) |
| `3e6beceb...` | Train | 3.19 ADU | 9,752 | 5 | 2 | 70 | 0 (Exact) |
| `4a8e6cd4...` | Train | 0.57 ADU | 891 | 1 | 0 | 70 | 0 (Exact) |
| `4e2cdd92...` | Train | 0.57 ADU | 865 | 3 | 0 | 70 | 0 (Exact) |
| `66ed4268...` | Train | 4.31 ADU | 13,368 | 19 | 14 | 70 | 0 (Exact) |
| `7030c2ac...` | Train | 3.18 ADU | 9,976 | 10 | 3 | 70 | 0 (Exact) |
| `7d7fefcb...` | Train | 0.94 ADU | 542 | 1 | 0 | 70 | 0 (Exact) |
| `7e5f3e1c...` | Train | 0.58 ADU | 860 | 2 | 0 | 70 | 0 (Exact) |
| `CAM_B_f000207` | Val | 42.71 ADU | 14,278 | 18 | 7 | 70 | 0 (Exact) |
| `CAM_B_f000276` | Val | 43.00 ADU | 13,719 | 12 | 5 | 70 | 0 (Exact) |
| **TOTALS** | **10 Images** | **—** | **64,764** | **71** | **31** | **700** | **0 (100.0% Lossless)** |

---

## 📄 License & Attribution
- **Candidate:** Varshith Reddy Peram (`varshithreddy13@gmail.com`)
- **Institution:** Digantara Research and Technologies Pvt. Ltd.
- **Repository:** [https://github.com/varshithreddyperam/Digantara_work](https://github.com/varshithreddyperam/Digantara_work)
