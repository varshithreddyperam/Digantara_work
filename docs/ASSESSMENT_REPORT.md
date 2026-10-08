# Digantara AI/ML Data Annotation Intern Assessment Technical Report

**Candidate:** Varshith Reddy Peram  
**Email:** varshithreddy13@gmail.com  
**GitHub Repository:** [https://github.com/varshithreddyperam/Digantara_work](https://github.com/varshithreddyperam/Digantara_work)  
**Date:** October 8, 2026  
**Subject:** End-to-End Space Situational Awareness (SSA) Imagery Preprocessing, Lossless Tiling, Morphology Disambiguation, and Pixel-Level Mask Segmentation

---

## 1. Executive Summary & Dataset Architecture

This technical report details the end-to-end implementation and validation of the Space Situational Awareness (SSA) data annotation pipeline designed for Digantara's real-sky observation dataset. The dataset consists of 10 single-band FITS images captured by an optical telescope sensor (**MARS-6100-18GTM-TF**). 

### Key Dataset Characteristics
- **Native Dimensions:** Identical across all 10 images: **9568 × 6380 pixels** (~61.04 megapixels per image, total 610.4 million pixels).
- **Format & Header Cards:** Standard astronomical FITS format with primary image HDU, `BITPIX = 16`, `BSCALE = 1`, and `BZERO = 32768`. Ingestion via Astropy with `memmap=False` ensures conversion to true unsigned 16-bit integer (`uint16`, range $[0, 65535]$).
- **Data Integrity:** **0 NaNs and 0 Infs** across the entire dataset.
- **Sensor Operating Regimes Observed:**
  1. *UUID-Named Frames (8 images):* Pedestal median of 1–4 ADU, with maximum values capped at 4095 ADU (12-bit ADC mode stored in 16-bit containers) or 1001–1360 ADU. Local noise floor is very low ($\sigma_{MAD} \approx 0.5 - 4.3$ ADU).
  2. *CAM_B Frames (2 images):* Full 16-bit sensor dynamics with a fixed electronic bias pedestal of $\approx 43$ ADU, noise floor $\sigma_{MAD} \approx 43.0$ ADU, and maximum values reaching saturation at 65535 ADU (~33–38 saturated pixels per frame).

---

## 2. Question 2(a): Preprocessing Methodology & Rationale

### A. Separation of Scientific Data and Display Imagery
Standard machine learning pipelines frequently make the mistake of irreversibly modifying scientific imagery through destructive smoothing or histogram equalization. In our architecture:
- **Scientific Intensity Data:** Background-subtracted 2D arrays ($I_{sub} = \max(0, I - I_{bg})$) retain linear radiometric fidelity for flux measurement, centroiding, and morphology analysis.
- **Display Representation:** An 8-bit unsigned integer (`uint8`, $[0, 255]$) image is derived purely for computer vision visualization and normalized YOLO polygon contour extraction.

### B. Spatially Varying Background & Noise Estimation
SSA wide-field sensors suffer from large-scale background variations caused by atmospheric airglow, twilight glow, lens vignetting, and sensor thermal gradients.
- We decompose each $9568 \times 6380$ frame into a $128 \times 128$ block grid, computing robust block medians immune to stars and streaks.
- Bilinear interpolation reconstructs a smooth 2D continuous background surface $I_{bg}(x, y)$.
- Background subtraction yields $I_{sub} = \max(0, I(x, y) - I_{bg}(x, y))$.
- Local noise $\sigma$ is estimated from the residual $R = I - I_{bg}$ using robust Median Absolute Deviation (MAD):
  $$\sigma_{MAD} = 1.4826 \times \text{median}(|R - \text{median}(R)|)$$
  This provides a mathematically principled noise floor completely unaffected by stellar sources or streaks.

### C. Filtering Policy
Aggressive spatial filtering (such as Gaussian, median, or bilateral filtering) was deliberately rejected. Such filtering blurs Point Spread Function (PSF) boundaries, erodes faint stars into the noise floor, and expands streak cross-sections. No dark-frame or flat-field calibration frames were provided in the dataset; therefore, synthetic calibrations were omitted to avoid injecting false spatial artifacts.

### D. Non-Linear Contrast Enhancement for Display
To compress the high-dynamic-range signals into an 8-bit image without washing out faint stars or saturating bright streaks, an inverse hyperbolic sine ($\text{asinh}$) stretch was applied:
$$I_{disp}(x, y) = 255 \times \frac{\text{arcsinh}\left(3.0 \times \frac{I_{sub} - v_{min}}{v_{max} - v_{min}}\right)}{\text{arcsinh}(30.0)}$$
where $v_{min}$ and $v_{max}$ are the 0.5% and 99.8% percentiles of positive pixel flux.

---

## 3. Question 2(b): Tiling Without Data Loss (Exact 1024 × 1024)

### A. Mathematical Derivation of Tile Grid
Given native image dimensions $W = 9568$ and $H = 6380$ with target tile dimensions $T = 1024$:
- **Horizontal Columns:** $N_{cols} = \lceil 9568 / 1024 \rceil = 10 \text{ columns}$
- **Vertical Rows:** $N_{rows} = \lceil 6380 / 1024 \rceil = 7 \text{ rows}$
- **Total Tiles per Image:** $10 \times 7 = 70 \text{ tiles}$ (700 tiles across the 10 images)
- **Padded Dimensions:**
  $$W_{pad} = 10 \times 1024 = 10,240 \text{ pixels}$$
  $$H_{pad} = 7 \times 1024 = 7,168 \text{ pixels}$$
- **Padding Quantities:**
  - **Right Padding:** $\Delta x = 10,240 - 9568 = 672 \text{ pixels}$
  - **Bottom Padding:** $\Delta y = 7,168 - 6380 = 788 \text{ pixels}$

### B. Boundary Handling & Zero-Padding Protocol
- Padding is applied **strictly to the right and bottom boundaries**. The top-left corner remains origin $(0, 0)$.
- Valid pixel binary masks $M_{valid}(x, y)$ explicitly demarcate the true sensor domain ($[0:6380, 0:9568]$) from the zero-padded regions ($x \ge 9568$ or $y \ge 6380$).
- Annotations and source detections are strictly masked to valid pixels; padding is excluded.

### C. Seam-Free Detection Architecture
Tiling images prior to detection creates severe artificial edge truncation: streaks or stars straddling tile borders get clipped, corrupting their aspect ratio and generating broken instances.
To ensure 100% seam integrity:
1. Detection and morphology classification are performed on the **unified full-size image**.
2. The resulting authoritative semantic mask is tiled into the 70 patches.
3. Coordinates for local tile labels are transformed via:
   $$x_{tile} = x_{full} - x_{offset}, \quad y_{tile} = y_{full} - y_{offset}$$
   and normalized by $1024.0$.

### D. Manifest Specification & Lossless Reassembly Verification
Every tile is cataloged in a structured manifest (JSON and CSV) recording image ID, tile ID, grid row/col, $x/y$ offsets, valid dimensions, and padding amounts.
Upon reassembling all 70 tiles onto a $7168 \times 10240$ canvas and cropping to original dimensions $[:6380, :9568]$, automated verification proved:
$$\max |I_{original} - I_{reconstructed}| = 0$$
$$\max |M_{original} - M_{reconstructed}| = 0$$
**Every image and mask achieves 100.0% bit-for-bit lossless reconstruction.**

---

## 4. Question 2(c): Distinguishing Short/Fat Streaks from Blobs

Short/fat streaks (caused by slow tumbling debris or high orbital regimes like GEO) can have apparent widths similar to bright stellar cores. Distinguishing them requires multi-parameter shape tensor and PSF comparison:

1. **Second-Order Central Moments & Covariance Eigenvalues:**
   For contour $C$, central moments are computed:
   $$\mu_{20} = \frac{M_{20}}{M_{00}} - \bar{x}^2, \quad \mu_{02} = \frac{M_{02}}{M_{00}} - \bar{y}^2, \quad \mu_{11} = \frac{M_{11}}{M_{00}} - \bar{x}\bar{y}$$
   The eigenvalues of the moment covariance matrix are:
   $$\lambda_{1,2} = \frac{\mu_{20} + \mu_{02}}{2} \pm \sqrt{\left(\frac{\mu_{20} - \mu_{02}}{2}\right)^2 + \mu_{11}^2}$$
   Principal axes: semi-major $a = 2\sqrt{\lambda_1}$, semi-minor $b = 2\sqrt{\lambda_2}$.
   - **Intrinsic Axis Ratio:** $R_{axis} = a / b$ (invariant to orientation angle).
   - **Eccentricity:** $e = \sqrt{1 - \lambda_2/\lambda_1}$.

2. **Minimum-Area Bounding Box Length:**
   The spatial traversal length $L = \max(w_{rect}, h_{rect})$ from rotated bounding box analysis (`cv2.minAreaRect`).

3. **Isoperimetric Compactness / Circularity:**
   $$C = \frac{4\pi \times \text{Area}}{\text{Perimeter}^2}$$
   Stellar blobs exhibit high circularity ($C > 0.60$), whereas elongated streaks show low circularity ($C < 0.45$).

4. **Comparison Against Local Stellar PSF:**
   Stellar point sources in the local telescope field share a characteristic optical PSF width (FWHM $\approx 3.0 - 4.5$ pixels).
   - A candidate is classified as **Object/Streak (Class 1)** if:
     $$L \ge \max(14.0\text{ px}, 2.5 \times \text{FWHM}_{local}) \quad \text{AND} \quad R_{axis} \ge 1.85 \quad \text{AND} \quad C < 0.65$$
   - Saturated stars, despite large pixel areas, maintain circular symmetry ($R_{axis} < 1.50, C > 0.70$) and are classified as **Star/Blob (Class 0)**.

5. **Intensity Profile Along Major Axis:**
   A stellar blob exhibits a radial Gaussian profile peaking sharply at the centroid. A streak exhibits an extended plateau profile representing uniform transit exposure.

---

## 5. Question 2(d): Criteria for Faint/Small Blobs as Star Class

Faint stars near the detection threshold must be reliably separated from sensor dark spikes, shot noise, and hot pixels:

1. **Dual-Threshold Hysteresis Formulation:**
   - **Core Seed Threshold:** $T_{core} = k_{core} \times \sigma_{bg}$ ($k_{core} = 3.8$).
   - **Boundary Delineation Threshold:** $T_{boundary} = k_{boundary} \times \sigma_{bg}$ ($k_{boundary} = 2.0$).
   Connected components are initialized at core seeds and expanded to $T_{boundary}$, ensuring full isophotal delineations down to the visible boundary.

2. **Spatial Support & Noise/Hot Pixel Rejection:**
   Isolated single-pixel or 2-pixel spikes caused by cosmic rays, hot pixels, or thermal noise lack optical diffraction support. Optical physics dictates that any real celestial source is convolved with the telescope's optical transfer function, yielding a minimum spatial footprint:
   $$\text{Area} \ge 3 \text{ pixels}$$
   Components with $\text{Area} < 3\text{ px}$ are rejected as noise artifacts.

3. **Centroid Centrality & Peak Profile:**
   Faint stars must possess a well-defined intensity peak centrally located within their local 3×3 centroid window.

4. **Review Queue for Ambiguous Detections:**
   Detections with borderline metrics ($1.55 \le R_{axis} < 1.95$ and length 12–18 pixels, often caused by close overlapping binary stars) are assigned provisional classes and flagged for manual audit (`needs_review = True`), generating inspection crops in `docs/review_queue/`.

---

## 6. Quantitative Dataset Summary & Validation Findings

### Pipeline Execution Summary Across All 10 Images

| Image ID | Assigned Split | Noise $\sigma_{bg}$ | Star / Blob Count | Object / Streak Count | Review Flags | Tiles Generated | Reconstruction Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1a600998...` | Train | 0.85 ADU | 513 | 0 | 0 | 70 | 0 (Bit-for-bit exact) |
| `3e6beceb...` | Train | 3.19 ADU | 9,752 | 5 | 2 | 70 | 0 (Bit-for-bit exact) |
| `4a8e6cd4...` | Train | 0.57 ADU | 891 | 1 | 0 | 70 | 0 (Bit-for-bit exact) |
| `4e2cdd92...` | Train | 0.57 ADU | 865 | 3 | 0 | 70 | 0 (Bit-for-bit exact) |
| `66ed4268...` | Train | 4.31 ADU | 13,368 | 19 | 14 | 70 | 0 (Bit-for-bit exact) |
| `7030c2ac...` | Train | 3.18 ADU | 9,976 | 10 | 3 | 70 | 0 (Bit-for-bit exact) |
| `7d7fefcb...` | Train | 0.94 ADU | 542 | 1 | 0 | 70 | 0 (Bit-for-bit exact) |
| `7e5f3e1c...` | Train | 0.58 ADU | 860 | 2 | 0 | 70 | 0 (Bit-for-bit exact) |
| `CAM_B_f000207` | Val | 42.71 ADU | 14,278 | 18 | 7 | 70 | 0 (Bit-for-bit exact) |
| `CAM_B_f000276` | Val | 43.00 ADU | 13,719 | 12 | 5 | 70 | 0 (Bit-for-bit exact) |
| **TOTALS** | **10 Images** | **—** | **64,764** | **71** | **31** | **700** | **0 (100.0% Lossless)** |

### Strict Validation Checks
- **YOLO Ultralytics Segmentation:** 700 tile label files checked. Total 27,350 polygon instances verified across Class 0 (`star_blob`) and Class 1 (`object_streak`). All coordinates strictly bounded in $[0.0, 1.0]$. Zero invalid coordinates. Zero degenerate polygons. 12 negative background tiles formatted as valid empty `.txt` files.
- **Semantic Mask Format:** Single-channel 8-bit PNGs strictly encoded as 0 (background), 1 (star/blob), 2 (object/streak).
- **Leakage-Free Splitting:** Train (8 images, 560 tiles) and Validation (2 images, 140 tiles) split strictly by full exposure to prevent spatial data leakage.

---

## 7. AI Assistance Disclosure & Reproducibility

- **AI Assistance Disclosure:** Antigravity (LLM coding assistant) was used to accelerate script boilerplate, ReportLab styling, and documentation structuring. All astronomical detection algorithms, mathematical derivations, parameter calibrations, and validation assertions were implemented, run, and empirically verified against the actual assessment dataset.
- **Reproducibility:** A single documented command executes the complete workflow:
  ```bash
  python scripts/run_pipeline.py
  ```
  Repository: [https://github.com/varshithreddyperam/Digantara_work](https://github.com/varshithreddyperam/Digantara_work)
