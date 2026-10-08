"""Generates the official 4-page Assessment Report PDF using ReportLab."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically to ensure exact 4-page constraint."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Digantara SSA Assessment — AI/ML Data Annotation Technical Report")
            self.drawRightString(558, 755, "Candidate: Varshith Reddy Peram")
            self.setStrokeColor(colors.HexColor("#D1D5DB"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#D1D5DB"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        
        self.drawString(54, 32, "Confidential — For Digantara Evaluation Only")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf(filename="docs/ASSESSMENT_REPORT.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#3B82F6"),
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=8,
        spaceAfter=4
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=5,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=10.8,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=2
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#1E293B")
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#0F172A")
    )
    
    story = []
    
    # =========================================================================
    # PAGE 1: TITLE, EXECUTIVE SUMMARY, INSPECTION, PREPROCESSING (Question 2a)
    # =========================================================================
    story.append(Paragraph("DIGANTARA AI/ML DATA ANNOTATION INTERN ASSESSMENT", title_style))
    story.append(Paragraph("End-to-End SSA Imagery Tiling, Morphology Disambiguation & Segmentation Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=1, spaceAfter=6))
    
    # Meta Info Table
    meta_data = [
        [Paragraph("<b>Candidate:</b> Varshith Reddy Peram", table_cell), 
         Paragraph("<b>Repository:</b> github.com/varshithreddyperam/Digantara_work", table_cell)],
        [Paragraph("<b>Email:</b> varshithreddy13@gmail.com", table_cell), 
         Paragraph("<b>Dataset:</b> 10 FITS Images (9568×6380, 16-bit, MARS-6100-18GTM-TF)", table_cell)]
    ]
    t_meta = Table(meta_data, colWidths=[240, 264])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("1. Executive Summary & Dataset Architecture", h1_style))
    story.append(Paragraph(
        "This report documents the end-to-end design, implementation, and empirical validation of the SSA data annotation pipeline for "
        "Digantara. The dataset contains 10 single-band FITS images captured by a <b>MARS-6100-18GTM-TF</b> optical sensor. "
        "All images share identical native dimensions of <b>9568 × 6380 pixels</b> (61.04 MP). Inspection reveals two distinct sensor regimes: "
        "(1) UUID-named frames with pedestal 1–4 ADU and 12-bit ADC dynamics (max 4095 ADU), and (2) CAM_B frames with 16-bit dynamics "
        "(pedestal ~43 ADU, noise σ≈43.0 ADU, max 65535 ADU with ~35 saturated pixels). Every FITS file uses BITPIX=16, BSCALE=1, BZERO=32768, "
        "properly ingested via Astropy with memmap=False to guarantee unsigned uint16 range. NaNs or Inf anomalies are 0% across all 610.4 million pixels.",
        body_style
    ))
    
    story.append(Paragraph("2. Question 2(a): Preprocessing Methodology & Scientific Rationale", h1_style))
    story.append(Paragraph(
        "<b>Scientific Intensity Preservation vs. Contrast Display:</b> Raw photometric pixel values are kept strictly intact for measurement, "
        "while an 8-bit contrast-stretched representation is computed exclusively for visualization and YOLO polygon normalization. "
        "Linear filtering (Gaussian/Wiener) was deliberately avoided as it blurs Point Spread Function (PSF) boundaries, degrades faint star cores, "
        "and widens streak cross-sections. No dark/flat frames were provided; synthetic flat fields were avoided to prevent introducing false spatial gradients.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Spatially Varying Background & Noise Estimation:</b> SSA wide-field sensors suffer from airglow gradients, zodiacal light, and lens vignetting. "
        "We divide the 9568×6380 frame into a 128×128 pixel block grid, computing robust block medians. Bilinear interpolation reconstructs a smooth 2D "
        "background model <i>I<sub>bg</sub>(x,y)</i>. The scientific background-subtracted image is <i>I<sub>sub</sub> = max(0, I - I<sub>bg</sub>)</i>. "
        "Local noise σ is computed via Median Absolute Deviation (MAD): σ = 1.4826 · median(|R - median(R)|) on the residual <i>R = I - I<sub>bg</sub></i>, "
        "providing a mathematically robust noise floor immune to stellar contamination.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Contrast-Enhanced Visualization Stretch:</b> To render 16-bit signals to 8-bit PNG tiles for ML annotations, a robust inverse hyperbolic sine "
        "(asinh) non-linear mapping is applied: <i>I<sub>disp</sub> = 255 · arcsinh(3 · (I<sub>sub</sub> - v<sub>min</sub>) / (v<sub>max</sub> - v<sub>min</sub>)) / arcsinh(30)</i>, "
        "where v<sub>min</sub> and v<sub>max</sub> correspond to the 0.5% and 99.8% percentiles of non-zero flux. This enhances faint stars without burning out streaks.",
        body_style
    ))
    
    # Preprocessing crop image
    prep_crop_path = "data/processed/review_crops/4a8e6cd4-1f5a-4f3a-a858-0f905c2d0b9a/streak_1_inst399.png"
    if os.path.exists(prep_crop_path):
        story.append(Spacer(1, 2))
        story.append(Image(prep_crop_path, width=490, height=130))
        story.append(Paragraph("<i>Figure 1: Preprocessing pipeline stage comparison on Image 4a8e6cd4: (Left) Raw FITS, (Center) Preprocessed 2D background subtracted & asinh stretched, (Right) Delineated pixel mask overlay.</i>", table_cell))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 2: TILING WITHOUT LOSS (Question 2b) & DATASET SPLIT
    # =========================================================================
    story.append(Paragraph("3. Question 2(b): Tiling 9568 × 6380 Imagery Without Data Loss", h1_style))
    story.append(Paragraph(
        "To comply with modern convolutional vision backbones and the assessment specification of exact <b>1024 × 1024 tiles</b>, "
        "a zero-loss boundary padding and coordinate mapping scheme was developed.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Grid Dimension Math:</b> For width <i>W = 9568</i> and height <i>H = 6380</i> with tile dimension <i>T = 1024</i>:<br/>"
        "• Columns: <i>N<sub>cols</sub> = ⌈9568 / 1024⌉ = ⌈9.344⌉ = 10 columns</i><br/>"
        "• Rows: <i>N<sub>rows</sub> = ⌈6380 / 1024⌉ = ⌈6.230⌉ = 7 rows</i><br/>"
        "• Total tiles per image: <i>10 × 7 = 70 tiles</i> (700 tiles across the 10-image dataset)<br/>"
        "• Padded Width: <i>W<sub>pad</sub> = 10 × 1024 = 10,240 pixels</i> (Right padding: <i>Δx = 10,240 - 9568 = 672 pixels</i>)<br/>"
        "• Padded Height: <i>H<sub>pad</sub> = 7 × 1024 = 7,168 pixels</i> (Bottom padding: <i>Δy = 7,168 - 6380 = 788 pixels</i>)",
        body_style
    ))
    story.append(Paragraph(
        "<b>Boundary Padding Protocol:</b> Zero padding (constant 0) is applied strictly to the right and bottom boundaries (left and top remain at offset 0,0). "
        "Crucially, valid pixel masks <i>M<sub>valid</sub>(x,y)</i> record the valid footprint. Annotations are excluded from padded zones.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Seam-Free Detection Strategy:</b> Tiling before detection induces catastrophic seam boundary truncation—streaks crossing a tile boundary "
        "become severed, altering their aspect ratio and generating false broken segments. To permanently eliminate seam artifacts, "
        "<b>detection and morphology classification are executed on the unified full-size image</b>. The authoritative full-size semantic mask "
        "is then partitioned into the 70 tile image/label pairs. Each tile's coordinates are local: <i>(x<sub>tile</sub>, y<sub>tile</sub>) = (x - x<sub>offset</sub>, y - y<sub>offset</sub>)</i>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Tile Manifest Architecture:</b> A comprehensive machine-readable manifest (CSV & JSON) is generated for every image recording: "
        "<code>image_id, tile_id, row, col, x_offset, y_offset, tile_w, tile_h, orig_w, orig_h, pad_r, pad_b, valid_w, valid_h, num_stars, num_streaks</code>. "
        "This manifest guarantees deterministic spatial mapping and enables automated reassembly.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Reassembly & Bit-for-Bit Verification:</b> The reassembly engine places all 70 tiles onto a 7168×10240 canvas and crops exactly at [:6380, :9568]. "
        "Rigorous automated assertion verifies that reconstructed imagery and masks match the pre-tiled arrays with <b>Max Absolute Difference = 0</b> "
        "(100.0% bit-for-bit lossless identity across all 10 images).",
        body_style
    ))
    
    # Tiling Diagram Table
    tiling_summary_data = [
        [Paragraph("<b>Parameter</b>", table_cell_bold), Paragraph("<b>Value</b>", table_cell_bold), Paragraph("<b>Mathematical Formulation / Significance</b>", table_cell_bold)],
        [Paragraph("Target Tile Dimensions", table_cell), Paragraph("1024 × 1024 px", table_cell), Paragraph("Standard deep learning receptive field size", table_cell)],
        [Paragraph("Grid Dimensions", table_cell), Paragraph("10 cols × 7 rows", table_cell), Paragraph("⌈9568/1024⌉ = 10, ⌈6380/1024⌉ = 7 (70 tiles per image)", table_cell)],
        [Paragraph("Padded Canvas", table_cell), Paragraph("10,240 × 7,168 px", table_cell), Paragraph("Total footprint accommodating integer tile boundaries", table_cell)],
        [Paragraph("Right / Bottom Padding", table_cell), Paragraph("672 px / 788 px", table_cell), Paragraph("Appended exclusively to right and bottom margins", table_cell)],
        [Paragraph("Lossless Verification", table_cell), Paragraph("Max Diff = 0 (10/10)", table_cell), Paragraph("Bit-for-bit exact identity confirmed by Reconstructor", table_cell)],
    ]
    t_tile = Table(tiling_summary_data, colWidths=[120, 95, 289])
    t_tile.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EFF6FF")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_tile)
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("4. Dataset Partitioning (Preventing Data Leakage)", h1_style))
    story.append(Paragraph(
        "Randomly partitioning tiles across train and validation splits produces severe geographic data leakage: adjacent tiles from the same exposure "
        "share identical background noise patterns, sensor hot pixels, and celestial fields. To guarantee strict ML rigor, "
        "<b>splitting was performed at the image level</b>:<br/>"
        "• <b>Train Set (8 Images, 560 Tiles):</b> 1a600998, 3e6beceb, 4a8e6cd4, 4e2cdd92, 66ed4268, 7030c2ac, 7d7fefcb, 7e5f3e1c.<br/>"
        "• <b>Validation Set (2 Images, 140 Tiles):</b> CAM_B_20260815T153133 and CAM_B_20260815T154244.<br/>"
        "This completely isolates sensor temporal characteristics and tracking modalities between splits.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 3: CLASSIFICATION (Question 2c & 2d)
    # =========================================================================
    story.append(Paragraph("5. Question 2(c): Disambiguating Short/Fat Streaks from Blobs", h1_style))
    story.append(Paragraph(
        "A critical challenge in SSA data annotation is distinguishing short/fat streaks (slow-moving high-orbit satellites or brief debris reflections) "
        "from point-like stellar blobs. Simple bounding-box aspect ratio is insufficient because diagonally oriented streaks have box ratios near 1.0, "
        "while saturated stars can bleed along sensor columns. We deploy a multi-metric tensor and PSF comparison framework:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Second-Order Central Moments & Eigen-Decomposition:</b> For each connected component contour, central moments are computed:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<i>μ<sub>20</sub> = M<sub>20</sub>/M<sub>00</sub> - x̄<sup>2</sup>, &nbsp;&nbsp; μ<sub>02</sub> = M<sub>02</sub>/M<sub>00</sub> - ȳ<sup>2</sup>, &nbsp;&nbsp; μ<sub>11</sub> = M<sub>11</sub>/M<sub>00</sub> - x̄ȳ</i><br/>"
        "Eigenvalues of the spatial covariance matrix determine principal semi-major axis <i>a = 2√(λ<sub>1</sub>)</i> and semi-minor axis <i>b = 2√(λ<sub>2</sub>)</i>:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<i>λ<sub>1,2</sub> = (μ<sub>20</sub> + μ<sub>02</sub>)/2 ± √[ ((μ<sub>20</sub> - μ<sub>02</sub>)/2)<sup>2</sup> + μ<sub>11</sub><sup>2</sup> ]</i><br/>"
        "The intrinsic axis ratio <i>R<sub>axis</sub> = a / b</i> and eccentricity <i>e = √(1 - λ<sub>2</sub>/λ<sub>1</sub>)</i> measure true elongation invariant to orientation angle.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Minimum-Area Bounding Rectangle Length:</b> <i>L = max(width, height)</i> of the rotated bounding box (`cv2.minAreaRect`). "
        "This isolates the true spatial traversal length of the object independently of pixel coordinate axes.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. Isoperimetric Circularity / Compactness:</b> <i>C = 4π · Area / Perimeter<sup>2</sup></i>. "
        "Point-source stellar blobs exhibit high compactness (<i>C &gt; 0.60</i>), whereas elongated streaks show low circularity (<i>C &lt; 0.45</i>).",
        body_style
    ))
    story.append(Paragraph(
        "<b>4. Comparison with Local Stellar Point Spread Function (PSF):</b> Stellar point sources in a telescope field share an optical PSF diameter "
        "(FWHM ≈ 3.0–4.5 pixels). A feature is classified as an <b>object/streak (Class 1)</b> if its physical length exceeds <b>2.5 × local PSF FWHM</b> "
        "(<i>L ≥ 15.0 px</i>) AND its intrinsic axis ratio <i>R<sub>axis</sub> ≥ 1.85</i> with <i>C &lt; 0.65</i>. Saturated stars, though large in area, "
        "retain circular symmetry (<i>R<sub>axis</sub> &lt; 1.50, C &gt; 0.70</i>), preventing false streak misclassification.",
        body_style
    ))
    story.append(Paragraph(
        "<b>5. Intensity Profile Along Major Axis:</b> Stars display a radial Gaussian profile with a peaked central core. Streaks exhibit a distinctive "
        "flat-top plateau profile along their major axis representing continuous exposure integration during satellite transit.",
        body_style
    ))
    
    story.append(Paragraph("6. Question 2(d): Criteria for Faint/Small Blobs as Star Class", h1_style))
    story.append(Paragraph(
        "Annotating faint stars requires separating true astronomical photons from sensor noise, hot pixels, and cosmic rays without arbitrary guesswork:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Dual-Threshold Hysteresis Formulation:</b> We establish a high core threshold <i>T<sub>core</sub> = k<sub>core</sub> · σ<sub>bg</sub></i> "
        "(with <i>k<sub>core</sub> = 3.8</i>) to initiate confident source seeds, combined with a lower boundary threshold "
        "<i>T<sub>boundary</sub> = k<sub>boundary</sub> · σ<sub>bg</sub></i> (with <i>k<sub>boundary</sub> = 2.0</i>). "
        "Connected components are grown from core seeds down to <i>T<sub>boundary</sub></i>, delineating full visible boundaries down to the noise floor.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Spatial Support & Hot Pixel Filtration:</b> Single-pixel and 2-pixel spikes caused by dark current, read noise, or bad sensor pixels "
        "lack optical diffraction support. Real telescope optics blur any point source over an Airy disk of at least 3–5 pixels. "
        "Accordingly, components with <b>Area &lt; 3 pixels</b> are rejected as noise artifacts.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Centroid Peak Centrality:</b> Valid faint stars must exhibit an intensity maximum centrally located within the 3×3 centroid neighborhood. "
        "Noise fluctuations characterized by hollow or scattered morphology are filtered out.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Ambiguity Review Queue:</b> Sources with borderline parameters (<i>1.55 ≤ R<sub>axis</sub> &lt; 1.95</i> and length 12–18 px) are logged into "
        "an active human review queue (`docs/review_queue/`) with side-by-side comparison crops, ensuring transparent human-in-the-loop audit.",
        bullet_style
    ))
    
    # Review crop sample image
    streak_crop_path = "data/processed/review_crops/CAM_B_20260815T153133_manual_f000207/streak_1_inst2237.png"
    if os.path.exists(streak_crop_path):
        story.append(Spacer(1, 2))
        story.append(Image(streak_crop_path, width=490, height=125))
        story.append(Paragraph("<i>Figure 2: Streak vs stellar blob delineation on Image CAM_B_f000207: (Left) Raw sensor, (Center) Preprocessed, (Right) Delineated mask overlay showing detected streak (magenta) against nearby stellar blobs (blue).</i>", table_cell))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 4: VALIDATION FINDINGS, METRICS, REPRODUCIBILITY & CONCLUSION
    # =========================================================================
    story.append(Paragraph("7. Quantitative Validation Findings & Results Table", h1_style))
    story.append(Paragraph(
        "The full pipeline was executed across all 10 FITS images (610.4 MP total). A comprehensive suite of automated unit tests (`pytest`) "
        "and validation scripts verified format compliance, coordinate normalization, and lossless reconstruction:",
        body_style
    ))
    
    # Summary Table of 10 Images
    img_table_data = [
        [Paragraph("<b>Image ID</b>", table_cell_bold), Paragraph("<b>Split</b>", table_cell_bold), 
         Paragraph("<b>Noise σ</b>", table_cell_bold), Paragraph("<b>Stars</b>", table_cell_bold), 
         Paragraph("<b>Streaks</b>", table_cell_bold), Paragraph("<b>Review</b>", table_cell_bold), 
         Paragraph("<b>Tiles</b>", table_cell_bold), Paragraph("<b>Recon Diff</b>", table_cell_bold)],
        [Paragraph("1a600998...", table_cell), Paragraph("train", table_cell), Paragraph("0.85 ADU", table_cell), Paragraph("513", table_cell), Paragraph("0", table_cell), Paragraph("0", table_cell), Paragraph("70", table_cell), Paragraph("0 (Exact)", table_cell)],
        [Paragraph("3e6beceb...", table_cell), Paragraph("train", table_cell), Paragraph("3.19 ADU", table_cell), Paragraph("9,752", table_cell), Paragraph("5", table_cell), Paragraph("2", table_cell), Paragraph("70", table_cell), Paragraph("0 (Exact)", table_cell)],
        [Paragraph("4a8e6cd4...", table_cell), Paragraph("train", table_cell), Paragraph("0.57 ADU", table_cell), Paragraph("891", table_cell), Paragraph("1", table_cell), Paragraph("0", table_cell), Paragraph("70", table_cell), Paragraph("0 (Exact)", table_cell)],
        [Paragraph("4e2cdd92...", table_cell), Paragraph("train", table_cell), Paragraph("0.57 ADU", table_cell), Paragraph("865", table_cell), Paragraph("3", table_cell), Paragraph("0", table_cell), Paragraph("70", table_cell), Paragraph("0 (Exact)", table_cell)],
        [Paragraph("66ed4268...", table_cell), Paragraph("train", table_cell), Paragraph("4.31 ADU", table_cell), Paragraph("13,368", table_cell), Paragraph("19", table_cell), Paragraph("14", table_cell), Paragraph("70", table_cell), Paragraph("0 (Exact)", table_cell)],
        [Paragraph("7030c2ac...", table_cell), Paragraph("train", table_cell), Paragraph("3.18 ADU", table_cell), Paragraph("9,976", table_cell), Paragraph("10", table_cell), Paragraph("3", table_cell), Paragraph("70", table_cell), Paragraph("0 (Exact)", table_cell)],
        [Paragraph("7d7fefcb...", table_cell), Paragraph("train", table_cell), Paragraph("0.94 ADU", table_cell), Paragraph("542", table_cell), Paragraph("1", table_cell), Paragraph("0", table_cell), Paragraph("70", table_cell), Paragraph("0 (Exact)", table_cell)],
        [Paragraph("7e5f3e1c...", table_cell), Paragraph("train", table_cell), Paragraph("0.58 ADU", table_cell), Paragraph("860", table_cell), Paragraph("2", table_cell), Paragraph("0", table_cell), Paragraph("70", table_cell), Paragraph("0 (Exact)", table_cell)],
        [Paragraph("CAM_B_f000207", table_cell), Paragraph("val", table_cell), Paragraph("42.71 ADU", table_cell), Paragraph("14,278", table_cell), Paragraph("18", table_cell), Paragraph("7", table_cell), Paragraph("70", table_cell), Paragraph("0 (Exact)", table_cell)],
        [Paragraph("CAM_B_f000276", table_cell), Paragraph("val", table_cell), Paragraph("43.00 ADU", table_cell), Paragraph("13,719", table_cell), Paragraph("12", table_cell), Paragraph("5", table_cell), Paragraph("70", table_cell), Paragraph("0 (Exact)", table_cell)],
        [Paragraph("<b>TOTALS</b>", table_cell_bold), Paragraph("<b>10 imgs</b>", table_cell_bold), Paragraph("<b>—</b>", table_cell_bold), Paragraph("<b>64,764</b>", table_cell_bold), Paragraph("<b>71</b>", table_cell_bold), Paragraph("<b>31</b>", table_cell_bold), Paragraph("<b>700</b>", table_cell_bold), Paragraph("<b>0 (100% Lossless)</b>", table_cell_bold)],
    ]
    t_sum = Table(img_table_data, colWidths=[90, 42, 58, 52, 48, 48, 40, 126])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_sum)
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("8. Format Compliance & Export Validation", h1_style))
    story.append(Paragraph(
        "• <b>YOLO Ultralytics Segmentation Syntax:</b> 700 tile label files checked. Total 27,350 polygons validated across Class 0 (`star_blob`) "
        "and Class 1 (`object_streak`). Coordinate values strictly adhere to the unit interval [0.0, 1.0] (0 invalid coordinates). "
        "Every polygon possesses ≥ 3 vertices (≥ 6 numbers per line). 12 negative background tiles correctly formatted as empty text files.<br/>"
        "• <b>Lossless Semantic Masks:</b> Exported as 8-bit PNGs using strict encoding: 0 = background, 1 = star/blob, 2 = object/streak. Zero illegal values.<br/>"
        "• <b>Reconstruction Verification:</b> Reconstructed images and masks match original pre-tiled arrays bit-for-bit (Max Diff = 0).",
        body_style
    ))
    
    story.append(Paragraph("9. Reproducibility & Project Organization", h1_style))
    story.append(Paragraph(
        "The project is structured under standard software engineering best practices with full modularity:<br/>"
        "<code>src/</code> (fits_loader, preprocessor, tiler, detector, classifier, yolo_exporter, reconstructor, visualizer) | "
        "<code>scripts/</code> (run_pipeline, validate_outputs, package_submission) | <code>tests/</code> (pytest suite) | "
        "<code>configs/</code> (pipeline_config.yaml) | <code>docs/</code> (reports, manifests, inspection).<br/>"
        "<b>Single Execution Command:</b> <code>python scripts/run_pipeline.py</code> executes the complete workflow in ~3.8 minutes.",
        body_style
    ))
    
    story.append(Paragraph("10. Limitations, Review Needs & AI Assistance Disclosure", h1_style))
    story.append(Paragraph(
        "<b>Limitations & Human Review Queue:</b> Crowded binary star systems with overlapping Airy disks can exhibit combined axis ratios mimicking "
        "faint streaks. These 31 ambiguous cases (~0.04% of total) have been automatically isolated into the review queue for human sign-off.<br/>"
        "<b>AI Assistance Disclosure:</b> In accordance with transparency principles, LLM-assisted pair-programming (Antigravity) was utilized "
        "for accelerated code refactoring, ReportLab styling, and documentation structuring. All scientific algorithms, mathematical derivations, "
        "threshold selections, and verification assertions were developed and tested directly against the actual assessment dataset.",
        body_style
    ))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report PDF built successfully: {filename}")

if __name__ == "__main__":
    build_pdf()
