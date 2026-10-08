"""Scan images to find streak candidates, star fields, noise statistics, and save inspection crops."""

import os
import glob
import numpy as np
import cv2
from astropy.io import fits

def find_candidate_streaks_and_stars(img_path, output_dir):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Scanning {basename}...")
    
    with fits.open(img_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        
    h, w = data.shape
    
    # 1. Background estimation using grid-based median
    grid_size = 128
    gh, gw = h // grid_size, w // grid_size
    # Quick downsample
    down = cv2.resize(data, (gw, gh), interpolation=cv2.INTER_AREA)
    # Estimate global noise MAD
    diff = data - cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    mad_sigma = np.median(np.abs(diff - np.median(diff))) * 1.4826
    print(f"  Estimated local noise sigma: {mad_sigma:.2f}")
    
    # Threshold for finding bright / elongated components
    # Using 4 * sigma above local background
    bg_full = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    sub = np.clip(data - bg_full, 0, None)
    thresh = max(12.0, 3.5 * mad_sigma)
    binary = (sub > thresh).astype(np.uint8)
    
    # Morphological opening to filter 1-pixel hot pixels
    # Find connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print(f"  Found {num_labels - 1} raw connected components above {thresh:.1f}")
    
    # Find components with high elongation / length
    streak_candidates = []
    star_candidates = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        length = np.hypot(bw, bh)
        
        # Check moments
        if area >= 15 and (aspect > 2.5 or length > 25):
            streak_candidates.append((i, area, aspect, length, stats[i], centroids[i]))
        elif 4 <= area <= 150 and aspect < 2.0:
            star_candidates.append((i, area, aspect, stats[i], centroids[i]))
            
    print(f"  Streak candidates: {len(streak_candidates)}, Star candidates: {len(star_candidates)}")
    
    # Save a couple representative crops for visual inspection
    crops_dir = os.path.join(output_dir, basename)
    os.makedirs(crops_dir, exist_ok=True)
    
    # Top streaks by length/area
    streak_candidates.sort(key=lambda x: x[3], reverse=True)
    for idx, (lbl, area, aspect, length, st, cent) in enumerate(streak_candidates[:5]):
        cx, cy = int(cent[0]), int(cent[1])
        r = 128
        y1, y2 = max(0, cy - r), min(h, cy + r)
        x1, x2 = max(0, cx - r), min(w, cx + r)
        crop = data[y1:y2, x1:x2]
        
        # Normalize crop for display using percentile stretch
        p1, p99 = np.percentile(crop, [1, 99.8])
        if p99 > p1:
            disp = np.clip((crop - p1) / (p99 - p1) * 255.0, 0, 255).astype(np.uint8)
        else:
            disp = np.zeros_like(crop, dtype=np.uint8)
        cv2.imwrite(os.path.join(crops_dir, f"streak_{idx+1}_len{int(length)}_asp{aspect:.1f}.png"), disp)
        
    # Representative stars
    if star_candidates:
        cent = star_candidates[len(star_candidates)//2][4]
        cx, cy = int(cent[0]), int(cent[1])
        r = 128
        y1, y2 = max(0, cy - r), min(h, cy + r)
        x1, x2 = max(0, cx - r), min(w, cx + r)
        crop = data[y1:y2, x1:x2]
        p1, p99 = np.percentile(crop, [1, 99.5])
        disp = np.clip((crop - p1) / (max(p99, p1 + 1) - p1) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(crops_dir, "sample_star_field.png"), disp)

def main():
    out_dir = "docs/inspection_crops"
    os.makedirs(out_dir, exist_ok=True)
    fits_files = sorted(glob.glob("Datasets_Assessment/*.fits"))
    for f in fits_files:
        find_candidate_streaks_and_stars(f, out_dir)

if __name__ == "__main__":
    main()
