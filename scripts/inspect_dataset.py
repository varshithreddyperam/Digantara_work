"""Dataset inspection script for Digantara SSA FITS images."""

import os
import glob
import json
import numpy as np
from astropy.io import fits

def inspect_fits_file(filepath):
    print(f"=== Inspecting {os.path.basename(filepath)} ===")
    info = {"filename": os.path.basename(filepath), "size_bytes": os.path.getsize(filepath)}
    
    with fits.open(filepath) as hdul:
        info["num_hdus"] = len(hdul)
        hdu_details = []
        for i, hdu in enumerate(hdul):
            detail = {
                "index": i,
                "name": hdu.name,
                "is_image": hdu.is_image,
                "header_cards": len(hdu.header),
            }
            if hdu.data is not None:
                detail["shape"] = list(hdu.data.shape)
                detail["dtype"] = str(hdu.data.dtype)
            else:
                detail["shape"] = None
                detail["dtype"] = None
            hdu_details.append(detail)
        info["hdus"] = hdu_details
        
        # Primary HDU / Image HDU
        primary = hdul[0]
        hdr = primary.header
        important_keys = [
            "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", 
            "BSCALE", "BZERO", "EXPTIME", "GAIN", "DATE-OBS", 
            "INSTRUME", "OBJECT", "FILTER", "TELESCOP", "OBSERVER"
        ]
        info["header_sample"] = {k: hdr[k] for k in important_keys if k in hdr}
        
        data = primary.data
        if data is None and len(hdul) > 1:
            data = hdul[1].data
            hdr = hdul[1].header
            
        if data is not None:
            # Check dimensions: FITS NAXIS1 is width (cols), NAXIS2 is height (rows).
            # In numpy, data.shape is (height, width) = (NAXIS2, NAXIS1)
            height, width = data.shape
            info["height"] = height
            info["width"] = width
            
            # Check invalid pixels
            nan_count = int(np.isnan(data).sum()) if np.issubdtype(data.dtype, np.floating) else 0
            inf_count = int(np.isinf(data).sum()) if np.issubdtype(data.dtype, np.floating) else 0
            zero_count = int((data == 0).sum())
            max_val = float(np.max(data))
            min_val = float(np.min(data))
            
            # Sample for robust statistics
            sample = data[::4, ::4].flatten()
            q001, q01, q05, q25, q50, q75, q95, q99, q999 = np.percentile(
                sample, [0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9]
            )
            # Estimate background noise via Median Absolute Deviation (MAD)
            med = float(q50)
            mad = float(np.median(np.abs(sample - med)))
            sigma_mad = mad * 1.4826
            
            # Check saturation
            saturated_65535 = int((data >= 65535).sum())
            saturated_65500 = int((data >= 65500).sum())
            
            stats = {
                "dtype": str(data.dtype),
                "min": min_val,
                "max": max_val,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "zero_count": zero_count,
                "percentiles": {
                    "p0_1": float(q001),
                    "p1": float(q01),
                    "p5": float(q05),
                    "p25": float(q25),
                    "median": float(q50),
                    "p75": float(q75),
                    "p95": float(q95),
                    "p99": float(q99),
                    "p99_9": float(q999),
                },
                "estimated_bg_median": med,
                "estimated_bg_sigma": sigma_mad,
                "saturated_65535_count": saturated_65535,
                "saturated_65500_count": saturated_65500,
            }
            info["stats"] = stats
            print(f"  Shape: {height}x{width}, Dtype: {data.dtype}")
            print(f"  Min: {min_val}, Max: {max_val}, Median: {med:.1f}, MAD Sigma: {sigma_mad:.2f}")
            print(f"  P99: {q99:.1f}, P99.9: {q999:.1f}, Saturated (>65500): {saturated_65500}")
            
    return info

def main():
    fits_files = sorted(glob.glob("Datasets_Assessment/*.fits"))
    print(f"Found {len(fits_files)} FITS files.")
    all_info = []
    for f in fits_files:
        info = inspect_fits_file(f)
        all_info.append(info)
        
    os.makedirs("docs", exist_ok=True)
    with open("docs/dataset_inspection.json", "w") as fp:
        json.dump(all_info, fp, indent=2)
    print("Saved inspection report to docs/dataset_inspection.json")

if __name__ == "__main__":
    main()
