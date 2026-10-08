"""FITS image loader and header metadata parser for SSA imagery."""

import os
from typing import Dict, Any, Tuple
import numpy as np
from astropy.io import fits

class FITSLoader:
    """Robust FITS loader handling 16-bit SSA sensor imagery."""

    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a FITS file, extracting image data and key header information.
        
        Args:
            filepath: Path to the FITS file.
            
        Returns:
            Tuple of (image_array as uint16, metadata dictionary).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"FITS file not found: {filepath}")
            
        with fits.open(filepath, memmap=False) as hdul:
            primary = hdul[0]
            data = primary.data
            hdr = primary.header
            
            if data is None and len(hdul) > 1:
                data = hdul[1].data
                hdr = hdul[1].header
                
            if data is None:
                raise ValueError(f"No image data found in FITS file: {filepath}")
                
            # Astropy automatically applies BSCALE and BZERO on primary.data
            # Convert or preserve uint16
            if data.dtype != np.uint16:
                # If astropy converted uint16 with BZERO=32768 to int32 or float64
                if np.min(data) >= 0 and np.max(data) <= 65535:
                    data = data.astype(np.uint16)
                else:
                    data = np.clip(data, 0, 65535).astype(np.uint16)
            else:
                data = data.copy()
                
            height, width = data.shape
            metadata = {
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                "image_id": os.path.splitext(os.path.basename(filepath))[0],
                "width": int(width),
                "height": int(height),
                "bitpix": int(hdr.get("BITPIX", 16)),
                "bscale": float(hdr.get("BSCALE", 1.0)),
                "bzero": float(hdr.get("BZERO", 32768.0)),
                "date_obs": str(hdr.get("DATE-OBS", "UNKNOWN")),
                "instrument": str(hdr.get("INSTRUME", "UNKNOWN")),
                "exptime": float(hdr.get("EXPTIME", 0.0)) if "EXPTIME" in hdr else None,
                "gain": float(hdr.get("GAIN", 0.0)) if "GAIN" in hdr else None,
                "raw_min": int(np.min(data)),
                "raw_max": int(np.max(data)),
            }
            
            return data, metadata
