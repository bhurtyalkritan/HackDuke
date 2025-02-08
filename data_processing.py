# data_processing.py

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nli
from nilearn.masking import compute_brain_mask
import os

# Default file paths
DEFAULT_NII_PATH = "data/IIT_TDI_sum.nii"
DEFAULT_TIMESERIES_PATH = "data/time_series.csv"

def load_nii_file(uploaded_file=None):
    """
    Load a NIfTI file from either a file-like object or use the default file.
    
    Args:
        uploaded_file: Optional file-like object containing NIfTI data
        
    Returns:
        nibabel.Nifti1Image: Loaded NIfTI image
    """
    if uploaded_file is None:
        if os.path.exists(DEFAULT_NII_PATH):
            return nib.load(DEFAULT_NII_PATH)
        else:
            raise FileNotFoundError(f"Default NIfTI file not found at {DEFAULT_NII_PATH}")
    
    file_holder = nib.FileHolder(fileobj=uploaded_file)
    nii = nib.Nifti1Image.from_file_map({'header': file_holder, 'image': file_holder})
    return nii

def load_time_series(uploaded_csv=None):
    """
    Load time series data from either a file-like object or use the default file.
    
    Args:
        uploaded_csv: Optional file-like object containing CSV data
        
    Returns:
        pandas.DataFrame: Loaded time series data
    """
    if uploaded_csv is None:
        if os.path.exists(DEFAULT_TIMESERIES_PATH):
            return pd.read_csv(DEFAULT_TIMESERIES_PATH)
        else:
            raise FileNotFoundError(f"Default time series file not found at {DEFAULT_TIMESERIES_PATH}")
    
    return pd.read_csv(uploaded_csv)

def skull_strip(nii_data):
    """Apply a brain mask to remove skull and non-brain tissues."""
    brain_mask = compute_brain_mask(nii_data)
    masked_img = nli.math_img("img1 * img2", img1=nii_data, img2=brain_mask)
    return masked_img

def apply_segmentation(nii_data, atlas_data):
    """Resample the atlas to match the input NIfTI file for segmentation."""
    labels_img = nli.resample_to_img(
        source_img=atlas_data.maps,
        target_img=nii_data,
        interpolation='nearest'
    )
    return labels_img

def calculate_region_statistics(data, labels_img, atlas_labels):
    """Compute mean intensity and volume (voxel count) for each region."""
    regions = np.unique(labels_img.get_fdata())
    stats = []
    for region in regions:
        if region == 0:  # skip background
            continue
        region_voxels = data[labels_img.get_fdata() == region]
        mean_intensity = np.mean(region_voxels)
        volume = np.count_nonzero(region_voxels)
        stats.append({
            'Region': atlas_labels[int(region)],
            'Mean Intensity': mean_intensity,
            'Volume': volume
        })
    return stats

def individual_statistics(data, labels_img, region_label, atlas_labels):
    """Compute stats for a single region (region_label)."""
    region_data = data[labels_img.get_fdata() == region_label]
    mean_intensity = np.mean(region_data)
    volume = np.count_nonzero(region_data)
    return {
        'Region': atlas_labels[int(region_label)],
        'Mean Intensity': mean_intensity,
        'Volume': volume
    }
