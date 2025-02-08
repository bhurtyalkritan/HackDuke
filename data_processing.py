# data_processing.py

import nibabel as nib
import numpy as np
from nilearn import image as nli
from nilearn.masking import compute_brain_mask

def load_nii_file(uploaded_file):
    """Load a NIfTI file from a file-like object."""
    file_holder = nib.FileHolder(fileobj=uploaded_file)
    nii = nib.Nifti1Image.from_file_map({'header': file_holder, 'image': file_holder})
    return nii

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
