# stats_analysis.py

import numpy as np
import pandas as pd
import statsmodels.api as sm

def run_glm(region_data, labels_img, region_index, time_series, atlas_labels):
    """
    Run a GLM on the region_data and time_series.
    Returns the fitted results, T-test, mean_intensity_over_time, and a small summary dictionary.
    """
    # Number of time points in the NIfTI data (last dimension)
    num_time_points = region_data.shape[-1]
    num_time_points = min(num_time_points, time_series.shape[0])  # use the smaller one

    region_values = region_data[labels_img.get_fdata() == region_index]
    # Flatten across voxels, take only up to num_time_points
    region_values = region_values.flatten()[:num_time_points]

    # Run GLM
    glm_model = sm.GLM(np.asarray(region_values), np.asarray(time_series.iloc[:num_time_points]))
    results = glm_model.fit()

    # T-test for the intercept (first coefficient)
    t_test = results.t_test([1] + [0] * (time_series.shape[1] - 1))

    # Compute mean intensity over time
    # (Here, as an example, we assume each time point is a separate 3D volume.)
    # If region_data is 4D: shape = (X, Y, Z, T), we could do:
    # mean_intensity_over_time[i] = region_data[:,:,:,i][labels_img==region_index].mean()
    # but region_data might already be 3D if not functional data.
    # For demonstration, let's assume it is 4D:
    # mean_intensity_over_time = [region_data[:,:,:,i][labels_img.get_fdata()==region_index].mean() for i in range(num_time_points)]

    # If region_data is 2D or 3D (not time-series), you might need to adapt this code to your actual data structure.
    # We'll just compute a single mean if there's no time dimension.
    if len(region_data.shape) == 4:
        mean_intensity_over_time = []
        for i in range(num_time_points):
            vol_i = region_data[:,:,:,i]
            mean_intensity_over_time.append(vol_i[labels_img.get_fdata()==region_index].mean())
    else:
        # Fallback for 3D data (no time dimension)
        mean_intensity_over_time = [region_values.mean()] * num_time_points

    summary_dict = {
        "region_name": atlas_labels[int(region_index)],
        "deviance": results.deviance,
        "pearson_chi2": results.pearson_chi2
    }

    return results, t_test, mean_intensity_over_time, summary_dict
