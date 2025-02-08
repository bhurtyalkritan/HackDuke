# plotting.py

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def plot_slice(data, slice_number, axis=0):
    """Plot a 2D slice of the 3D volume (data)."""
    fig, ax = plt.subplots()
    if axis == 0:  # Sagittal
        slice_data = data[slice_number, :, :]
    elif axis == 1:  # Coronal
        slice_data = data[:, slice_number, :]
    else:  # Axial
        slice_data = data[:, :, slice_number]

    ax.imshow(slice_data.T, cmap="gray", origin="lower")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    return fig

def plot_highlighted_slice(data, slice_number, axis, labels_img, region_label):
    """Plot a slice with a region highlighted in red contour."""
    fig, ax = plt.subplots()
    if axis == 0:  # Sagittal
        slice_data = data[slice_number, :, :]
        roi_data = labels_img.get_fdata()[slice_number, :, :] == region_label
    elif axis == 1:  # Coronal
        slice_data = data[:, slice_number, :]
        roi_data = labels_img.get_fdata()[:, slice_number, :] == region_label
    else:  # Axial
        slice_data = data[:, :, slice_number]
        roi_data = labels_img.get_fdata()[:, :, slice_number] == region_label

    ax.imshow(slice_data.T, cmap="gray", origin="lower")
    ax.contour(roi_data, colors='red', linewidths=0.5)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    return fig

def plot_3d_brain(data, labels_img, atlas_labels, annotations=None):
    """
    Create a 3D scatter plot of the brain highlighting high-intensity voxels with clickable annotations.
    
    Parameters:
    -----------
    data : numpy.ndarray
        3D array of brain data
    labels_img : nibabel image
        Atlas labels image
    atlas_labels : dict
        Dictionary mapping label indices to region names
    annotations : dict, optional
        Dictionary of existing annotations with format {(x,y,z): 'annotation text'}
    """
    if annotations is None:
        annotations = {}
    
    coords = np.array(np.nonzero(data > np.percentile(data, 95)))
    x, y, z = coords
    intensities = data[x, y, z]
    regions = labels_img.get_fdata()[x, y, z]

    # Create hover texts including any existing annotations
    hover_texts = []
    for i, (region, intensity) in enumerate(zip(regions, intensities)):
        point_coord = (int(x[i]), int(y[i]), int(z[i]))
        annotation = annotations.get(point_coord, "")
        text = f"Region: {atlas_labels[int(region)]}<br>Intensity: {intensity:.2f}"
        if annotation:
            text += f"<br>Note: {annotation}"
        hover_texts.append(text)

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=intensities,
            colorscale='Viridis',
            colorbar=dict(title='Intensity', thickness=15, xpad=10),
            opacity=0.8
        ),
        text=hover_texts,
        hoverinfo='text',
        customdata=list(zip(x, y, z)),  # Store coordinates for click events
        hovertemplate="%{text}<extra></extra>"
    )])

    # Add a button to toggle annotation mode
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Left-Right'),
            yaxis=dict(title='Posterior-Anterior'),
            zaxis=dict(title='Inferior-Superior')
        ),
        margin=dict(r=10, l=10, b=10, t=10),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="Add Annotation",
                method="relayout",
                args=[{"clickmode": "event"}]
            )]
        )],
        clickmode="event"
    )
    return fig

def annotate_slice(data, slice_number, axis=0, annotations=[]):
    """Plot a slice and place textual annotations."""
    fig, ax = plt.subplots()
    if axis == 0:  # Sagittal
        slice_data = data[slice_number, :, :]
    elif axis == 1:  # Coronal
        slice_data = data[:, slice_number, :]
    else:  # Axial
        slice_data = data[:, :, slice_number]

    ax.imshow(slice_data.T, cmap="gray", origin="lower")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    # Overlay annotations
    for annotation in annotations:
        ax.text(annotation['x'], annotation['y'], annotation['text'],
                color='red', fontsize=12, ha='left')

    return fig

def generate_charts(stats_df):
    """Generate scatter and pie charts for region statistics."""
    fig_scatter = px.scatter(
        stats_df,
        x='Volume',
        y='Mean Intensity',
        color='Region',
        title="Volume vs Intensity Scatter Plot"
    )

    fig_pie = px.pie(
        stats_df,
        values='Volume',
        names='Region',
        title="Volume Distribution by Region"
    )
    return fig_scatter, fig_pie

def plot_time_series(time_series, mean_intensity_over_time, region_label, atlas_labels):
    """Plot mean intensity over time for a given region."""
    fig = px.line(
        x=range(len(mean_intensity_over_time)),
        y=mean_intensity_over_time,
        labels={'x': 'Time Point', 'y': 'Mean Intensity'},
        title=f'Mean Intensity Over Time for {atlas_labels[int(region_label)]}'
    )
    return fig
