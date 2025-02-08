# atlas_data.py

from nilearn import datasets

# Fetch the Harvard-Oxford Cortical Atlas
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')
atlas_labels = atlas['labels']
