
from nilearn import datasets

atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')
atlas_labels = atlas['labels']
