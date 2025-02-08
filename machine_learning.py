def run_fmri_decoding():
    """
    This function demonstrates a full fMRI decoding pipeline using Nilearn.
    It downloads the Haxby dataset, visualizes the data and mask, extracts
    features, and performs various decoding (classification) analyses with
    cross-validation, including saving and plotting the SVC weight maps.
    
    Note: This function may open interactive plots (e.g. via view_img) and
    downloads about 310 MB of data if not already cached.
    """
    # --- Step 1: Retrieve the Haxby dataset ---
    from nilearn import datasets, plotting, decoding
    import pandas as pd
    from nilearn.image import mean_img, index_img
    from sklearn.model_selection import KFold, LeaveOneGroupOut
    from pathlib import Path

    # Fetch the Haxby dataset (if not already cached)
    haxby_dataset = datasets.fetch_haxby()
    # The first subject's functional (4D) nifti image is stored in fmri_filename
    fmri_filename = haxby_dataset.func[0]

    # --- Step 2: Visualize a mean fMRI image ---
    # Create a mean image from the 4D fMRI data for visualization purposes.
    mean_epi = mean_img(fmri_filename, copy_header=True)
    # Display the mean fMRI image (opens an interactive viewer)
    plotting.view_img(mean_epi, threshold=None)

    # --- Step 3: Define and visualize the mask ---
    # Retrieve the mask of the ventral temporal (VT) cortex from the dataset.
    mask_filename = haxby_dataset.mask_vt[0]
    # Plot the VT mask using the subject's anatomical image as the background.
    plotting.plot_roi(mask_filename, bg_img=haxby_dataset.anat[0], cmap="Paired")

    # --- Step 4: Load behavioral labels ---
    # Load the behavioral information from a CSV file using pandas.
    behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=" ")
    # The 'labels' column indicates the experimental conditions.
    conditions = behavioral["labels"]

    # --- Step 5: Select only "face" and "cat" conditions ---
    # Create a mask to select only samples with labels "face" or "cat".
    condition_mask = conditions.isin(["face", "cat"])
    # Because the data is a single 4D image, extract the selected images with index_img.
    fmri_niimgs = index_img(fmri_filename, condition_mask)
    # Convert the selected conditions to a NumPy array.
    conditions = conditions[condition_mask].to_numpy()
    # (After this step, 'conditions' has a shape of (216,))

    # --- Step 6: Decoding with a Support Vector Classifier (SVC) ---
    # Create a Decoder object with an SVC estimator and z-score normalization.
    decoder = decoding.Decoder(
        estimator="svc", mask=mask_filename, standardize="zscore_sample"
    )
    # Fit the decoder on the data and corresponding conditions.
    decoder.fit(fmri_niimgs, conditions)
    # Use the trained decoder to predict labels on the same data.
    prediction = decoder.predict(fmri_niimgs)
    # Compute training accuracy (note: this accuracy is not cross-validated)
    acc = (prediction == conditions).sum() / float(len(conditions))
    # The variable 'acc' holds the training accuracy.

    # --- Step 7: Manual leave-out cross-validation ---
    # Leave out the last 30 data points as a test set.
    fmri_niimgs_train = index_img(fmri_niimgs, slice(0, -30))
    fmri_niimgs_test = index_img(fmri_niimgs, slice(-30, None))
    conditions_train = conditions[:-30]
    conditions_test = conditions[-30:]
    # Create and fit a new decoder on the training set.
    decoder_cv = decoding.Decoder(
        estimator="svc", mask=mask_filename, standardize="zscore_sample"
    )
    decoder_cv.fit(fmri_niimgs_train, conditions_train)
    # Predict the labels on the test set.
    prediction_cv = decoder_cv.predict(fmri_niimgs_test)
    # Compute the test accuracy.
    test_acc = (prediction_cv == conditions_test).sum() / float(len(conditions_test))
    # The variable 'test_acc' holds the accuracy for the manual leave-out cross-validation.

    # --- Step 8: KFold cross-validation (manual loop) ---
    # Define a 5-fold cross-validation strategy.
    cv = KFold(n_splits=5)
    # Iterate over each fold.
    for fold, (train, test) in enumerate(cv.split(conditions), start=1):
        # Create and fit a decoder for the current fold.
        decoder_fold = decoding.Decoder(
            estimator="svc", mask=mask_filename, standardize="zscore_sample"
        )
        decoder_fold.fit(index_img(fmri_niimgs, train), conditions[train])
        # Predict labels for the test split.
        prediction_fold = decoder_fold.predict(index_img(fmri_niimgs, test))
        # Compute the accuracy for this fold.
        fold_acc = (prediction_fold == conditions[test]).sum() / float(len(conditions[test]))
        # The variable 'fold_acc' holds the accuracy for the current fold.

    # --- Step 9: Built-in cross-validation with the Decoder ---
    # Perform built-in cross-validation with 5 folds.
    n_folds = 5
    decoder_cv_builtin = decoding.Decoder(
        estimator="svc",
        mask=mask_filename,
        standardize="zscore_sample",
        cv=n_folds,
        scoring="accuracy",
    )
    decoder_cv_builtin.fit(fmri_niimgs, conditions)
    # The attribute 'cv_params_' stores the best performing parameters per fold.
    # For example, the best parameters for the 'face' class (if available) could be accessed via:
    # best_params_face = decoder_cv_builtin.cv_params_.get("face", None)

    # --- Step 10: Cross-validation with LeaveOneGroupOut ---
    # Import LeaveOneGroupOut for cross-validation that respects run boundaries.
    from sklearn.model_selection import LeaveOneGroupOut
    cv_leaveone = LeaveOneGroupOut()
    # Extract the run labels (or chunks) for the selected conditions.
    run_label = behavioral["chunks"][condition_mask].to_numpy()
    # Create a decoder with LeaveOneGroupOut cross-validation.
    decoder_leaveone = decoding.Decoder(
        estimator="svc", mask=mask_filename, standardize="zscore_sample", cv=cv_leaveone
    )
    decoder_leaveone.fit(fmri_niimgs, conditions, groups=run_label)
    # The attribute 'cv_scores_' contains the cross-validation scores for each class.
    cv_scores = decoder_leaveone.cv_scores_

    # --- Step 11: Inspect and save model weights ---
    # Retrieve the SVC discriminating weights (coefficients).
    coef_ = decoder_leaveone.coef_
    # 'coef_' is a numpy array with shape (1, number_of_voxels).
    # Retrieve the coefficient image for the 'face' class.
    coef_img = decoder_leaveone.coef_img_["face"]
    # Define an output directory and save the coefficient image as a Nifti file.
    output_dir = Path.cwd() / "results" / "plot_decoding_tutorial"
    output_dir.mkdir(exist_ok=True, parents=True)
    coef_img.to_filename(output_dir / "haxby_svc_weights.nii.gz")
    # The SVC weight map has now been saved to disk.

    # --- Step 12: Plot the SVC weight map ---
    # Display the weight map overlaid on the subject's anatomical image.
    plotting.view_img(
        coef_img,
        bg_img=haxby_dataset.anat[0],
        title="SVM weights",
        dim=-1,
    )

    # --- Step 13: Estimate chance level performance with a dummy classifier ---
    # Create a dummy classifier that uses a simple baseline strategy.
    dummy_decoder = decoding.Decoder(
        estimator="dummy_classifier",
        mask=mask_filename,
        cv=cv_leaveone,
        standardize="zscore_sample",
    )
    # Fit the dummy classifier on the data.
    dummy_decoder.fit(fmri_niimgs, conditions, groups=run_label)
    # The attribute 'cv_scores_' now contains the dummy classifier's cross-validation scores,
    # which serve as an estimate of chance-level performance.


if __name__ == "__main__":
    run_fmri_decoding_tutorial()
