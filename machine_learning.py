def run_fmri_decoding():
    from nilearn import datasets, plotting, decoding
    import pandas as pd
    from nilearn.image import mean_img, index_img
    from sklearn.model_selection import KFold, LeaveOneGroupOut
    from nilearn.decoding import Decoder
    from sklearn.model_selection import train_test_split
    from pathlib import Path

    haxby_dataset = datasets.fetch_haxby()
    fmri_filename = haxby_dataset.func[0]
    mean_epi = mean_img(fmri_filename, copy_header=True)
    plotting.view_img(mean_epi, threshold=None)
    mask_filename = haxby_dataset.mask_vt[0]
    plotting.plot_roi(mask_filename, bg_img=haxby_dataset.anat[0], cmap="Paired")
    behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=" ")
    conditions = behavioral["labels"]
    condition_mask = conditions.isin(["face", "cat"])
    fmri_niimgs = index_img(fmri_filename, condition_mask)
    conditions = conditions[condition_mask].to_numpy()
    
    decoder = decoding.Decoder(
        estimator="svc", mask=mask_filename, standardize="zscore_sample"
    )
    decoder.fit(fmri_niimgs, conditions)
    prediction = decoder.predict(fmri_niimgs)
    acc = (prediction == conditions).sum() / float(len(conditions))
    
    fmri_niimgs_train = index_img(fmri_niimgs, slice(0, -30))
    fmri_niimgs_test = index_img(fmri_niimgs, slice(-30, None))
    conditions_train = conditions[:-30]
    conditions_test = conditions[-30:]
    
    decoder_cv = decoding.Decoder(
        estimator="svc", mask=mask_filename, standardize="zscore_sample"
    )
    decoder_cv.fit(fmri_niimgs_train, conditions_train)
    prediction_cv = decoder_cv.predict(fmri_niimgs_test)
    test_acc = (prediction_cv == conditions_test).sum() / float(len(conditions_test))
    
    cv = KFold(n_splits=5)
    for fold, (train, test) in enumerate(cv.split(conditions), start=1):
        decoder_fold = decoding.Decoder(
            estimator="svc", mask=mask_filename, standardize="zscore_sample"
        )
        decoder_fold.fit(index_img(fmri_niimgs, train), conditions[train])
        prediction_fold = decoder_fold.predict(index_img(fmri_niimgs, test))
        fold_acc = (prediction_fold == conditions[test]).sum() / float(len(conditions[test]))
    
    n_folds = 5
    decoder_cv_builtin = decoding.Decoder(
        estimator="svc",
        mask=mask_filename,
        standardize="zscore_sample",
        cv=n_folds,
        scoring="accuracy",
    )
    decoder_cv_builtin.fit(fmri_niimgs, conditions)
    
    from sklearn.model_selection import LeaveOneGroupOut
    cv_leaveone = LeaveOneGroupOut()
    run_label = behavioral["chunks"][condition_mask].to_numpy()
    decoder_leaveone = decoding.Decoder(
        estimator="svc", mask=mask_filename, standardize="zscore_sample", cv=cv_leaveone
    )
    decoder_leaveone.fit(fmri_niimgs, conditions, groups=run_label)
    cv_scores = decoder_leaveone.cv_scores_
    
    coef_ = decoder_leaveone.coef_
    coef_img = decoder_leaveone.coef_img_["face"]
    output_dir = Path.cwd() / "results" / "plot_decoding_tutorial"
    output_dir.mkdir(exist_ok=True, parents=True)
    coef_img.to_filename(output_dir / "haxby_svc_weights.nii.gz")
    
    plotting.view_img(
        coef_img,
        bg_img=haxby_dataset.anat[0],
        title="SVM weights",
        dim=-1,
    )
    
    dummy_decoder = decoding.Decoder(
        estimator="dummy_classifier",
        mask=mask_filename,
        cv=cv_leaveone,
        standardize="zscore_sample",
    )
    dummy_decoder.fit(fmri_niimgs, conditions, groups=run_label)

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

def classify_schizophrenia(fmri_data, labels=None, model=None, test_size=0.2, random_state=42):
    """
    Classify schizophrenia patients using fMRI data.
    
    Parameters:
    -----------
    fmri_data : list of Nifti1Image
        List of fMRI images for classification
    labels : array-like, optional
        Corresponding labels (1 for schizophrenia, 0 for control)
        Required for training mode
    model : Decoder, optional
        Pre-trained Decoder model for prediction
    test_size : float, default=0.2
        Proportion of data to use for testing (training mode only)
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    If in training mode:
        - trained_model : Decoder
        - test_predictions : array
        - test_scores : array
    If in prediction mode:
        - predictions : array
    """
    if model is None and labels is None:
        raise ValueError("Either a pre-trained model or labels must be provided")
        
    if model is None:
        # Training mode
        X_train, X_test, y_train, y_test = train_test_split(
            fmri_data, labels, test_size=test_size, random_state=random_state
        )
        
        decoder = Decoder(
            estimator='svc',
            mask_strategy='whole-brain-template',
            standardize=True,
            cv=5
        )
        
        decoder.fit(X_train, y_train)
        predictions = decoder.predict(X_test)
        scores = decoder.decision_function(X_test)
        
        return decoder, predictions, scores
    else:
        # Prediction mode
        return model.predict(fmri_data)