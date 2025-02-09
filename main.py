import streamlit as st
from data_processing import classify_schizophrenia, load_nii_file
import nibabel as nib
import numpy as np

st.set_page_config(page_title="Schizophrenia Detection", layout="wide")

st.title("Schizophrenia Detection from fMRI Data")

# File upload section
uploaded_file = st.file_uploader("Upload fMRI NIfTI file", type=["nii", "nii.gz"])

if uploaded_file is not None:
    try:
        # Load and process the NIfTI file
        with st.spinner("Processing fMRI data..."):
            nii_img = load_nii_file(uploaded_file)
            
        # Perform classification
        with st.spinner("Analyzing for schizophrenia..."):
            model, predictions, scores = classify_schizophrenia([nii_img])
            
        # Display results
        st.success("Analysis complete!")
        st.subheader("Results")
        
        if predictions[0] == 1:
            st.error("Schizophrenia detected")
        else:
            st.success("No schizophrenia detected")
            
        st.write(f"Confidence score: {scores[0]:.2f}")
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a NIfTI file to begin analysis")
