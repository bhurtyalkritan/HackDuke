# main.py

import streamlit as st
import numpy as np
import pandas as pd
import json
import io
import statsmodels.api as sm

from atlas_data import atlas, atlas_labels
from data_processing import (load_nii_file, skull_strip, apply_segmentation,
                             calculate_region_statistics, individual_statistics)
from plotting import (plot_slice, plot_highlighted_slice, plot_3d_brain,
                      annotate_slice, generate_charts, plot_time_series)
from stats_analysis import run_glm
from report_generation import generate_pdf_report, test_pdf, generate_pdf_visit

st.title('Neuro X')

if 'annotations' not in st.session_state:
    st.session_state.annotations = []

uploaded_file = st.sidebar.file_uploader("Choose a NII file", type=["nii", "gz"])

if uploaded_file:
    nii_data = load_nii_file(uploaded_file)
else:
    nii_data = load_nii_file('data/IIT_TDI_sum.nii')

visit=False
if True:
    data = nii_data.get_fdata()

    with st.expander("2D Slice and Annotations"):
        axis = st.selectbox('Select the axis for slicing:', options=['Sagittal', 'Coronal', 'Axial'], index=2)
        axis_map = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}
        slice_num = st.slider('Select Slice Number',
                              min_value=0,
                              max_value=data.shape[axis_map[axis]] - 1,
                              value=data.shape[axis_map[axis]] // 2)

        skull_strip_option = st.checkbox("Apply Skull Stripping")
        if skull_strip_option:
            nii_data = skull_strip(nii_data)
            data = nii_data.get_fdata()

        highlight_region = st.checkbox("Highlight Region")
        segmentation_applied = False
        labels_img = None

        if highlight_region:
            segmentation_applied = st.checkbox("Apply Segmentation for Highlighting")
            if segmentation_applied:
                labels_img = apply_segmentation(nii_data, atlas)
                region_index = st.selectbox(
                    "Select Region to Highlight",
                    options=[(i, atlas_labels[i]) for i in np.unique(labels_img.get_fdata().astype(int)) if i != 0]
                )
                region_label = region_index[0]
                fig = plot_highlighted_slice(data, slice_num, axis_map[axis], labels_img, region_label)
                st.pyplot(fig)
            else:
                fig = annotate_slice(data, slice_num, axis_map[axis], st.session_state.annotations)
                st.pyplot(fig)
        else:
            fig = annotate_slice(data, slice_num, axis_map[axis], st.session_state.annotations)
            st.pyplot(fig)

        buffer_fig = io.BytesIO()
        fig.savefig(buffer_fig, format="png")
        buffer_fig.seek(0)
        export_filename = f"{axis}_{slice_num}.png"
        st.download_button(
            label="Export Annotated Slice",
            data=buffer_fig,
            file_name=export_filename,
            mime="image/png"
        )

        annotations_json = json.dumps(st.session_state.annotations)
        st.download_button(
            label="Export Annotations",
            data=annotations_json,
            file_name="annotations.json",
            mime="application/json"
        )

        uploaded_annotations = st.sidebar.file_uploader("Import Annotations", type=["json"])
        if uploaded_annotations:
            st.session_state.annotations = json.loads(uploaded_annotations.getvalue())
            st.rerun()

        st.write("## Annotations")
        x = st.number_input("X Coordinate", min_value=0, max_value=data.shape[0] - 1, value=0)
        y = st.number_input("Y Coordinate", min_value=0, max_value=data.shape[1] - 1, value=0)
        text = st.text_input("Annotation Text")

        if st.button("Add Annotation"):
            st.session_state.annotations.append({'x': x, 'y': y, 'text': text})
            st.rerun()

        for i, annotation in enumerate(st.session_state.annotations):
            col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
            with col1:
                st.write(f"{i + 1}.")
            with col2:
                st.write(f"({annotation['x']}, {annotation['y']})")
            with col3:
                new_text = st.text_input(
                    f"Text for annotation {i + 1}",
                    annotation['text'],
                    key=f"annotation_{i}"
                )
                st.session_state.annotations[i]['text'] = new_text
            with col4:
                if st.button("Delete", key=f"delete_{i}"):
                    st.session_state.annotations.pop(i)
                    st.rerun()

    with st.sidebar.expander("3D Scan"):
        segmentation_applied_3d = st.checkbox("Apply Segmentation for 3D")
        labels_img_3d = None
        if segmentation_applied_3d:
            labels_img_3d = apply_segmentation(nii_data, atlas)
            region_index = st.selectbox(
                "Select Region for Statistics",
                options=[(i, atlas_labels[i]) for i in np.unique(labels_img_3d.get_fdata().astype(int)) if i != 0]
            )
            region_label = region_index[0]
            region_stats = individual_statistics(data, labels_img_3d, region_label, atlas_labels)
            st.write(f"**Region**: {region_stats['Region']}")
            st.write(f"**Mean Intensity**: {region_stats['Mean Intensity']:.2f}")
            st.write(f"**Volume**: {region_stats['Volume']}")

    uploaded_csv = st.sidebar.file_uploader("Upload time-series data (CSV) for GLM Analysis", type=["csv"])
    
    if uploaded_csv:
        time_series = pd.read_csv(uploaded_csv)
    else:
        time_series = pd.read_csv('data/time_series.csv')
        
    time_series = sm.add_constant(time_series) 

    with st.expander("3D View"):
        if segmentation_applied_3d and labels_img_3d is not None:
            fig_3d = plot_3d_brain(data, labels_img_3d, atlas_labels)
            st.plotly_chart(fig_3d, use_container_width=True)

    if segmentation_applied_3d and labels_img_3d is not None:
        stats = calculate_region_statistics(data, labels_img_3d, atlas_labels)
        stats_df = pd.DataFrame(stats)
        fig_scatter, fig_pie = generate_charts(stats_df)

        with st.expander("Statistical Analysis"):
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.plotly_chart(fig_pie, use_container_width=True)

        if uploaded_csv is not None and 'time_series' in locals():
                    region_index_val = region_index[0]
                    X = time_series[['Intercept', 'Condition1', 'Condition2']]  
                    y = time_series['Time']  

                    model = sm.GLM(y, X, family=sm.families.Gaussian())
                    results = model.fit()

                    st.write(results.summary())

                    t_test = results.t_test([1, 0, 0])  
                    st.write("T-test results for intercept:")
                    st.write(t_test.summary_frame())

                    st.write("**GLM Analysis Report:**")
                    st.write("The General Linear Model (GLM) analysis revealed the following key results:")
                    st.write(f"- **Deviance**: {results.deviance:.4f}")
                    st.write(f"- **Pearson Chi2**: {results.pearson_chi2:.4f}")
                    st.write(f"- **Coefficients:**")

                    coef_df = pd.DataFrame({
                        "Coefficient": results.params,
                        "Std Error": results.bse,
                        "z-value": results.tvalues,
                        "p-value": results.pvalues
                    })
                    st.table(coef_df)

                    if t_test.tvalue[0] > 1.96 or t_test.tvalue[0] < -1.96:
                        st.write(
                            "The t-test for the intercept is statistically significant at the 0.05 level, "
                            "indicating a significant relationship between the intercept and the response variable."
                        )
                    else:
                        st.write(
                            "The t-test for the intercept is not statistically significant at the 0.05 level, "
                            "suggesting that the relationship may not be significant."
                        )

                    fig_time_series = plot_time_series(time_series, time_series['Condition1'], region_label, atlas_labels)   
                    st.plotly_chart(fig_time_series, use_container_width=True)
                    fig_axial = plot_slice(data, data.shape[2]//2, axis=2)
                    fig_coronal = plot_slice(data, data.shape[1]//2, axis=1)
                    fig_sagittal = plot_slice(data, data.shape[0]//2, axis=0)

                    if st.button("Generate PDF Report"):
                        visit=False
                        st.session_state.generate_pdf = True
                    if st.button("Generate PDF Visit Report"):
                        visit=True
                        st.session_state.generate_pdf = True
                    if 'generate_pdf' in st.session_state and st.session_state.generate_pdf:
                        with st.spinner("Generating PDF report..."):

                            if visit==True:
                                pdf_output = generate_pdf_visit(
                                fig_scatter, 
                                fig_pie, 
                                fig_time_series,
                                fig_3d, 
                                fig_axial, 
                                fig_coronal, 
                                fig_sagittal,
                                coef_df, 
                                results, 
                                region_index)
                            else:
                                pdf_output = generate_pdf_report(
                                fig_scatter, 
                                fig_pie, 
                                fig_time_series,
                                fig_3d, 
                                fig_axial, 
                                fig_coronal, 
                                fig_sagittal,
                                coef_df, 
                                results, 
                                region_index
                            )
                            st.download_button(
                                label="Download Report",
                                data=pdf_output,
                                file_name="Brain_Analysis_Report.pdf",
                                mime="application/pdf"
                            )
                            st.session_state.generate_pdf = False

                    if st.button("Generate Test PDF"):
                        st.session_state.generate_test_pdf = True

                    if 'generate_test_pdf' in st.session_state and st.session_state.generate_test_pdf:
                        with st.spinner("Generating Test PDF report..."):
                            test_pdf_output = test_pdf()
                            st.download_button(
                                label="Download Test Report",
                                data=test_pdf_output,
                                file_name="Test_Report.pdf",
                                mime="application/pdf"
                            )
                            st.session_state.generate_test_pdf = False
