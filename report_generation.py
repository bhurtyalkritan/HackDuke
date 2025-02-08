# report_generation.py

import io
import pandas as pd
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle)
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
def generate_pdf_visit(fig_scatter,
                        fig_pie,
                        fig_time_series,
                        fig_3d_brain,
                        fig_axial,
                        fig_coronal,
                        fig_sagittal,
                        coef_df,
                        results,
                        selected_region):
    """
    Generate a PDF report using ReportLab.
    Returns a BytesIO buffer containing the PDF data.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Caption', fontSize=10, leading=12, spaceAfter=6))
    elements = []

    def add_image(fig, elements, caption, styles, is_plotly=False):
        """
        Utility to convert a Matplotlib/Plotly figure to image and insert into ReportLab.
        """
        img_buffer = io.BytesIO()
        if is_plotly:
            fig.write_image(img_buffer, format='png')
        else:
            fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img = Image(img_buffer, width=5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(caption, styles['Caption']))
        elements.append(Spacer(1, 12))

    # Title Page
    elements.append(Paragraph("MRI Scan Visit Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Patient: Jack Doe", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Patient ID: 213465798", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Visit Date: 2/08/2025", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Institution: Healthcare Hospitals", styles['Normal']))
    elements.append(Spacer(1, 24))

    # Abstract
    elements.append(Paragraph("Abstract", styles['Heading1']))
    abstract_text = (
        "This report provides an in-depth analysis of brain imaging data using a Generalized "
        "Linear Model (GLM). The GLM approach allows for flexible modeling of various types of "
        "response variables, including count data, binary data, and continuous positive values. "
        "The aim is to explore brain imaging data using advancedstatistical techniques to identify patterns  "
        "and significant changes in brain activity over time. This approach allows for a better analysis"
        "of one's brain and helps in visualization for both the patient and the Doctors. Attached with this report"
        "will be a file containing your brain scans. Ask your Neurologist for help in observing your MRI data,"
        "and to understand these results" 
    )
    elements.append(Paragraph(abstract_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Introduction
    elements.append(Paragraph("Clinical History", styles['Heading1']))
    intro_text = (
        "The patient has had a long history of sudden migraines and occasional seizures."
        "Patient has no prior history of any head trauma, and has no history of neurological disorders "
        "Patient's grandfather is diagnosed with epilepsy, but has no other family history of neurological disorders "
        "MRI scan requested to rule out any neurological abnormalities."
    )
    elements.append(Paragraph(intro_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Methods
    elements.append(Paragraph("Methods", styles['Heading1']))
    methods_text = (
        "Proceeding the MRI scan, the results were taken and converted into an NII file for"
        "collection and processing. NiBabel was utilized for reading and writing neuroimaging files, "
        "and Nilearn was used for processing and analyzing the data. The data was consequently paired"
        "with GLM Time Series Analysis for results."
    )
    elements.append(Paragraph(methods_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Exploratory Data Analysis
    elements.append(Paragraph("MRI Analysis", styles['Heading1']))
    elements.append(Spacer(1, 12))

    # Add Figures with Captions
    add_image(fig_scatter, elements, "Figure 1: Volume vs Intensity Scatter Plot", styles, is_plotly=True)
    add_image(fig_pie, elements, "Figure 2: Volume Distribution by Region", styles, is_plotly=True)
    add_image(fig_3d_brain, elements, "Figure 3: 3D Brain Plot", styles, is_plotly=True)
    add_image(fig_axial, elements, "Figure 4: Axial Slice View", styles)
    add_image(fig_coronal, elements, "Figure 5: Coronal Slice View", styles)
    add_image(fig_sagittal, elements, "Figure 6: Sagittal Slice View", styles)

    # Results
    elements.append(Paragraph("Results", styles['Heading1']))
    results_text = (
        "The GLM analysis for the selected brain region revealed the following key results. "
        "The deviance and Pearson chi-squared values indicate the goodness-of-fit of the model. "
        "The table below presents the coefficients, standard errors, z-values, and p-values for "
        "the model parameters."
    )
    elements.append(Paragraph(results_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Add GLM Results Table
    coef_table_data = [coef_df.columns.values.tolist()] + coef_df.values.tolist()
    from reportlab.platypus import TableStyle, Table
    coef_table = Table(coef_table_data)
    coef_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(coef_table)
    elements.append(Spacer(1, 12))

    # Discussion
    elements.append(Paragraph("Discussion", styles['Heading1']))
    discussion_text = (
        "The discussion section provides an interpretation of the GLM results. The analysis "
        "demonstrated the significance of the selected brain region and its activity over time. "
        "The model's deviance and Pearson chi-squared values indicate a good fit, suggesting the "
        "robustness of the findings. This section elaborates on the implications of the results, "
        "their relevance to the research objectives, and potential areas for future research."
    )
    elements.append(Paragraph(discussion_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Overview
    elements.append(Paragraph("Radiologist Note:", styles['Heading1']))
    overview_text = (
        "This section provides the Radiologist Note regarding the conclusions made from the"
        "data analysis performed"
    )
    elements.append(Paragraph(overview_text, styles['Normal']))
    elements.append(Spacer(1, 12))


    # Libraries Used
    elements.append(Paragraph("Libraries Used", styles['Heading1']))
    libraries_text = (
        "The following libraries were used in this Analysis:"
    )
    elements.append(Paragraph(libraries_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    libraries = [
        "- **Nilearn**: Simplifies scikit-learn in the context of neuroimaging.",
        "- **NiBabel**: Provides read and write access to various neuroimaging file formats.",
        "- **Streamlit**: Used to create an interactive web application interface.",
        "- **Matplotlib and Plotly**: Used for generating visualizations.",
        "- **Statsmodels**: Used for statistical analysis.",
        "- **ReportLab**: Used for generating PDF reports."
    ]
    for library in libraries:
        elements.append(Paragraph(library, styles['Normal']))
        elements.append(Spacer(1, 12))

    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

#-----------------------------------------------------------------------------------

def generate_pdf_report(fig_scatter,
                        fig_pie,
                        fig_time_series,
                        fig_3d_brain,
                        fig_axial,
                        fig_coronal,
                        fig_sagittal,
                        coef_df,
                        results,
                        selected_region):
    """
    Generate a PDF report using ReportLab.
    Returns a BytesIO buffer containing the PDF data.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Caption', fontSize=10, leading=12, spaceAfter=6))
    elements = []

    def add_image(fig, elements, caption, styles, is_plotly=False):
        """
        Utility to convert a Matplotlib/Plotly figure to image and insert into ReportLab.
        """
        img_buffer = io.BytesIO()
        if is_plotly:
            fig.write_image(img_buffer, format='png')
        else:
            fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img = Image(img_buffer, width=5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(caption, styles['Caption']))
        elements.append(Spacer(1, 12))

    # Title Page
    elements.append(Paragraph("Brain Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Author: Your Name", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Institution: Your Institution", styles['Normal']))
    elements.append(Spacer(1, 24))

    # Abstract
    elements.append(Paragraph("Abstract", styles['Heading1']))
    abstract_text = (
        "This report provides an in-depth analysis of brain imaging data using a Generalized "
        "Linear Model (GLM). The GLM approach allows for flexible modeling of various types of "
        "response variables, including count data, binary data, and continuous positive values. "
        "In this analysis, we focus on a specific region of the brain and assess its activity "
        "over time using time-series data. The aim is to explore brain imaging data using advanced "
        "statistical techniques to identify patterns and significant changes in brain activity over time."
    )
    elements.append(Paragraph(abstract_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Introduction
    elements.append(Paragraph("Introduction", styles['Heading1']))
    intro_text = (
        "The introduction outlines the importance of the study, previous research, and the gaps "
        "that this research aims to fill. We utilized neuroimaging data to identify patterns and "
        "significant changes in brain activity over time. This section provides background "
        "information and the objectives of the study, highlighting the significance of analyzing "
        "brain imaging data to understand neurological conditions better."
    )
    elements.append(Paragraph(intro_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Methods
    elements.append(Paragraph("Methods", styles['Heading1']))
    methods_text = (
        "We used NiBabel for reading and writing  our neuroimaging files, "
        "and Nilearn for processing and analyzing the data. Streamlit was employed to create an "
        "interactive web interface for analysis and visualization. This section provides detailed "
        "steps on how the data was handled, processed, and analyzed using various tools and libraries."
    )
    elements.append(Paragraph(methods_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Exploratory Data Analysis
    elements.append(Paragraph("Exploratory Data Analysis", styles['Heading1']))
    eda_text = (
        "Exploratory Data Analysis (EDA) is crucial for understanding the underlying patterns in "
        "the data. The following figures illustrate the data distribution and intensity values "
        "across different brain regions. We present 2D slice views (axial, coronal, sagittal) and "
        "a 3D brain plot to provide a comprehensive visual representation of the brain imaging data."
    )
    elements.append(Paragraph(eda_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Add Figures with Captions
    add_image(fig_scatter, elements, "Figure 1: Volume vs Intensity Scatter Plot", styles, is_plotly=True)
    add_image(fig_pie, elements, "Figure 2: Volume Distribution by Region", styles, is_plotly=True)
    add_image(fig_3d_brain, elements, "Figure 3: 3D Brain Plot", styles, is_plotly=True)
    add_image(fig_axial, elements, "Figure 4: Axial Slice View", styles)
    add_image(fig_coronal, elements, "Figure 5: Coronal Slice View", styles)
    add_image(fig_sagittal, elements, "Figure 6: Sagittal Slice View", styles)

    # Results
    elements.append(Paragraph("Results", styles['Heading1']))
    results_text = (
        "The GLM analysis for the selected brain region revealed the following key results. "
        "The deviance and Pearson chi-squared values indicate the goodness-of-fit of the model. "
        "The table below presents the coefficients, standard errors, z-values, and p-values for "
        "the model parameters."
    )
    elements.append(Paragraph(results_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Add GLM Results Table
    coef_table_data = [coef_df.columns.values.tolist()] + coef_df.values.tolist()
    from reportlab.platypus import TableStyle, Table
    coef_table = Table(coef_table_data)
    coef_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(coef_table)
    elements.append(Spacer(1, 12))

    # Discussion
    elements.append(Paragraph("Discussion", styles['Heading1']))
    discussion_text = (
        "The discussion section provides an interpretation of the GLM results. The analysis "
        "demonstrated the significance of the selected brain region and its activity over time. "
        "The model's deviance and Pearson chi-squared values indicate a good fit, suggesting the "
        "robustness of the findings. This section elaborates on the implications of the results, "
        "their relevance to the research objectives, and potential areas for future research."
    )
    elements.append(Paragraph(discussion_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Overview
    elements.append(Paragraph("Overview", styles['Heading1']))
    overview_text = (
        "This section provides a detailed overview of how the application was developed using "
        "various tools and libraries:"
    )
    elements.append(Paragraph(overview_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    steps = [
        "1. Load the necessary libraries and datasets.",
        "2. Develop functions for loading and processing NIfTI files using NiBabel.",
        "3. Implement visualization functions using Matplotlib and Plotly.",
        "4. Create interactive elements using Streamlit for user input and interaction.",
        "5. Perform statistical analysis using Statsmodels and display the results.",
        "6. Generate a detailed PDF report of the analysis using ReportLab."
    ]
    for step in steps:
        elements.append(Paragraph(step, styles['Normal']))
        elements.append(Spacer(1, 12))

    # Libraries Used
    elements.append(Paragraph("Libraries Used", styles['Heading1']))
    libraries_text = (
        "The following libraries were used in this application:"
    )
    elements.append(Paragraph(libraries_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    libraries = [
        "- **Nilearn**: Simplifies scikit-learn in the context of neuroimaging.",
        "- **NiBabel**: Provides read and write access to various neuroimaging file formats.",
        "- **Streamlit**: Used to create an interactive web application interface.",
        "- **Matplotlib and Plotly**: Used for generating visualizations.",
        "- **Statsmodels**: Used for statistical analysis.",
        "- **ReportLab**: Used for generating PDF reports."
    ]
    for library in libraries:
        elements.append(Paragraph(library, styles['Normal']))
        elements.append(Spacer(1, 12))

    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


# Example test PDF function
def test_pdf():
    """Generate a simple test PDF to verify functionality."""
    from reportlab.pdfgen import canvas
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica", 16)
    c.drawString(72, height - 72, "Test PDF Report")

    c.setFont("Helvetica", 12)
    c.drawString(72, height - 96, "This is a test report to check PDF generation.")

    c.save()
    buffer.seek(0)
    return buffer
