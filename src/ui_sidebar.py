# src/ui_sidebar.py
import streamlit as st
from pathlib import Path


def render_sidebar():
    """
    Renders the clinical configuration sidebar.
    Returns a dict of runtime configuration.
    """

    with st.sidebar:
        st.markdown("### ⚙️ System Controls")

        model_path = st.text_input(
            "Model checkpoint",
            value="models/efficientnet_b0_best.pth",
            help="Path to trained EfficientNet model",
        )

        st.markdown("---")
        st.markdown("#### 🔒 Decision Gates")

        confidence_gate = st.toggle(
            "Enable confidence gate (OK / REVIEW)",
            value=True,
        )
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.50,
            max_value=0.99,
            value=0.90,
            step=0.01,
        )

        quality_gate = st.toggle(
            "Enable image quality gate (OK / RETAKE)",
            value=True,
        )

        st.markdown("**Image quality thresholds**")
        brightness_min = st.slider("Brightness min", 0.00, 0.50, 0.08, 0.01)
        brightness_max = st.slider("Brightness max", 0.10, 0.90, 0.30, 0.01)
        contrast_min = st.slider("Contrast min", 0.00, 0.30, 0.10, 0.01)
        sharpness_min = st.slider("Sharpness min", 0.00, 0.20, 0.00, 0.01)

        st.markdown("---")
        st.markdown("#### 📄 Report & Visuals")

        facility_name = st.text_input(
            "Facility / Hospital name",
            value="Retina-AI Eye Clinic",
        )

        show_probabilities = st.toggle("Show probabilities", value=True)
        show_gradcam = st.toggle("Show Grad-CAM", value=True)

    return {
        "model_path": model_path,
        "confidence_gate": confidence_gate,
        "confidence_threshold": confidence_threshold,
        "quality_gate": quality_gate,
        "brightness_min": brightness_min,
        "brightness_max": brightness_max,
        "contrast_min": contrast_min,
        "sharpness_min": sharpness_min,
        "facility_name": facility_name,
        "show_probabilities": show_probabilities,
        "show_gradcam": show_gradcam,
    }
