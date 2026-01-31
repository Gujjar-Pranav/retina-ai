# src/ui_style.py
from __future__ import annotations
import streamlit as st


def inject_global_css() -> None:
    """
    Clean hospital-grade UI:
    - Fix top header cut-off
    - Reduce sidebar font size but keep readable
    - Card layout + badges + spacing helpers
    """
    st.markdown(
        """
        <style>
        /* ---------- Global layout ---------- */
        .block-container {
            padding-top: 2.2rem !important;   /* FIX: avoids title overlap/cut */
            padding-bottom: 2.2rem !important;
            max-width: 1180px;
        }

        /* ---------- Typography ---------- */
        html, body, [class*="css"]  {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        }

        h1, h2, h3, h4 {
            letter-spacing: 0.2px;
            color: #0F172A;
        }

        /* Slightly more “clinical” look */
        .muted {
            color: rgba(15, 23, 42, 0.65);
            font-size: 0.92rem;
        }

        /* ---------- Spacing helpers ---------- */
        .spacer-6 { height: 6px; }
        .spacer-10 { height: 10px; }
        .spacer-12 { height: 12px; }
        .spacer-16 { height: 16px; }
        .spacer-20 { height: 20px; }

        /* ---------- Cards ---------- */
        .card {
            background: #FFFFFF;
            border: 1px solid rgba(2, 6, 23, 0.10);
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 2px 10px rgba(2, 6, 23, 0.04);
        }

        .row {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        /* ---------- Badges ---------- */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: 0.82rem;
            border: 1px solid rgba(2, 6, 23, 0.10);
        }

        .badge-ok {
            background: #0F766E;
            color: white;
            border: none;
        }

        .badge-review {
            background: #B45309;
            color: white;
            border: none;
        }

        .badge-retake {
            background: #991B1B;
            color: white;
            border: none;
        }

        .badge-gray {
            background: #F1F5F9;
            color: #0F172A;
        }

        /* ---------- Sidebar styling ---------- */
        section[data-testid="stSidebar"] {
            background: #FBFDFF;
            border-right: 1px solid rgba(2, 6, 23, 0.08);
        }

        section[data-testid="stSidebar"] * {
            font-size: 0.86rem !important;  /* smaller font but readable */
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-size: 0.95rem !important;
        }

        /* Sidebar tiny helper text */
        .tiny {
            font-size: 0.80rem;
            color: rgba(15, 23, 42, 0.62);
            line-height: 1.35;
        }

        /* Improve Streamlit input spacing */
        div[data-testid="stTextInput"] > label,
        div[data-testid="stNumberInput"] > label,
        div[data-testid="stSelectbox"] > label,
        div[data-testid="stToggle"] > label,
        div[data-testid="stSlider"] > label {
            font-weight: 600;
            color: #0F172A;
        }

        /* ---------- Buttons ---------- */
        button[kind="primary"] {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
