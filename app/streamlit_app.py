from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch


def _ensure_project_root_on_path() -> Path:
    """
    Keep original behavior: add project root so `src/` imports work.
    Moved into a function to satisfy Ruff E402 (no runtime code at module import time).
    """
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# NOTE: src.* imports happen AFTER root path is ensured (inside main)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _fmt_mtime(ts: float) -> str:
    # Local time display
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def render_reports_tab(root: Path) -> None:
    """
    Staff + Admin can view/download already generated PDFs.
    """
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    st.markdown("### Reports")
    st.caption("Download previously generated PDF screening reports.")

    pdfs = sorted(reports_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not pdfs:
        st.info("No reports generated yet.")
        return

    q = st.text_input("Search reports", placeholder="Search by report ID / filename (e.g., RPT-12AB34CDEF)")
    if q.strip():
        qq = q.strip().lower()
        pdfs = [p for p in pdfs if qq in p.name.lower()]

    st.markdown(f"**Found:** {len(pdfs)} report(s)")
    st.markdown('<div class="spacer-8"></div>', unsafe_allow_html=True)

    for p in pdfs[:50]:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1], gap="large")
            with c1:
                st.markdown(f"**{p.name}**")
                st.caption(f"Last modified: {_fmt_mtime(p.stat().st_mtime)}")
            with c2:
                with open(p, "rb") as f:
                    st.download_button(
                        "⬇️ Download",
                        data=f,
                        file_name=p.name,
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"dl_{p.name}",
                    )


def main() -> None:
    ROOT = _ensure_project_root_on_path()

    # Import after path fix (no functionality change)
    from src.auth import (
        admin_create_user_panel,
        can_access_registry,
        can_access_reports,
        can_access_screening,
        require_login,
        sidebar_identity,
    )
    from src.model_loader import build_and_load
    from src.ui_registry import render_registry
    from src.ui_screening import render_screening
    from src.ui_sidebar import render_sidebar
    from src.ui_style import inject_global_css

    st.set_page_config(
        page_title="Retina-AI — Clinical DR Screening",
        page_icon="🧿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_global_css()

    # ✅ Login gate
    user = require_login(ROOT)

    # Sidebar identity + logout
    sidebar_identity(user)

    # Admin-only user management panel (sidebar expander)
    admin_create_user_panel(ROOT, user)

    # Sidebar controls (ONLY for Screening users)
    cfg: dict = {}
    if can_access_screening(user.role):
        cfg = render_sidebar()

    # Header
    st.markdown("## 🧿 Retina-AI — Clinical DR Screening")
    st.markdown(
        '<div class="muted">AI-assisted diabetic retinopathy screening • For clinical decision support only</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="spacer-12"></div>', unsafe_allow_html=True)

    device = choose_device()

    # Load model only if Screening is accessible
    model = None
    backend = None
    if can_access_screening(user.role):
        try:
            model, backend = build_and_load(
                model_path=cfg.get("model_path", ""),
                device=device,
            )
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.stop()

    # Build tabs based on role
    tabs: list[str] = []
    if can_access_registry(user.role):
        tabs.append("Registry")
    if can_access_screening(user.role):
        tabs.append("Screening")
    if can_access_reports(user.role):
        tabs.append("Reports")

    if not tabs:
        st.error("Your role has no access to any modules. Contact admin.")
        st.stop()

    tab_objs = st.tabs(tabs)

    for name, tab in zip(tabs, tab_objs):
        with tab:
            if name == "Registry":
                render_registry(ROOT=ROOT)

            elif name == "Screening":
                render_screening(
                    ROOT=ROOT,
                    device=device,
                    model=model,
                    backend=backend,
                )

            elif name == "Reports":
                render_reports_tab(ROOT)


if __name__ == "__main__":
    main()
