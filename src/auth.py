# src/auth.py
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

ROLE_ADMIN = "admin"
ROLE_STAFF = "staff"
ROLE_CLINICIAN = "clinician"

SESSION_KEY = "auth_user"


def users_file(root: Path) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "users.json"


@dataclass(frozen=True)
class AuthUser:
    user_id: str
    role: str


# -----------------------------
# Validation rules
# -----------------------------
def normalize_user_id(user_id: str) -> str:
    return (user_id or "").strip().lower()


def is_valid_user_id(user_id: str) -> bool:
    u = normalize_user_id(user_id)
    # allow letters, digits, underscore, dot, dash; 3-24 chars
    if not (3 <= len(u) <= 24):
        return False
    for ch in u:
        if not (ch.isalnum() or ch in {"_", "-", "."}):
            return False
    return True


def is_valid_password(pw: str) -> bool:
    pw = (pw or "").strip()
    return 4 <= len(pw) <= 8


# -----------------------------
# Hashing (lightweight)
# -----------------------------
def _hash_pw(pw: str, salt: str) -> str:
    """
    Salted SHA256 (lightweight; not for high-security production).
    """
    import hashlib

    return hashlib.sha256((salt + pw).encode("utf-8")).hexdigest()


def _hash_pw_legacy_unsalted(pw: str) -> str:
    """
    Legacy fallback for old users.json entries that stored unsalted sha256(password).
    """
    import hashlib

    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def _load_db(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_db(path: Path, db: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(db, indent=2), encoding="utf-8")


def ensure_admin_exists(root: Path) -> None:
    """
    Ensure there is at least one admin user record in users.json.
    Creates a default admin silently on first run if no admin exists.

    Default (silent):
      user_id: admin
      password: admin123  (7 chars, fits 4–8 rule)
    """
    path = users_file(root)
    db = _load_db(path)

    # If any admin exists, done.
    for rec in db.values():
        if isinstance(rec, dict) and rec.get("role") == ROLE_ADMIN:
            return

    # Create default admin (silent)
    user_id = "admin"
    pw = "admin123"  # ✅ 7 chars
    salt = secrets.token_hex(10)

    db[user_id] = {
        "role": ROLE_ADMIN,
        "salt": salt,
        "pw_hash": _hash_pw(pw, salt),
    }
    _save_db(path, db)


def authenticate(root: Path, user_id: str, password: str) -> Optional[AuthUser]:
    """
    Backward compatible:
    - New format: {role, salt, pw_hash}
    - Legacy format: {role, pw_hash} where pw_hash == sha256(password) (no salt)
    If legacy login succeeds, the record is migrated to salted format automatically.
    """
    u = normalize_user_id(user_id)
    pw = (password or "").strip()

    if not is_valid_user_id(u) or not is_valid_password(pw):
        return None

    path = users_file(root)
    db = _load_db(path)
    rec = db.get(u)

    if not isinstance(rec, dict):
        return None

    role = rec.get("role", "")
    if role not in {ROLE_ADMIN, ROLE_STAFF, ROLE_CLINICIAN}:
        return None

    pw_hash = rec.get("pw_hash", "")
    salt = rec.get("salt", "")

    # --- New salted path ---
    if salt and pw_hash:
        if _hash_pw(pw, salt) != pw_hash:
            return None
        return AuthUser(user_id=u, role=role)

    # --- Legacy unsalted path ---
    if pw_hash and not salt:
        if _hash_pw_legacy_unsalted(pw) != pw_hash:
            return None

        # ✅ migrate to salted after successful login
        new_salt = secrets.token_hex(10)
        rec["salt"] = new_salt
        rec["pw_hash"] = _hash_pw(pw, new_salt)
        db[u] = rec
        _save_db(path, db)

        return AuthUser(user_id=u, role=role)

    return None


def get_current_user() -> Optional[AuthUser]:
    u = st.session_state.get(SESSION_KEY)
    if isinstance(u, dict) and "user_id" in u and "role" in u:
        return AuthUser(user_id=u["user_id"], role=u["role"])
    if isinstance(u, AuthUser):
        return u
    return None


def set_current_user(user: AuthUser) -> None:
    st.session_state[SESSION_KEY] = {"user_id": user.user_id, "role": user.role}


def logout() -> None:
    st.session_state.pop(SESSION_KEY, None)


# -----------------------------
# Access control helpers
# -----------------------------
def can_access_registry(role: str) -> bool:
    return role in {ROLE_ADMIN, ROLE_STAFF}


def can_access_screening(role: str) -> bool:
    return role in {ROLE_ADMIN, ROLE_CLINICIAN}


def can_access_reports(role: str) -> bool:
    return role in {ROLE_ADMIN, ROLE_STAFF,ROLE_CLINICIAN}


# -----------------------------
# UI: login + admin user manager
# -----------------------------
def require_login(root: Path) -> AuthUser:
    """
    Simple login gate.
    - No signup UI
    - No "default admin created" message
    """
    user = get_current_user()
    if user:
        return user

    # Ensure an admin exists (silent)
    ensure_admin_exists(root)

    st.markdown("## 🔐 Login")
    st.caption("Sign in with your user ID and password (4–8 chars).")

    with st.form("login_form", clear_on_submit=False):
        uid = st.text_input("User ID", placeholder="e.g., admin / staff01 / dr_singh")
        pw = st.text_input("Password (4–8 chars)", type="password", max_chars=8)
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        u = authenticate(root, uid, pw)
        if u is None:
            st.error("Invalid credentials or format. (User ID 3–24, Password 4–8)")
        else:
            set_current_user(u)
            st.success("Signed in ✅")
            st.rerun()

    st.stop()


def sidebar_identity(user: AuthUser) -> None:
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**User:** `{user.user_id}`  \n**Role:** `{user.role}`")
        if st.button("Logout", use_container_width=True):
            logout()
            st.rerun()


def admin_create_user_panel(root: Path, current_user: AuthUser) -> None:
    """
    Admin-only panel to create new users (staff/clinician/admin).
    """
    if current_user.role != ROLE_ADMIN:
        return

    path = users_file(root)
    db = _load_db(path)

    with st.sidebar.expander("👑 Admin: User management", expanded=False):
        st.caption("Create users for staff/clinicians. Password must be 4–8 characters.")

        with st.form("create_user_form"):
            new_uid = st.text_input("New user ID", placeholder="e.g., staff02 / dr_patel")
            new_role = st.selectbox("Role", options=[ROLE_STAFF, ROLE_CLINICIAN, ROLE_ADMIN])
            new_pw = st.text_input("Password (4–8)", type="password", max_chars=8)
            submit = st.form_submit_button("Create user", use_container_width=True)

        if submit:
            uid = normalize_user_id(new_uid)
            if not is_valid_user_id(uid):
                st.error("User ID must be 3–24 chars: letters/digits/._- only.")
            elif uid in db:
                st.error("User ID already exists.")
            elif not is_valid_password(new_pw):
                st.error("Password length must be 4–8 characters.")
            else:
                salt = secrets.token_hex(10)
                db[uid] = {"role": new_role, "salt": salt, "pw_hash": _hash_pw(new_pw.strip(), salt)}
                _save_db(path, db)
                st.success(f"Created user `{uid}` with role `{new_role}` ✅")

        st.markdown("**Existing users**")
        rows = [{"user_id": k, "role": v.get("role", "")} for k, v in sorted(db.items())]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No users found.")
