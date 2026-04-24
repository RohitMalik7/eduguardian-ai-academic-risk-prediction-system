# =============================================================================
# ICT304 - AI System Development
# Murdoch University - School of Information Technology
# =============================================================================
# PROJECT TITLE : AI Academic Risk Prediction & Early Intervention System
# SYSTEM NAME   : Early Academic Risk Prediction Engine
# UNIT CODE     : ICT304
# ASSIGNMENT    : Final Project (Assignment 2) - Full AI System
#                 Built upon Assignment 1 Prototype (submitted 20 Feb 2026)
# TEAM MEMBERS  : Rohit Kumar Malik
#                 Izaan Shumaiz
#                 Mohamed Sinan
# SUPERVISOR    : Mr. Karim Tout
# DUE DATE      : 3 April 2026
# =============================================================================
# app/auth.py
# ICT304 - EduGuardian AI - Session-based Authentication
# =============================================================================

from flask import session, redirect, url_for
from functools import wraps
from database.models import validate_login


def login_user(email: str, password: str, role: str) -> bool:
    """
    Validate credentials against the database and populate the Flask session.
    Stores student_id so templates and reports can display the academic identifier directly.
    """
    user = validate_login(email, password)
    if not user:
        return False
    if user["role"] != role:
        return False

    session["user_id"]     = user["user_id"]
    session["user_name"]   = user["full_name"]
    session["user_role"]   = user["role"]
    session["user_email"]  = user["email"]
    session["student_id"]  = user["student_id"]   # e.g. "S1000" or None for staff
    return True


def logout_user() -> None:
    session.clear()


def current_user() -> dict | None:
    """Return the current logged-in user dict from session, or None."""
    if "user_id" not in session:
        return None
    return {
        "id":         session["user_id"],
        "name":       session.get("user_name"),
        "role":       session.get("user_role"),
        "email":      session.get("user_email"),
        "student_id": session.get("student_id"),   # available in templates as user.student_id
    }


def login_required(role: str = None):
    """Decorator that enforces login and optionally enforces a specific role."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            user = current_user()
            if not user:
                return redirect(url_for("main.login"))
            if role and user["role"] != role:
                return redirect(url_for("main.login"))
            return f(*args, **kwargs)
        return wrapped
    return decorator