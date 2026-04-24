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
# app/utils.py
# ICT304 - EduGuardian AI - Session-based Authentication
# =============================================================================

"""
app/utils.py - Shared helper functions for the EduRisk Flask app.

Used by routes.py and templates. In the backend phase, chart helpers
that need real data will call database/models.py or src/predict.py.
"""


def format_risk_badge(risk_label: str) -> str:
    if not risk_label:
        return "badge-amber"

    label = str(risk_label).strip().upper()

    mapping = {
        "HIGH": "badge-red",
        "MEDIUM": "badge-amber",
        "LOW": "badge-green",
        "UNKNOWN": "badge-amber",
    }
    return mapping.get(label, "badge-amber")


def calculate_weighted_total(scores: dict, assessments: list) -> float | None:
    """
    Calculate a student's weighted total score.

    Args:
        scores:      {"Assignment 1": 75.0, "Quiz": 80.0, ...}
        assessments: [{"name": "Assignment 1", "weight": 20}, ...]

    Returns:
        Weighted total (0–100, rounded to 1 dp), or None if any score missing.

    Example:
        scores      = {"Assignment 1": 80, "Final Exam": 70}
        assessments = [{"name": "Assignment 1", "weight": 20},
                       {"name": "Final Exam",   "weight": 50}]
        -> (80/100 * 20) + (70/100 * 50) = 16 + 35 = 51.0
    """
    total = 0.0
    for a in assessments:
        val = scores.get(a["name"])
        if val is None:
            return None
        total += (float(val) / 100.0) * float(a["weight"])
    return round(total, 1)


def classify_risk(weighted_total: float | None) -> str:
    """
    Rule-based risk classification used before ML model is integrated.

    Args:
        weighted_total: score from calculate_weighted_total()

    Returns:
        "High", "Med", or "Low"

    Thresholds (adjust after model evaluation):
        < 40  -> High risk
        40–60 -> Medium risk
        > 60  -> Low risk
    """
    if weighted_total is None:
        return "Unknown"
    if weighted_total < 40:
        return "High"
    if weighted_total < 60:
        return "Med"
    return "Low"


def register_jinja_filters(app) -> None:
    """
    Register custom Jinja2 filters on the Flask app.
    Called inside create_app() in app/__init__.py.

    Template usage:
        {{ student.risk | risk_badge }}
    """
    app.jinja_env.filters["risk_badge"] = format_risk_badge
