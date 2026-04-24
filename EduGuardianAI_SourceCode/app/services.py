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
# app/services.py
# ICT304 - EduGuardian AI - Academic Risk Prediction System
# Service layer: bridges Flask routes with the AI pipeline (src/).
# =============================================================================

import os
import sys
import subprocess

from database.models import (
    get_students_in_unit,
    get_unit_by_code,
    get_unit_config,
    get_quiz_scores_list,
    get_assignment_scores,
    save_risk_result,
    get_full_cohort_progress,
    get_risk_summary_for_unit,
)

# -- path setup --------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.predict import predict_student_risk


# =============================================================================
# PIPELINE HELPERS
# =============================================================================

def _model_exists() -> bool:
    model_path = os.path.join(BASE_DIR, "models", "risk_model.joblib")
    return os.path.exists(model_path)


def _data_processed() -> bool:
    data_path = os.path.join(BASE_DIR, "data", "processed", "ds1_train_smote.csv")
    return os.path.exists(data_path)


def _run_preprocessing() -> dict:
    """
    Run preprocess.py as a subprocess.
    Returns {"success": bool, "output": str, "error": str}
    """
    script = os.path.join(SRC_DIR, "preprocess.py")
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=300,
            cwd=BASE_DIR
        )
        return {
            "success": result.returncode == 0,
            "output":  result.stdout,
            "error":   result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "Preprocessing timed out (>5 min)."}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def _run_training() -> dict:
    """
    Run train_model.py as a subprocess.
    Returns {"success": bool, "output": str, "error": str}
    """
    script = os.path.join(SRC_DIR, "train_model.py")
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=600,
            cwd=BASE_DIR
        )
        return {
            "success": result.returncode == 0,
            "output":  result.stdout,
            "error":   result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "Training timed out (>10 min)."}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


# =============================================================================
# UNIT CONFIG CONVERTER
# =============================================================================

def build_unit_config_from_db(cfg_row) -> dict | None:
    """Convert a DB unit_config row to the format expected by predict.py."""
    if not cfg_row:
        return None
    return {
        "quiz_count":        int(cfg_row["quiz_count"]),
        "quiz_max_marks":    float(cfg_row["quiz_max_marks"]),
        "quiz_weight":       float(cfg_row["quiz_weight_pct"]),
        "assignment_count":  2,
        "assignment_max":    100,
        "assignment_weight": float(cfg_row["a1_weight_pct"]) + float(cfg_row["a2_weight_pct"]),
        "has_exam":          True,
        "exam_max":          100,
        "exam_weight":       float(cfg_row["final_weight_pct"]),
        "pass_mark":         float(cfg_row["pass_mark_pct"]),
    }


# =============================================================================
# SINGLE STUDENT PREDICTION
# =============================================================================

def run_prediction_for_one_student(user_id, unit_id, student_id,
                                   student_name, unit_code,
                                   hybrid_config=None) -> dict:
    """Run the full hybrid prediction for a single student and save to DB."""
    cfg_row     = get_unit_config(unit_id)
    unit_config = build_unit_config_from_db(cfg_row)
    quiz_scores = get_quiz_scores_list(user_id, unit_id)
    assignments = get_assignment_scores(user_id, unit_id)

    result = predict_student_risk(
        student_id    = student_id,
        unit_name     = unit_code,
        quiz_scores   = quiz_scores,
        a1            = assignments.get(1),
        a2            = assignments.get(2),
        midterm       = None,
        participation = None,
        projects      = None,
        student_name  = student_name,
        unit_config   = unit_config,
        save_report   = False,  # PDF/JSON reports are exported on demand, not during every model run
        hybrid_config = hybrid_config,
    )

    # Persist risk result to DB so student portal can read it
    cfg_used = result.get("hybrid_config_used", {}) or {}

    save_risk_result(
        user_id                = user_id,
        unit_id                = unit_id,
        quiz_avg_pct           = result["trend"]["current_avg_pct"],
        projected_avg_pct      = result["trend"]["projected_avg_pct"],
        trend_direction        = result["trend"]["trend"],
        trend_slope            = result["trend"]["slope"],
        a1_score               = assignments.get(1),
        a2_score               = assignments.get(2),
        combined_risk_score    = result["final_score"],
        risk_level             = result["final_risk_level"],
        ml_probability         = result["ml_probability"],
        selected_model         = result["ml_model_name"],
        marks_earned           = 0.0,
        marks_available        = 0.0,
        ml_weight_used         = cfg_used.get("ml_weight", 0.55),
        engine_weight_used     = cfg_used.get("engine_weight", 0.45),
        medium_threshold_used  = cfg_used.get("medium_threshold", 0.35),
        high_threshold_used    = cfg_used.get("high_threshold", 0.55),
    )
    return result


# =============================================================================
# FULL UNIT PREDICTION - main entry point for /api/predict
# =============================================================================

def run_prediction_for_unit(unit_code: str, hybrid_config=None) -> dict:
    """
    Run the full AI pipeline for all students in a unit.

    Flow:
      1. Check if model exists -> if not, run preprocessing + training first.
      2. Predict risk for every enrolled student.
      3. Save results to DB.
      4. Return summary + per-student results.

    Raises:
        ValueError: If unit not found.
        RuntimeError: If preprocessing or training fails.
    """
    unit = get_unit_by_code(unit_code)
    if not unit:
        raise ValueError(f"Unit not found: {unit_code}")

    # -- Step 1: Ensure model is ready ---------------------------------------
    pipeline_log = []

    if not _data_processed():
        pipeline_log.append("Starting data preprocessing...")
        prep_result = _run_preprocessing()
        if not prep_result["success"]:
            raise RuntimeError(
                "Preprocessing failed:\n" + (prep_result["error"] or "Unknown error")
            )
        pipeline_log.append("Preprocessing complete.")

    if not _model_exists():
        pipeline_log.append("Training AI model (this may take a few minutes on first run)...")
        train_result = _run_training()
        if not train_result["success"]:
            raise RuntimeError(
                "Model training failed:\n" + (train_result["error"] or "Unknown error")
            )
        pipeline_log.append("Model training complete.")

    # -- Step 2: Run predictions for all students -----------------------------
    unit_id  = unit["unit_id"]
    students = get_students_in_unit(unit_id)

    results = []
    errors  = []
    for s in students:
        try:
            result = run_prediction_for_one_student(
                user_id       = s["user_id"],
                unit_id       = unit_id,
                student_id    = s["student_id"],
                student_name  = s["full_name"],
                unit_code     = unit_code,
                hybrid_config = hybrid_config,
            )
            results.append({
                "user_id":            s["user_id"],
                "student_id":         s["student_id"],
                "full_name":          s["full_name"],
                "risk_level":         result["final_risk_level"],
                "final_score":        result["final_score"],
                "ml_probability":     result["ml_probability"],
                "confidence":         result["confidence"],
                "agreement":          result["agreement"],
                "confidence_message": result["confidence_message"],
                "alert":              result["alert"],
                "trend":              result["trend"],
            })
        except Exception as e:
            errors.append({
                "student_id": s.get("student_id", "?"),
                "error":      str(e),
            })

    summary = get_risk_summary_for_unit(unit_id)

    return {
        "unit_id":      unit_id,
        "unit_code":    unit_code,
        "summary":      summary,
        "results":      results,
        "errors":       errors,
        "pipeline_log": pipeline_log,
        "hybrid_config_used": hybrid_config,
    }


def run_prediction_for_unit_with_reports(unit_code: str, hybrid_config=None) -> dict:
    """
    Re-run predictions WITH save_report=True so JSON audit files
    are written to reports/. Called only when staff explicitly
    clicks Export JSON - not during normal model run.
    """
    unit = get_unit_by_code(unit_code)
    if not unit:
        raise ValueError(f"Unit not found: {unit_code}")

    unit_id  = unit["unit_id"]
    students = get_students_in_unit(unit_id)
    saved    = 0

    for s in students:
        try:
            cfg_row     = get_unit_config(unit_id)
            unit_config = build_unit_config_from_db(cfg_row)
            quiz_scores = get_quiz_scores_list(s["user_id"], unit_id)
            assignments = get_assignment_scores(s["user_id"], unit_id)

            predict_student_risk(
                student_id    = s["student_id"],
                unit_name     = unit_code,
                quiz_scores   = quiz_scores,
                a1            = assignments.get(1),
                a2            = assignments.get(2),
                student_name  = s["full_name"],
                unit_config   = unit_config,
                save_report   = True,   # Only here do we save to disk
                hybrid_config = hybrid_config,
            )
            saved += 1
        except Exception:
            pass

    return {"saved": saved}