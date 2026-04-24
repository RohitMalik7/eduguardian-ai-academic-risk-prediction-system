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
#
# FILE          : predict.py
# PURPOSE       : Load the trained ML model and run a full hybrid prediction
#                 for a single student. Combines:
#
#   Signal 1 - ML Model Prediction (trained Random Forest from train_model.py)
#     Loads the saved risk_model.joblib, scales the input features using the
#     saved scaler.joblib, and returns a probability score + ML risk label.
#
#   Signal 2 - Deterministic Risk Engine (risk_engine.py)
#     Runs the quiz trend analyser + assignment risk scorer and returns a
#     combined deterministic risk score and risk level.
#
#   Signal 3 - Hybrid Final Decision
#     Fuses both signals using a weighted average.
#     Default configuration:
#       ML probability  × 0.55
#       Engine score    × 0.45
#     These values can be overridden at runtime by staff through the portal.
#     Produces a single final risk level: LOW / MEDIUM / HIGH
#     and a structured output ready for the Flask app to display.
#
# WHY HYBRID FUSION:
#   Neither signal alone is sufficient:
#   - ML model alone: black-box, cannot explain WHY a student is flagged,
#     and requires complete feature data (all 5 features must be present).
#   - Risk engine alone: deterministic, no learning from historical patterns,
#     cannot detect subtle combinations that only appear in training data.
#   - Hybrid: ML provides statistical pattern matching across all features.
#     Engine provides interpretable, real-time trend reasoning. Together they
#     produce a more robust, explainable, and reliable prediction.
#
# INPUT         : models/risk_model.joblib   (trained ML model package)
#                 models/scaler.joblib       (fitted StandardScaler)
#                 Student data: quiz scores, assignment scores, midterm score,
#                 participation score, projects score.
#
# OUTPUT        : dict containing:
#                   ml_probability    - ML model at-risk probability (0.0–1.0)
#                   ml_risk_label     - "At-Risk" or "Not At-Risk" from ML
#                   engine_score      - Deterministic combined score (0.0–1.0)
#                   engine_risk_level - "LOW", "MEDIUM", or "HIGH" from engine
#                   final_risk_level  - "LOW", "MEDIUM", or "HIGH" (hybrid)
#                   final_score       - Fused hybrid score (0.0–1.0)
#                   confidence        - Confidence label (High / Moderate / Low)
#                   alert             - Staff alert dict from risk_engine.py
#                   feature_values    - Scaled feature values (for transparency)
#                   report_path       - Path to saved JSON audit report
# =============================================================================

import os
import sys
import warnings
import numpy as np
import joblib

warnings.filterwarnings("ignore")

# Add src directory to path so risk_engine can be imported when predict.py
# is called from outside the src/ folder (e.g. from the Flask app)
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from risk_engine import (run_risk_assessment, DEFAULT_UNIT_CONFIG,
                         generate_alert, save_report as engine_save_report)

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_FILE  = os.path.join(MODELS_DIR, "risk_model.joblib")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.joblib")

# =============================================================================
# DEFAULT HYBRID CONFIGURATION
# These values define the validated default hybrid behaviour of the system.
#
# Default settings:
#   - ML weight         : 0.55
#   - Engine weight     : 0.45
#   - Medium threshold  : 0.35
#   - High threshold    : 0.55
#
# Rationale:
#   The ML signal is weighted slightly higher because it was trained on a
#   large multi-dataset training pool and captures subtle multivariate
#   patterns. The deterministic engine remains heavily weighted so the
#   final decision always retains interpretability and trend-based logic.
#
# Runtime behaviour:
#   These values are the system defaults only. Staff can override them
#   for a specific run through the Flask portal without changing the
#   validated baseline configuration stored in code.
# =============================================================================

DEFAULT_HYBRID_CONFIG = {
    "ml_weight": 0.55,
    "engine_weight": 0.45,
    "medium_threshold": 0.35,
    "high_threshold": 0.55,
}

# Backward-compatible aliases retained for test compatibility
ML_WEIGHT = DEFAULT_HYBRID_CONFIG["ml_weight"]
ENGINE_WEIGHT = DEFAULT_HYBRID_CONFIG["engine_weight"]
FINAL_MEDIUM_THRESHOLD = DEFAULT_HYBRID_CONFIG["medium_threshold"]
FINAL_HIGH_THRESHOLD = DEFAULT_HYBRID_CONFIG["high_threshold"]

# Feature order must match FINAL_FEATURES in train_model.py exactly
FINAL_FEATURES = [
    "Quizzes_Avg",
    "Assignments_Avg",
    "Midterm_Score",
    "Participation_Score",
    "Projects_Score",
]

def resolve_hybrid_config(config=None):
    """
    Resolve runtime hybrid configuration.

    If no config is provided, the validated default configuration is used.
    If a config is provided, it overrides the defaults for that prediction run.

    Validation rules:
      - ml_weight and engine_weight must each be in [0, 1]
      - ml_weight + engine_weight must equal 1.0
      - medium_threshold and high_threshold must each be in (0, 1)
      - medium_threshold must be lower than high_threshold
    """
    cfg = dict(DEFAULT_HYBRID_CONFIG)

    if config:
        cfg["ml_weight"] = float(config.get("ml_weight", cfg["ml_weight"]))
        cfg["engine_weight"] = float(config.get("engine_weight", cfg["engine_weight"]))
        cfg["medium_threshold"] = float(config.get("medium_threshold", cfg["medium_threshold"]))
        cfg["high_threshold"] = float(config.get("high_threshold", cfg["high_threshold"]))

    if not (0.0 <= cfg["ml_weight"] <= 1.0):
        raise ValueError("ml_weight must be between 0 and 1.")

    if not (0.0 <= cfg["engine_weight"] <= 1.0):
        raise ValueError("engine_weight must be between 0 and 1.")

    if abs((cfg["ml_weight"] + cfg["engine_weight"]) - 1.0) > 1e-6:
        raise ValueError("ml_weight and engine_weight must sum to 1.0.")

    if not (0.0 < cfg["medium_threshold"] < 1.0):
        raise ValueError("medium_threshold must be between 0 and 1.")

    if not (0.0 < cfg["high_threshold"] < 1.0):
        raise ValueError("high_threshold must be between 0 and 1.")

    if cfg["medium_threshold"] >= cfg["high_threshold"]:
        raise ValueError("medium_threshold must be less than high_threshold.")

    return cfg

# =============================================================================
# MODEL LOADER
# Loads the saved model package and scaler from disk.
# Uses a simple module-level cache so the model is only loaded once per
# session - prevents repeated disk reads when predicting multiple students
# in the Flask app.
# =============================================================================

_model_cache  = None
_scaler_cache = None


def load_model():
    """
    Load the trained ML model package from disk.
    Cached after first load - safe to call multiple times.

    Returns:
        dict: Model package with keys: model, model_name, features, threshold.

    Raises:
        FileNotFoundError: If risk_model.joblib does not exist.
                           Run train_model.py first to generate it.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "Model file not found: " + MODEL_FILE + "\n"
            "Run train_model.py first to train and save the model."
        )

    _model_cache = joblib.load(MODEL_FILE)
    return _model_cache


def load_scaler():
    """
    Load the fitted StandardScaler from disk.
    Cached after first load.

    Returns:
        StandardScaler: Fitted scaler from preprocess.py.

    Raises:
        FileNotFoundError: If scaler.joblib does not exist.
                           Run preprocess.py first to generate it.
    """
    global _scaler_cache
    if _scaler_cache is not None:
        return _scaler_cache

    if not os.path.exists(SCALER_FILE):
        raise FileNotFoundError(
            "Scaler file not found: " + SCALER_FILE + "\n"
            "Run preprocess.py first to fit and save the scaler."
        )

    _scaler_cache = joblib.load(SCALER_FILE)
    return _scaler_cache


# =============================================================================
# FEATURE BUILDER
# Converts raw student input into the 5-feature vector expected by the ML model.
# Handles partial data gracefully - missing features are filled with 50.0
# (neutral midpoint) rather than crashing.
#
# FEATURE MAPPING:
#   Quizzes_Avg       <- projected quiz average from risk_engine trend analysis
#   Assignments_Avg   <- average of submitted assignment scores (0-100), or 50.0
#   Midterm_Score     <- midterm score directly entered by staff, or 50.0
#   Participation_Score <- participation score (normalised to 0-100), or 50.0
#   Projects_Score    <- project score directly entered by staff, or 50.0
#
# WHY USE PROJECTED QUIZ AVG FOR ML FEATURE:
#   The ML model was trained on full-trimester quiz averages. Mid-semester,
#   we only have partial data. Using the risk engine's projected average
#   (which extrapolates the trend line) gives the ML model the best possible
#   estimate of what the final quiz average will be - making predictions
#   consistent with how the model was trained.
# =============================================================================

def build_feature_vector(trend_result, a1=None, a2=None,
                         midterm=None, participation=None, projects=None):
    """
    Build the 5-feature input vector for the ML model.

    Args:
        trend_result  (dict):       Output from risk_engine.analyse_quiz_trend().
        a1            (float|None): Assignment 1 score (0-100).
        a2            (float|None): Assignment 2 score (0-100).
        midterm       (float|None): Midterm score (0-100), or None if not sat.
        participation (float|None): Participation score (0-10 scale), or None.
        projects      (float|None): Project score (0-100), or None.

    Returns:
        np.ndarray: Shape (1, 5) - ready to be passed to scaler.transform().
        dict: Human-readable feature values before scaling.
    """
    # Use projected quiz average from engine - best estimate for ML input
    quizzes_avg = trend_result["projected_avg_pct"]

    # Assignment average: average of available scores, or neutral 50.0
    submitted = [s for s in [a1, a2] if s is not None]
    assignments_avg = float(np.mean(submitted)) if submitted else 50.0

    # Midterm: use directly, or neutral 50.0 if not yet sat
    midterm_score = float(midterm) if midterm is not None else 50.0

    # Participation: normalise from 0-10 to 0-100, or neutral 50.0
    if participation is not None:
        participation_score = float(np.clip(participation, 0, 10)) * 10.0
    else:
        participation_score = 50.0

    # Projects: use directly, or neutral 50.0
    projects_score = float(projects) if projects is not None else 50.0

    feature_values = {
        "Quizzes_Avg":         round(quizzes_avg,         2),
        "Assignments_Avg":     round(assignments_avg,     2),
        "Midterm_Score":       round(midterm_score,       2),
        "Participation_Score": round(participation_score, 2),
        "Projects_Score":      round(projects_score,      2),
    }

    vector = np.array([[
        quizzes_avg,
        assignments_avg,
        midterm_score,
        participation_score,
        projects_score,
    ]], dtype=float)

    return vector, feature_values


# =============================================================================
# ML PREDICTION
# Takes the raw feature vector, applies the saved scaler, and runs prediction
# using the loaded model. Returns the at-risk probability and binary label.
# =============================================================================

def ml_predict(feature_vector):
    """
    Run the ML model on a pre-built feature vector.

    Args:
        feature_vector (np.ndarray): Shape (1, 5) - raw unscaled features.

    Returns:
        tuple:
            ml_probability (float): Probability of being at-risk (0.0–1.0).
            ml_label       (str):   "At-Risk" or "Not At-Risk".
            model_name     (str):   Name of the model used (e.g. "Random Forest").
            threshold      (float): Threshold used for classification.
    """
    package   = load_model()
    scaler    = load_scaler()
    model     = package["model"]
    threshold = package["threshold"]
    model_name = package["model_name"]

    # Scale the input using the scaler fitted during preprocessing
    scaled = scaler.transform(feature_vector)

    # Get probability of at-risk class (index 1)
    ml_prob = float(model.predict_proba(scaled)[0][1])

    # Apply the custom threshold (0.45) to favour recall
    ml_label = "At-Risk" if ml_prob >= threshold else "Not At-Risk"

    return ml_prob, ml_label, model_name, threshold


# =============================================================================
# HYBRID FUSION
# Combines the ML probability and deterministic engine score into one final
# risk level. Both signals are normalised to [0, 1] before blending.
#
# FUSION FORMULA:
#   final_score = (ml_probability × ml_weight) + (engine_score × engine_weight)
#
# Default configuration:
#   ml_weight     = 0.55
#   engine_weight = 0.45
#
# These values may be overridden at runtime through hybrid_config.
#
# AGREEMENT / DISAGREEMENT LOGIC:
#   If both signals agree (both HIGH or both LOW) -> high confidence.
#   If signals disagree (ML says At-Risk but engine says LOW, or vice versa)
#   -> moderate confidence. The hybrid score resolves the tie numerically.
#   This situation is explicitly flagged in the output for staff awareness.
# =============================================================================

def fuse_signals(ml_probability, engine_score, hybrid_config=None):
    """
    Fuse ML probability and engine score into a final hybrid risk level.

    Args:
        ml_probability (float): ML at-risk probability (0.0–1.0).
        engine_score   (float): Deterministic engine combined score (0.0–1.0).
        hybrid_config  (dict|None): Runtime hybrid configuration.

    Returns:
        tuple:
            final_score      (float): Fused score (0.0–1.0).
            final_risk_level (str):   "LOW", "MEDIUM", or "HIGH".
            confidence       (str):   "High", "Moderate", or "Low".
            agreement        (str):   "Agree" or "Disagree".
    """
    cfg = resolve_hybrid_config(hybrid_config)

    final_score = round(
        (ml_probability * cfg["ml_weight"]) +
        (engine_score   * cfg["engine_weight"]),
        4
    )

    if final_score >= cfg["high_threshold"]:
        final_risk_level = "HIGH"
    elif final_score >= cfg["medium_threshold"]:
        final_risk_level = "MEDIUM"
    else:
        final_risk_level = "LOW"

    ml_high     = ml_probability >= 0.5
    engine_high = engine_score   >= 0.5

    if ml_high == engine_high:
        agreement  = "Agree"
        confidence = "High"
    else:
        agreement  = "Disagree"
        gap = abs(ml_probability - engine_score)
        confidence = "Moderate" if gap < 0.3 else "Low"

    return final_score, final_risk_level, confidence, agreement


# =============================================================================
# CONFIDENCE LABEL BUILDER
# Converts numerical confidence into a descriptive label for staff display.
# Adds context about what the confidence level means in plain language.
# =============================================================================

def build_confidence_message(confidence, agreement, ml_prob, engine_score):
    """Build a plain-language confidence explanation for the staff interface."""
    if confidence == "High" and agreement == "Agree":
        return ("High confidence - both ML model and risk engine agree on "
                "this assessment (ML: " + str(round(ml_prob*100, 1)) +
                "% | Engine: " + str(round(engine_score*100, 1)) + "%)")
    elif confidence == "Moderate":
        return ("Moderate confidence - ML model and risk engine give "
                "different signals. Review manually. "
                "(ML: " + str(round(ml_prob*100, 1)) +
                "% | Engine: " + str(round(engine_score*100, 1)) + "%)")
    else:
        return ("Low confidence - signals strongly disagree. "
                "Additional data recommended before acting. "
                "(ML: " + str(round(ml_prob*100, 1)) +
                "% | Engine: " + str(round(engine_score*100, 1)) + "%)")


# =============================================================================
# MAIN PREDICTION FUNCTION
# This is the single entry point called by the Flask app and any other
# consumer of this module. Runs the full hybrid prediction pipeline for one
# student and returns a structured result dictionary.
# =============================================================================

def predict_student_risk(student_id, unit_name, quiz_scores,
                         a1=None, a2=None, midterm=None,
                         participation=None, projects=None,
                         student_name="Student", unit_config=None,
                         save_report=True, hybrid_config=None):
    """
    Run the full hybrid risk prediction for a single student.

    Args:
        student_id    (str):        Student ID (e.g. "S1042").
        unit_name     (str):        Unit name/code (e.g. "ICT304").
        quiz_scores   (list):       List of quiz scores entered so far.
        a1            (float|None): Assignment 1 score (0-100), or None.
        a2            (float|None): Assignment 2 score (0-100), or None.
        midterm       (float|None): Midterm exam score (0-100), or None.
        participation (float|None): Class participation score (0-10), or None.
        projects      (float|None): Project/coursework score (0-100), or None.
        student_name  (str):        Display name for alert generation.
        unit_config   (dict|None):  Unit configuration. Uses default if None.
        save_report   (bool):       Whether to save a JSON audit report.

    Returns:
        dict: Full hybrid prediction result. Keys:
            student_id, student_name, unit_name,
            ml_probability, ml_label, ml_model_name, ml_threshold,
            engine_score, engine_risk_level,
            final_score, final_risk_level, confidence, agreement,
            confidence_message, feature_values,
            trend (full quiz trend dict), assignment (full assign dict),
            alert (staff alert dict), report_path.
    """
    if unit_config is None:
        unit_config = DEFAULT_UNIT_CONFIG

    # --- Step 1: Run the deterministic risk engine ---
    engine_result = run_risk_assessment(
        student_id   = student_id,
        unit_name    = unit_name,
        quiz_scores  = quiz_scores,
        a1           = a1,
        a2           = a2,
        unit_config  = unit_config,
        student_name = student_name,
        save_json    = False    # We save one unified report below instead
    )

    trend_result  = engine_result["trend"]
    assign_result = engine_result["assignment"]
    engine_score  = engine_result["combined_score"]

    # --- Step 2: Build feature vector for ML model ---
    feature_vector, feature_values = build_feature_vector(
        trend_result  = trend_result,
        a1            = a1,
        a2            = a2,
        midterm       = midterm,
        participation = participation,
        projects      = projects,
    )

    # --- Step 3: ML model prediction ---
    try:
        ml_prob, ml_label, model_name, threshold = ml_predict(feature_vector)
        ml_available = True
    except FileNotFoundError as e:
        # Model not yet trained - degrade gracefully, engine signal only
        print("  [WARNING] " + str(e))
        print("  Running in engine-only mode (no ML signal).")
        ml_prob      = engine_score     # Fall back to engine score as ML proxy
        ml_label     = engine_result["risk_level"]
        model_name   = "Not available (run train_model.py)"
        threshold    = 0.45
        ml_available = False

    # --- Step 4: Hybrid signal fusion ---
    final_score, final_risk_level, confidence, agreement = fuse_signals(
        ml_prob, engine_score, hybrid_config=hybrid_config
    )

    # --- Step 5: Build confidence message for staff ---
    confidence_msg = build_confidence_message(confidence, agreement, ml_prob, engine_score)

    # --- Step 6: Re-generate alert using final hybrid risk level ---
    # Use the hybrid final level so alert matches the displayed risk
    alert = generate_alert(
        final_risk_level, trend_result, assign_result, student_name
    )

    # --- Step 7: Build the ML secondary signal dict for the audit report ---
    ml_result_dict = {
        "ml_available":   ml_available,
        "model_name":     model_name,
        "ml_probability": round(ml_prob, 4),
        "ml_label":       ml_label,
        "threshold":      threshold,
        "feature_values": feature_values,
    }

    # --- Step 8: Save unified JSON audit report ---
    report_path = None
    if save_report:
        report_path = engine_save_report(
            student_id     = student_id,
            unit_name      = unit_name,
            trend_result   = trend_result,
            assign_result  = assign_result,
            combined_score = final_score,
            risk_level     = final_risk_level,
            alert          = alert,
            ml_result      = ml_result_dict,
            unit_config    = unit_config,
        )

    return {
        "student_id":         student_id,
        "student_name":       student_name,
        "unit_name":          unit_name,
        # ML signal
        "ml_probability":     round(ml_prob, 4),
        "ml_label":           ml_label,
        "ml_model_name":      model_name,
        "ml_threshold":       threshold,
        "ml_available":       ml_available,
        # Engine signal
        "engine_score":       engine_score,
        "engine_risk_level":  engine_result["risk_level"],
        # Hybrid final decision
        "final_score":        final_score,
        "final_risk_level":   final_risk_level,
        "confidence":         confidence,
        "agreement":          agreement,
        "confidence_message": confidence_msg,
        "hybrid_config_used": resolve_hybrid_config(hybrid_config),
        # Detailed sub-results
        "feature_values":     feature_values,
        "trend":              trend_result,
        "assignment":         assign_result,
        "alert":              alert,
        "report_path":        report_path,
    }


# =============================================================================
# QUICK SELF-TEST - runs when file is executed directly
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ICT304 - predict.py Self-Test")
    print("="*70)

    tests = [
        {
            "label":       "HIGH risk - declining quizzes, failed assignment",
            "student_id":  "TEST001",
            "unit_name":   "ICT304",
            "quiz_scores": [18, 15, 12, 9, 7],
            "a1":          38.0,
            "a2":          None,
            "midterm":     42.0,
            "participation": 3.0,
            "projects":    45.0,
            "name":        "Test Student A",
        },
        {
            "label":       "LOW risk - improving quizzes, good assignments",
            "student_id":  "TEST002",
            "unit_name":   "ICT304",
            "quiz_scores": [12, 14, 16, 17, 18],
            "a1":          75.0,
            "a2":          80.0,
            "midterm":     72.0,
            "participation": 8.0,
            "projects":    78.0,
            "name":        "Test Student B",
        },
        {
            "label":       "MEDIUM risk - stable low scores, no assignments",
            "student_id":  "TEST003",
            "unit_name":   "ICT304",
            "quiz_scores": [10, 9, 10, 11, 10],
            "a1":          None,
            "a2":          None,
            "midterm":     None,
            "participation": None,
            "projects":    None,
            "name":        "Test Student C",
        },
    ]

    for t in tests:
        print("\n--- " + t["label"] + " ---")
        result = predict_student_risk(
            student_id    = t["student_id"],
            unit_name     = t["unit_name"],
            quiz_scores   = t["quiz_scores"],
            a1            = t["a1"],
            a2            = t["a2"],
            midterm       = t["midterm"],
            participation = t["participation"],
            projects      = t["projects"],
            student_name  = t["name"],
            save_report   = False,
        )
        print("  ML Probability  : " + str(result["ml_probability"]) +
              " (" + result["ml_label"] + ")")
        print("  Engine Score    : " + str(result["engine_score"]) +
              " (" + result["engine_risk_level"] + ")")
        print("  Final Score     : " + str(result["final_score"]))
        print("  FINAL RISK      : " + result["final_risk_level"])
        print("  Confidence      : " + result["confidence"] +
              " (" + result["agreement"] + ")")
        print("  Message         : " + result["confidence_message"])
        print("  Reasons         : " + str(result["alert"]["reasons"]))

    print("\n  [DONE] Self-test complete.")
    print("="*70 + "\n")
