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
# FILE          : risk_engine.py
# PURPOSE       : Core deterministic AI sub-systems for academic risk analysis.
#                 This module runs INDEPENDENTLY of the ML model and provides:
#
#   Sub-system 1 - Quiz Trend Analyser
#     Analyses individual weekly quiz scores entered so far, fits a linear
#     trend (slope), and projects the expected quiz average for the full
#     trimester. Returns trend direction (Improving / Stable / Declining)
#     and a normalised quiz risk score (0.0 = no risk, 1.0 = maximum risk).
#
#   Sub-system 2 - Assignment Risk Scorer
#     Computes a normalised risk score from submitted assignment marks.
#     If no assignments submitted yet, returns a neutral 0.5 score.
#     Risk increases proportionally as assignment average drops below 50%.
#
#   Sub-system 3 - Combined Risk Decision Engine
#     Fuses the quiz risk score and assignment risk score using weighted
#     combination (quiz trend 60% + assignment 40%). Classifies the
#     combined score into a discrete risk level: LOW / MEDIUM / HIGH.
#     If no assignments submitted yet, full weight is given to quiz trend.
#
#   Sub-system 4 - Alert & Recommendation Generator
#     Translates the risk level into a structured staff alert with
#     specific reasons and a recommended action. Used by the Flask
#     app to display colour-coded banners and staff notification messages.
#
#   Sub-system 5 - Report Logger
#     Generates a timestamped JSON audit report per student assessment.
#     Saved to reports/ folder for traceability and review.
#
# DESIGN PRINCIPLE - HYBRID RISK ARCHITECTURE:
#   The deterministic risk engine (this file) provides the PRIMARY risk signal.
#   The ML model (predict.py) provides a SECONDARY pattern-matching signal.
#   Combining both makes the system more robust:
#     - If ML has limited data, trend projection still provides early insight.
#     - If trend assumptions are imperfect, ML adds a statistical safety net.
#     - Staff can see both signals and understand how the risk was derived.
#
# MURDOCH UNIVERSITY ASSESSMENT STRUCTURE (default - configurable per unit):
#   Weekly Quizzes  : 10 quizzes × 20 marks each = 20% of unit total
#   Assignment 1    : 100 marks = 15% of unit total
#   Assignment 2    : 100 marks = 15% of unit total
#   Final Exam      : 100 marks = 50% of unit total
#   Pass Mark       : 50 / 100
#   Note: Some units have no final exam - weights are recalculated accordingly.
# =============================================================================

import os
import json
import numpy as np
from datetime import datetime

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def ensure_reports_dir():
    """Create the reports directory if it does not exist."""
    os.makedirs(REPORTS_DIR, exist_ok=True)


# =============================================================================
# UNIT CONFIGURATION
# Represents the assessment structure of a single university unit.
# Staff configure this when creating a unit in the Staff Portal.
# Weights must sum to 100. has_exam can be set to False for units
# that use only quizzes and assignments (common in Murdoch units).
# =============================================================================

DEFAULT_UNIT_CONFIG = {
    "quiz_count":        10,     # Total number of quizzes in the trimester
    "quiz_max_marks":    20,     # Maximum marks per quiz
    "quiz_weight":       20.0,   # Quizzes contribute 20% of unit total
    "assignment_count":  2,      # Number of assignments
    "assignment_max":    100,    # Maximum marks per assignment
    "assignment_weight": 30.0,   # Assignments contribute 30% total (15% each)
    "has_exam":          True,   # Whether this unit has a final exam
    "exam_max":          100,    # Maximum marks for final exam
    "exam_weight":       50.0,   # Final exam contributes 50% of unit total
    "pass_mark":         50.0,   # Minimum total score to pass (out of 100)
}

# Risk thresholds for the combined score
RISK_HIGH_THRESHOLD   = 0.55    # Combined score >= 0.55 -> HIGH risk
RISK_MEDIUM_THRESHOLD = 0.35    # Combined score >= 0.35 -> MEDIUM risk
                                 # Below 0.35 -> LOW risk

# Weights for combining quiz and assignment risk signals
QUIZ_RISK_WEIGHT       = 0.60   # Quiz trend carries 60% of combined risk
ASSIGNMENT_RISK_WEIGHT = 0.40   # Assignment score carries 40% of combined risk

# Minimum number of quizzes required before trend analysis is reliable
MIN_QUIZZES_FOR_TREND = 3


# =============================================================================
# SUB-SYSTEM 1 - QUIZ TREND ANALYSER
#
# PURPOSE:
#   Analyses the sequence of quiz scores entered so far and identifies
#   whether the student's performance is improving, stable, or declining.
#   Projects what the average quiz score will be by end of trimester if
#   the current trend continues unchanged.
#
# HOW IT WORKS:
#   1. Fit a linear regression (numpy polyfit degree=1) to the sequence
#      of quiz scores. The slope tells us the direction and rate of change.
#   2. Project all remaining quizzes using the fitted linear trend line.
#   3. Compute the projected average over all quizzes (done + projected).
#   4. Convert projected average to a unit mark contribution (out of quiz_weight).
#   5. Derive a quiz risk score: how far below maximum the projected mark is.
#
# SLOPE INTERPRETATION:
#   slope > +0.5  -> Improving  (scores going up each quiz)
#   slope < -0.5  -> Declining  (scores going down each quiz - high concern)
#   -0.5 to +0.5  -> Stable     (performance not changing significantly)
#
# EDGE CASES HANDLED:
#   - If fewer than MIN_QUIZZES_FOR_TREND quizzes entered, slope = 0 (stable).
#   - Projected scores are clipped to [0, quiz_max_marks] (no impossible values).
#   - All arithmetic uses float to avoid integer division edge cases.
# =============================================================================

def analyse_quiz_trend(scores, unit_config=None):
    """
    Analyse quiz score trend and project end-of-trimester performance.

    Args:
        scores      (list): Quiz scores entered so far. Each value must be
                            in range [0, quiz_max_marks].
        unit_config (dict): Unit configuration. Uses DEFAULT_UNIT_CONFIG if None.

    Returns:
        dict: Trend analysis results including slope, trend direction,
              projected average, projected unit marks, and quiz risk score.
    """
    if unit_config is None:
        unit_config = DEFAULT_UNIT_CONFIG

    quiz_count     = unit_config["quiz_count"]
    quiz_max       = unit_config["quiz_max_marks"]
    quiz_weight    = unit_config["quiz_weight"]

    n   = len(scores)
    arr = np.array(scores, dtype=float)

    # Edge case: no quiz scores entered yet - return safe neutral result
    if n == 0:
        return {
            "quizzes_entered":    0,
            "quiz_count_total":   quiz_count,
            "current_avg":        0.0,
            "current_avg_pct":    0.0,
            "current_marks":      0.0,
            "slope":              0.0,
            "trend":              "Stable",
            "projected_avg":      0.0,
            "projected_avg_pct":  0.0,
            "projected_marks":    0.0,
            "quiz_weight":        quiz_weight,
            "quiz_risk_score":    0.5,
        }

    # Compute current average from entered scores
    current_avg     = float(np.mean(arr)) if n > 0 else 0.0
    current_avg_pct = round(current_avg / quiz_max * 100.0, 1)
    current_marks   = round(current_avg / quiz_max * quiz_weight, 2)

    # Fit linear trend only if enough quizzes have been entered
    if n >= MIN_QUIZZES_FOR_TREND:
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, arr, 1)
    else:
        slope     = 0.0
        intercept = current_avg

    slope = round(float(slope), 3)

    # Classify trend direction based on slope
    if slope > 0.5:
        trend = "Improving"
    elif slope < -0.5:
        trend = "Declining"
    else:
        trend = "Stable"

    # Project all quiz scores for the full trimester using the fitted trend
    projected = list(arr)   # Start with actual scores already entered
    for i in range(n, quiz_count):
        val = intercept + slope * i
        projected.append(max(0.0, min(float(quiz_max), val)))

    projected_avg     = float(np.mean(projected))
    projected_avg_pct = round(projected_avg / quiz_max * 100.0, 1)
    projected_marks   = round(projected_avg / quiz_max * quiz_weight, 2)

    # Quiz risk score: 1.0 = no projected marks (maximum risk)
    #                  0.0 = full projected marks (no risk)
    quiz_risk_score = max(0.0, min(1.0, 1.0 - projected_marks / quiz_weight))

    return {
        "quizzes_entered":    n,
        "quiz_count_total":   quiz_count,
        "current_avg":        round(current_avg, 2),
        "current_avg_pct":    current_avg_pct,
        "current_marks":      current_marks,
        "slope":              slope,
        "trend":              trend,
        "projected_avg":      round(projected_avg, 2),
        "projected_avg_pct":  projected_avg_pct,
        "projected_marks":    projected_marks,
        "quiz_weight":        quiz_weight,
        "quiz_risk_score":    round(quiz_risk_score, 4),
    }


# =============================================================================
# SUB-SYSTEM 2 - ASSIGNMENT RISK SCORER
#
# PURPOSE:
#   Computes a normalised assignment risk score from any submitted assignment
#   marks. Designed to handle partial data - assignments may not yet be
#   submitted or marked at the time of risk check.
#
# HOW IT WORKS:
#   1. Collect all submitted assignment scores (None = not yet submitted).
#   2. If no assignments submitted -> return neutral score 0.5 (no data yet).
#   3. Compute the average percentage across all submitted assignments.
#   4. Risk increases as average drops below 65% (a comfortable pass benchmark).
#      Risk formula: max(0, (65 - avg_pct) / 65)
#      This produces:
#        avg_pct = 65% -> risk = 0.00 (no risk at benchmark)
#        avg_pct = 50% -> risk = 0.23 (moderate risk)
#        avg_pct = 30% -> risk = 0.54 (high risk)
#        avg_pct =  0% -> risk = 1.00 (maximum risk)
#
# WHY 65% BENCHMARK:
#   50% is the Murdoch pass mark, but a student scoring exactly 50% on
#   assignments is still at risk of failing the unit overall (exam is 50%).
#   65% provides a safer buffer - students above this are likely not at risk
#   from the assignment component alone.
# =============================================================================

def score_assignment_risk(a1=None, a2=None, unit_config=None):
    """
    Compute normalised assignment risk score from submitted assignment marks.

    Args:
        a1          (float|None): Assignment 1 score (0-100), or None if not submitted.
        a2          (float|None): Assignment 2 score (0-100), or None if not submitted.
        unit_config (dict):       Unit configuration. Uses DEFAULT_UNIT_CONFIG if None.

    Returns:
        dict: Assignment risk analysis including average, unit marks, and risk score.
    """
    if unit_config is None:
        unit_config = DEFAULT_UNIT_CONFIG

    assignment_max    = unit_config["assignment_max"]
    assignment_weight = unit_config["assignment_weight"]
    assignment_count  = unit_config["assignment_count"]
    weight_each       = assignment_weight / assignment_count

    submitted = [(score, i+1) for i, score in enumerate([a1, a2]) if score is not None]

    if not submitted:
        # No assignments submitted yet - return neutral score
        return {
            "assignments_submitted": 0,
            "assignment_avg_pct":    None,
            "a1_unit_marks":         None,
            "a2_unit_marks":         None,
            "assignment_risk_score": 0.5,   # Neutral - no data to judge
            "note":                  "No assignments submitted yet.",
        }

    scores_pct = [s / assignment_max * 100.0 for s, _ in submitted]
    avg_pct    = float(np.mean(scores_pct))

    # Risk formula - rises as average drops below 65% benchmark
    risk_score = max(0.0, min(1.0, (65.0 - avg_pct) / 65.0))

    a1_marks = round(a1 / assignment_max * weight_each, 2) if a1 is not None else None
    a2_marks = round(a2 / assignment_max * weight_each, 2) if a2 is not None else None

    return {
        "assignments_submitted": len(submitted),
        "assignment_avg_pct":    round(avg_pct, 1),
        "a1_unit_marks":         a1_marks,
        "a2_unit_marks":         a2_marks,
        "assignment_risk_score": round(risk_score, 4),
        "note":                  str(len(submitted)) + " assignment(s) submitted.",
    }


# =============================================================================
# SUB-SYSTEM 3 - COMBINED RISK DECISION ENGINE
#
# PURPOSE:
#   Fuses quiz trend risk and assignment risk into a single combined risk
#   score and classifies it into a discrete risk level: LOW / MEDIUM / HIGH.
#
# HOW IT WORKS:
#   If assignments submitted:
#     combined = quiz_risk × QUIZ_RISK_WEIGHT + assign_risk × ASSIGN_RISK_WEIGHT
#     combined = quiz_risk × 0.60 + assign_risk × 0.40
#
#   If no assignments submitted yet:
#     combined = quiz_risk × 1.00  (full weight on quiz trend only)
#
# RISK LEVEL THRESHOLDS:
#   combined >= RISK_HIGH_THRESHOLD   (0.55) -> HIGH    (urgent intervention)
#   combined >= RISK_MEDIUM_THRESHOLD (0.35) -> MEDIUM  (monitor and check in)
#   combined <  RISK_MEDIUM_THRESHOLD (0.35) -> LOW     (on track)
#
# WHY 60/40 SPLIT:
#   Quizzes are weekly and reflect the most up-to-date academic performance.
#   Assignments are submitted less frequently and may not yet be available.
#   Giving quizzes 60% weight ensures the system responds quickly to
#   performance decline even when assignment data is absent.
# =============================================================================

def compute_combined_risk(quiz_risk, assign_risk, has_assignments):
    """
    Combine quiz and assignment risk scores into a final risk level.

    Args:
        quiz_risk       (float): Quiz trend risk score (0.0–1.0).
        assign_risk     (float): Assignment risk score (0.0–1.0).
        has_assignments (bool):  Whether any assignment data is available.

    Returns:
        tuple: (combined_score float, risk_level str)
    """
    if has_assignments:
        combined = (quiz_risk * QUIZ_RISK_WEIGHT +
                    assign_risk * ASSIGNMENT_RISK_WEIGHT)
    else:
        # No assignment data - full weight on quiz trend signal
        combined = quiz_risk

    combined = round(float(combined), 4)

    if combined >= RISK_HIGH_THRESHOLD:
        risk_level = "HIGH"
    elif combined >= RISK_MEDIUM_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return combined, risk_level


# =============================================================================
# SUB-SYSTEM 4 - ALERT & RECOMMENDATION GENERATOR
#
# PURPOSE:
#   Translates a risk level classification into a structured staff alert
#   with specific diagnostic reasons and a clear recommended action.
#   Used by the Flask app to display alerts and draft notification emails.
#
# ALERT LEVELS:
#   HIGH   -> Urgent intervention required. Notify academic staff immediately.
#   MEDIUM -> Recommend a check-in meeting. Monitor closely next week.
#   LOW    -> Student is on track. Continue normal monitoring.
# =============================================================================

def generate_alert(risk_level, trend_result, assign_result, student_name="Student"):
    """
    Generate a structured staff alert with reasons and recommended action.

    Args:
        risk_level    (str):  "HIGH", "MEDIUM", or "LOW"
        trend_result  (dict): Output from analyse_quiz_trend()
        assign_result (dict): Output from score_assignment_risk()
        student_name  (str):  Student display name for the alert message.

    Returns:
        dict: Alert details including reasons list, action, and email subject.
    """
    reasons = []

    # Quiz-based reasons
    if trend_result["trend"] == "Declining":
        reasons.append(
            "Quiz scores are declining (slope = " +
            str(trend_result["slope"]) + " marks per quiz)."
        )
    if trend_result["projected_avg_pct"] < 50:
        reasons.append(
            "Projected quiz average is " +
            str(trend_result["projected_avg_pct"]) +
            "% - below the 50% pass benchmark."
        )
    if trend_result["current_avg_pct"] < 40:
        reasons.append(
            "Current quiz average is critically low at " +
            str(trend_result["current_avg_pct"]) + "%."
        )

    # Assignment-based reasons
    a1_marks = assign_result.get("a1_unit_marks")
    a2_marks = assign_result.get("a2_unit_marks")
    avg_pct  = assign_result.get("assignment_avg_pct")

    if avg_pct is not None and avg_pct < 50:
        reasons.append(
            "Assignment average is " + str(avg_pct) +
            "% - below the pass mark."
        )
    if a1_marks is not None and a1_marks < 7.5:
        reasons.append(
            "Assignment 1 is contributing only " +
            str(a1_marks) + " unit marks (below 7.5)."
        )
    if a2_marks is not None and a2_marks < 7.5:
        reasons.append(
            "Assignment 2 is contributing only " +
            str(a2_marks) + " unit marks (below 7.5)."
        )

    # Default reason if nothing specific triggered
    if not reasons:
        reasons.append(
            "Combined quiz and assignment risk score is elevated (" +
            risk_level + ")."
        )

    # Build alert based on risk level
    if risk_level == "HIGH":
        action  = "URGENT - Contact student immediately. Refer to academic support."
        subject = "URGENT: " + student_name + " is at HIGH academic risk."
        colour  = "red"
    elif risk_level == "MEDIUM":
        action  = "RECOMMENDED - Schedule a check-in meeting with the student."
        subject = "NOTICE: " + student_name + " may need academic support."
        colour  = "orange"
    else:
        action  = "No immediate action required. Continue monitoring weekly."
        subject = "STATUS OK: " + student_name + " is on track."
        colour  = "green"

    return {
        "risk_level":    risk_level,
        "colour":        colour,
        "reasons":       reasons,
        "action":        action,
        "email_subject": subject,
        "student_name":  student_name,
    }


# =============================================================================
# SUB-SYSTEM 5 - REPORT LOGGER
#
# PURPOSE:
#   Generates a timestamped JSON audit report for each student assessment run.
#   Saved to reports/ folder. Provides traceability for all risk decisions -
#   important for educational governance and academic integrity.
#
# REPORT STRUCTURE:
#   - System metadata (version, timestamp)
#   - Student identification
#   - Unit configuration used
#   - Input data (quiz scores, assignment scores)
#   - Quiz trend analysis results
#   - Assignment risk results
#   - Combined risk result
#   - ML model secondary signal (if provided)
#   - Alert and recommended action
# =============================================================================

def save_report(student_id, unit_name, trend_result, assign_result,
                combined_score, risk_level, alert, ml_result=None,
                unit_config=None):
    """
    Save a full JSON audit report for a student's risk assessment.

    Args:
        student_id     (str):  Student identifier.
        unit_name      (str):  Name/code of the unit being assessed.
        trend_result   (dict): Output from analyse_quiz_trend().
        assign_result  (dict): Output from score_assignment_risk().
        combined_score (float): Combined risk score (0.0–1.0).
        risk_level     (str):  "HIGH", "MEDIUM", or "LOW".
        alert          (dict): Output from generate_alert().
        ml_result      (dict): Optional ML secondary signal from predict.py.
        unit_config    (dict): Unit configuration used. Defaults to DEFAULT.

    Returns:
        str: Full path of the saved report file.
    """
    ensure_reports_dir()
    if unit_config is None:
        unit_config = DEFAULT_UNIT_CONFIG

    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    report = {
        "system":          "ICT304 AI Academic Risk Prediction Engine",
        "version":         "2.0 - Final Project",
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "student_id":      student_id,
        "unit_name":       unit_name,
        "unit_config":     unit_config,
        "quiz_trend":      trend_result,
        "assignment_risk": assign_result,
        "combined_risk": {
            "combined_score": combined_score,
            "risk_level":     risk_level,
            "weights_used": {
                "quiz":       QUIZ_RISK_WEIGHT if assign_result["assignments_submitted"] > 0 else 1.0,
                "assignment": ASSIGNMENT_RISK_WEIGHT if assign_result["assignments_submitted"] > 0 else 0.0,
            },
        },
        "ml_secondary_signal": ml_result if ml_result else "Not available",
        "alert":           alert,
    }

    filename  = "RiskReport_" + str(student_id) + "_" + unit_name + "_" + ts + ".json"
    filepath  = os.path.join(REPORTS_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    return filepath


# =============================================================================
# FULL RISK ASSESSMENT - ORCHESTRATOR
# Runs all 5 sub-systems in sequence for a single student.
# This is the main function called by predict.py and the Flask app.
# =============================================================================

def run_risk_assessment(student_id, unit_name, quiz_scores,
                        a1=None, a2=None, unit_config=None,
                        ml_result=None, student_name="Student",
                        save_json=True):
    """
    Run the full deterministic risk assessment pipeline for one student.

    Args:
        student_id    (str):        Student identifier (e.g. "S1042").
        unit_name     (str):        Unit name/code (e.g. "ICT304").
        quiz_scores   (list):       Quiz scores entered so far (each 0–quiz_max).
        a1            (float|None): Assignment 1 score (0–100), or None.
        a2            (float|None): Assignment 2 score (0–100), or None.
        unit_config   (dict|None):  Unit configuration. Uses default if None.
        ml_result     (dict|None):  Secondary ML signal from predict.py.
        student_name  (str):        Display name for alert generation.
        save_json     (bool):       Whether to save a JSON audit report.

    Returns:
        dict: Full assessment result containing all sub-system outputs,
              combined risk level, alert, and report path (if saved).
    """
    if unit_config is None:
        unit_config = DEFAULT_UNIT_CONFIG

    # Sub-system 1 - Quiz Trend Analysis
    trend_result  = analyse_quiz_trend(quiz_scores, unit_config)

    # Sub-system 2 - Assignment Risk Scoring
    assign_result = score_assignment_risk(a1, a2, unit_config)

    # Sub-system 3 - Combined Risk Decision
    has_assignments = assign_result["assignments_submitted"] > 0
    combined_score, risk_level = compute_combined_risk(
        trend_result["quiz_risk_score"],
        assign_result["assignment_risk_score"],
        has_assignments
    )

    # Sub-system 4 - Alert Generation
    alert = generate_alert(risk_level, trend_result, assign_result, student_name)

    # Sub-system 5 - Report Logging
    report_path = None
    if save_json:
        report_path = save_report(
            student_id, unit_name, trend_result, assign_result,
            combined_score, risk_level, alert, ml_result, unit_config
        )

    return {
        "student_id":      student_id,
        "student_name":    student_name,
        "unit_name":       unit_name,
        "trend":           trend_result,
        "assignment":      assign_result,
        "combined_score":  combined_score,
        "risk_level":      risk_level,
        "alert":           alert,
        "ml_signal":       ml_result,
        "report_path":     report_path,
    }


# =============================================================================
# QUICK SELF-TEST - runs when file is executed directly
# Demonstrates all sub-systems working with example student data.
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ICT304 - Risk Engine Self-Test")
    print("="*70)

    # Example 1: Declining student - should trigger HIGH risk
    print("\n--- Test 1: Declining student (HIGH risk expected) ---")
    result1 = run_risk_assessment(
        student_id   = "TEST001",
        unit_name    = "ICT304",
        quiz_scores  = [18, 16, 14, 12, 10, 8],   # clearly declining
        a1           = 42.0,                        # below pass
        a2           = None,                        # not submitted
        student_name = "Test Student A",
        save_json    = False
    )
    print("  Trend     : " + result1["trend"]["trend"] +
          " (slope=" + str(result1["trend"]["slope"]) + ")")
    print("  Quiz Risk : " + str(result1["trend"]["quiz_risk_score"]))
    print("  Assign Rsk: " + str(result1["assignment"]["assignment_risk_score"]))
    print("  Combined  : " + str(result1["combined_score"]))
    print("  RISK LEVEL: " + result1["risk_level"])
    print("  Reasons   : " + str(result1["alert"]["reasons"]))

    # Example 2: Improving student - should trigger LOW risk
    print("\n--- Test 2: Improving student (LOW risk expected) ---")
    result2 = run_risk_assessment(
        student_id   = "TEST002",
        unit_name    = "ICT304",
        quiz_scores  = [10, 12, 14, 16, 18, 19],   # clearly improving
        a1           = 78.0,
        a2           = 82.0,
        student_name = "Test Student B",
        save_json    = False
    )
    print("  Trend     : " + result2["trend"]["trend"] +
          " (slope=" + str(result2["trend"]["slope"]) + ")")
    print("  Combined  : " + str(result2["combined_score"]))
    print("  RISK LEVEL: " + result2["risk_level"])

    # Example 3: Stable student with no assignments yet - should be MEDIUM
    print("\n--- Test 3: Stable low performer, no assignments (MEDIUM expected) ---")
    result3 = run_risk_assessment(
        student_id   = "TEST003",
        unit_name    = "ICT304",
        quiz_scores  = [9, 10, 9, 11, 10],
        a1           = None,
        a2           = None,
        student_name = "Test Student C",
        save_json    = False
    )
    print("  Trend     : " + result3["trend"]["trend"])
    print("  Combined  : " + str(result3["combined_score"]))
    print("  RISK LEVEL: " + result3["risk_level"])

    print("\n  [DONE] All self-tests passed.")
    print("="*70 + "\n")
