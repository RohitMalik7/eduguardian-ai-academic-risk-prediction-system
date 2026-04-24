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
# database/models.py
# ICT304 AI Academic Risk Prediction System
# Data Access Layer - application read/write helpers for SQLite data
# =============================================================================

import sqlite3
from database.db import get_connection


# =============================================================================
# SECTION 1 - USER / AUTH FUNCTIONS
# Used by: app/auth.py
# =============================================================================

def get_user_by_email(email: str):
    """
    Fetch a single user row by email address.
    Returns a sqlite3.Row object or None if not found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE email = ?",
        (email.strip().lower(),)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: int):
    """Fetch a single user row by primary key."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def validate_login(email: str, password: str):
    """
    Validate login credentials.
    Returns the user row if credentials are correct, otherwise None.
    """
    user = get_user_by_email(email)
    if user and user["password"] == password:
        return user
    return None


def get_all_students():
    """
    Return all users with role = student, ordered by student_id.
    Used by staff portal to list all students.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE role = 'student' ORDER BY student_id",
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


# =============================================================================
# SECTION 2 - UNIT FUNCTIONS
# Used by: Flask routes and service-layer helpers for unit display and configuration
# =============================================================================

def get_all_units():
    """
    Return all units joined with their unit_config so the staff portal
    gets is_active, quiz_count, weights etc. in one query.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.*,
               uc.quiz_count, uc.quiz_max_marks, uc.quiz_weight_pct,
               uc.a1_max_marks, uc.a1_weight_pct,
               uc.a2_max_marks, uc.a2_weight_pct,
               uc.final_weight_pct, uc.pass_mark_pct, uc.is_active
        FROM units u
        LEFT JOIN unit_config uc ON uc.unit_id = u.unit_id
        ORDER BY u.unit_code
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_unit_by_code(unit_code: str):
    """Fetch a unit row by its code, e.g. 'ICT304'."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM units WHERE unit_code = ?",
        (unit_code.strip().upper(),)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def get_unit_by_id(unit_id: int):
    """Fetch a unit row by primary key."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM units WHERE unit_id = ?", (unit_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def get_active_units_for_student(user_id: int):
    """
    Return all units a student is enrolled in where is_active = 1.
    Students can only see and submit to active units.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.*, uc.quiz_count, uc.quiz_max_marks, uc.quiz_weight_pct,
               uc.a1_max_marks, uc.a1_weight_pct,
               uc.a2_max_marks, uc.a2_weight_pct,
               uc.final_weight_pct, uc.pass_mark_pct, uc.is_active
        FROM units u
        JOIN enrollments e   ON e.unit_id = u.unit_id
        JOIN unit_config uc  ON uc.unit_id = u.unit_id
        WHERE e.user_id = ?
          AND uc.is_active = 1
        ORDER BY u.unit_code
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


# =============================================================================
# SECTION 3 - UNIT CONFIG FUNCTIONS
# Used by: staff-facing Flask routes and dashboard actions
# =============================================================================

def get_unit_config(unit_id: int):
    """Fetch the configuration row for a given unit."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM unit_config WHERE unit_id = ?",
        (unit_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def save_unit_config(unit_id: int, quiz_count: int, quiz_max_marks: float,
                     quiz_weight_pct: float, a1_max_marks: float,
                     a1_weight_pct: float, a2_max_marks: float,
                     a2_weight_pct: float, final_weight_pct: float,
                     pass_mark_pct: float):
    """
    Insert or update the unit configuration.
    Staff uses this to set assessment structure before activating the unit.
    Does NOT activate the unit - call set_unit_active() separately.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO unit_config (
            unit_id, quiz_count, quiz_max_marks, quiz_weight_pct,
            a1_max_marks, a1_weight_pct, a2_max_marks, a2_weight_pct,
            final_weight_pct, pass_mark_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(unit_id) DO UPDATE SET
            quiz_count        = excluded.quiz_count,
            quiz_max_marks    = excluded.quiz_max_marks,
            quiz_weight_pct   = excluded.quiz_weight_pct,
            a1_max_marks      = excluded.a1_max_marks,
            a1_weight_pct     = excluded.a1_weight_pct,
            a2_max_marks      = excluded.a2_max_marks,
            a2_weight_pct     = excluded.a2_weight_pct,
            final_weight_pct  = excluded.final_weight_pct,
            pass_mark_pct     = excluded.pass_mark_pct,
            configured_at     = datetime('now')
    """, (
        unit_id, quiz_count, quiz_max_marks, quiz_weight_pct,
        a1_max_marks, a1_weight_pct, a2_max_marks, a2_weight_pct,
        final_weight_pct, pass_mark_pct
    ))
    conn.commit()
    conn.close()


def set_unit_active(unit_id: int, is_active: bool):
    """
    Activate or deactivate a unit.
    When active (is_active=True), students can see the unit and submit assessments.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE unit_config SET is_active = ? WHERE unit_id = ?",
        (1 if is_active else 0, unit_id)
    )
    conn.commit()
    conn.close()


# =============================================================================
# SECTION 4 - ENROLLMENT FUNCTIONS
# Used by: seed_db.py and staff-facing enrollment workflows
# =============================================================================

def enroll_student(user_id: int, unit_id: int):
    """
    Enroll a student in a unit.
    Safe to call multiple times - UNIQUE constraint prevents duplicates.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO enrollments (user_id, unit_id)
        VALUES (?, ?)
    """, (user_id, unit_id))
    conn.commit()
    conn.close()


def get_students_in_unit(unit_id: int):
    """
    Return all students enrolled in a given unit, joined with their user data.
    Used by staff portal to see the full cohort.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.user_id, u.student_id, u.full_name, u.email,
               e.enrolled_at
        FROM users u
        JOIN enrollments e ON e.user_id = u.user_id
        WHERE e.unit_id = ?
          AND u.role = 'student'
        ORDER BY u.student_id
    """, (unit_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


# =============================================================================
# SECTION 5 - QUIZ SUBMISSION FUNCTIONS
# Used by: student submission routes and staff cohort review screens
# =============================================================================

def save_quiz_score(user_id: int, unit_id: int, quiz_number: int, score: float):
    """
    Save or update a quiz score for a student.
    If the student has already submitted quiz_number, the score is overwritten.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO quiz_submissions (user_id, unit_id, quiz_number, score)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, unit_id, quiz_number) DO UPDATE SET
            score        = excluded.score,
            submitted_at = datetime('now')
    """, (user_id, unit_id, quiz_number, score))
    conn.commit()
    conn.close()


def get_quiz_scores(user_id: int, unit_id: int):
    """
    Return all quiz scores for a student in a unit, ordered by quiz_number.
    Returns a list of sqlite3.Row objects.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT quiz_number, score, submitted_at
        FROM quiz_submissions
        WHERE user_id = ? AND unit_id = ?
        ORDER BY quiz_number
    """, (user_id, unit_id))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_quiz_scores_list(user_id: int, unit_id: int):
    """
    Return quiz scores as a plain Python list of floats, ordered by quiz_number.
    This is the format expected by risk_engine.py's analyse_quiz_trend().
    """
    rows = get_quiz_scores(user_id, unit_id)
    return [row["score"] for row in rows]


def get_next_quiz_number(user_id: int, unit_id: int):
    """
    Return the next quiz number the student should submit.
    Returns None if all quizzes in the unit have been submitted.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT uc.quiz_count,
               COUNT(qs.submission_id) AS submitted_count
        FROM unit_config uc
        LEFT JOIN quiz_submissions qs
               ON qs.unit_id = uc.unit_id
              AND qs.user_id = ?
        WHERE uc.unit_id = ?
        GROUP BY uc.quiz_count
    """, (user_id, unit_id))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    next_num = row["submitted_count"] + 1
    return next_num if next_num <= row["quiz_count"] else None


def get_all_quiz_scores_for_unit(unit_id: int):
    """
    Return all quiz submissions across all students in a unit.
    Used by staff portal for cohort-level analysis.
    Returns rows with: user_id, student_id, full_name, quiz_number, score.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.user_id, u.student_id, u.full_name,
               qs.quiz_number, qs.score, qs.submitted_at
        FROM quiz_submissions qs
        JOIN users u ON u.user_id = qs.user_id
        WHERE qs.unit_id = ?
        ORDER BY u.student_id, qs.quiz_number
    """, (unit_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


# =============================================================================
# SECTION 6 - ASSIGNMENT SUBMISSION FUNCTIONS
# Used by: student_portal.py (submit), staff_portal.py (view)
# =============================================================================

def save_assignment_score(user_id: int, unit_id: int,
                          assignment_number: int, score: float):
    """
    Save or update an assignment score for a student.
    assignment_number must be 1 or 2.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO assignment_submissions (user_id, unit_id, assignment_number, score)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, unit_id, assignment_number) DO UPDATE SET
            score        = excluded.score,
            submitted_at = datetime('now')
    """, (user_id, unit_id, assignment_number, score))
    conn.commit()
    conn.close()


def get_assignment_scores(user_id: int, unit_id: int):
    """
    Return all assignment submissions for a student in a unit.
    Returns a dict: {1: score_or_None, 2: score_or_None}
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT assignment_number, score
        FROM assignment_submissions
        WHERE user_id = ? AND unit_id = ?
        ORDER BY assignment_number
    """, (user_id, unit_id))
    rows = cursor.fetchall()
    conn.close()
    result = {1: None, 2: None}
    for row in rows:
        result[row["assignment_number"]] = row["score"]
    return result


def get_all_assignment_scores_for_unit(unit_id: int):
    """
    Return all assignment submissions across all students in a unit.
    Used by staff portal for cohort-level reporting.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.user_id, u.student_id, u.full_name,
               asub.assignment_number, asub.score, asub.submitted_at
        FROM assignment_submissions asub
        JOIN users u ON u.user_id = asub.user_id
        WHERE asub.unit_id = ?
        ORDER BY u.student_id, asub.assignment_number
    """, (unit_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


# =============================================================================
# SECTION 7 - RISK RESULT FUNCTIONS
# Used by: predict.py (write), staff_portal.py (read), student_portal.py (read)
# =============================================================================

def save_risk_result(user_id: int, unit_id: int, quiz_avg_pct: float,
                     projected_avg_pct: float, trend_direction: str,
                     trend_slope: float, a1_score, a2_score,
                     combined_risk_score: float, risk_level: str,
                     ml_probability: float, selected_model: str,
                     marks_earned: float, marks_available: float,
                     ml_weight_used: float, engine_weight_used: float,
                     medium_threshold_used: float, high_threshold_used: float):
    """
    Insert or update the risk prediction result for a student in a unit.
    Called by predict.py after every quiz or assignment submission.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO risk_results (
            user_id, unit_id, quiz_avg_pct, projected_avg_pct,
            trend_direction, trend_slope, a1_score, a2_score,
            combined_risk_score, risk_level, ml_probability,
            selected_model, marks_earned, marks_available,
            ml_weight_used, engine_weight_used,
            medium_threshold_used, high_threshold_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, unit_id) DO UPDATE SET
            quiz_avg_pct            = excluded.quiz_avg_pct,
            projected_avg_pct       = excluded.projected_avg_pct,
            trend_direction         = excluded.trend_direction,
            trend_slope             = excluded.trend_slope,
            a1_score                = excluded.a1_score,
            a2_score                = excluded.a2_score,
            combined_risk_score     = excluded.combined_risk_score,
            risk_level              = excluded.risk_level,
            ml_probability          = excluded.ml_probability,
            selected_model          = excluded.selected_model,
            marks_earned            = excluded.marks_earned,
            marks_available         = excluded.marks_available,
            ml_weight_used          = excluded.ml_weight_used,
            engine_weight_used      = excluded.engine_weight_used,
            medium_threshold_used   = excluded.medium_threshold_used,
            high_threshold_used     = excluded.high_threshold_used,
            calculated_at           = datetime('now')
    """, (
        user_id, unit_id, quiz_avg_pct, projected_avg_pct,
        trend_direction, trend_slope, a1_score, a2_score,
        combined_risk_score, risk_level, ml_probability,
        selected_model, marks_earned, marks_available,
        ml_weight_used, engine_weight_used,
        medium_threshold_used, high_threshold_used
    ))
    conn.commit()
    conn.close()


def get_risk_result(user_id: int, unit_id: int):
    """
    Return the latest risk prediction result for a student in a unit.
    Returns None if no prediction has been run yet.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM risk_results
        WHERE user_id = ? AND unit_id = ?
    """, (user_id, unit_id))
    row = cursor.fetchone()
    conn.close()
    return row


def get_all_risk_results_for_unit(unit_id: int):
    """
    Return all risk results for every student in a unit.
    Joined with user info for display in staff portal.
    Returns rows ordered by risk_level (HIGH first) then student_id.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.user_id, u.student_id, u.full_name,
               rr.quiz_avg_pct, rr.projected_avg_pct,
               rr.trend_direction, rr.trend_slope,
               rr.a1_score, rr.a2_score,
               rr.combined_risk_score, rr.risk_level,
               rr.ml_probability, rr.selected_model,
               rr.marks_earned, rr.marks_available,
               rr.ml_weight_used, rr.engine_weight_used,
               rr.medium_threshold_used, rr.high_threshold_used,
               rr.calculated_at
        FROM risk_results rr
        JOIN users u ON u.user_id = rr.user_id
        WHERE rr.unit_id = ?
        ORDER BY
            CASE rr.risk_level
                WHEN 'HIGH'   THEN 1
                WHEN 'MEDIUM' THEN 2
                WHEN 'LOW'    THEN 3
                ELSE 4
            END,
            u.student_id
    """, (unit_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_risk_summary_for_unit(unit_id: int):
    """
    Return a summary dict with counts per risk level for a unit.
    Used by staff portal dashboard cards.
    Example return: {"HIGH": 3, "MEDIUM": 7, "LOW": 10, "total": 20}
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT risk_level, COUNT(*) as count
        FROM risk_results
        WHERE unit_id = ?
        GROUP BY risk_level
    """, (unit_id,))
    rows = cursor.fetchall()
    conn.close()
    summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "total": 0}
    for row in rows:
        if row["risk_level"] in summary:
            summary[row["risk_level"]] = row["count"]
            summary["total"] += row["count"]
    return summary


# =============================================================================
# SECTION 8 - COMBINED STUDENT PROGRESS VIEW
# Used by: staff_portal.py (full cohort table), student_portal.py (own progress)
# =============================================================================

def get_student_progress(user_id: int, unit_id: int):
    """
    Return a single dict with everything needed to display a student's
    current progress: quiz scores list, assignment scores, and latest risk result.

    Return format:
    {
        "quiz_scores":       [12.0, 15.5, 10.0, ...],   # list of raw scores
        "quizzes_submitted": 4,
        "assignment_1":      75.0 or None,
        "assignment_2":      None,
        "risk":              sqlite3.Row or None          # from risk_results
    }
    """
    quiz_scores = get_quiz_scores_list(user_id, unit_id)
    assignments = get_assignment_scores(user_id, unit_id)
    risk        = get_risk_result(user_id, unit_id)

    return {
        "quiz_scores":       quiz_scores,
        "quizzes_submitted": len(quiz_scores),
        "assignment_1":      assignments[1],
        "assignment_2":      assignments[2],
        "risk":              risk
    }


def get_full_cohort_progress(unit_id: int):
    """
    Return a list of JSON-serializable dictionaries for every enrolled student.
    Converts sqlite3.Row objects to plain dicts so Flask JSON responses work correctly.
    """
    students = get_students_in_unit(unit_id)
    cohort = []
    for student in students:
        progress = get_student_progress(student["user_id"], unit_id)
        # Convert risk sqlite3.Row to plain dict (required for JSON serialization)
        risk_row = progress.get("risk")
        risk_dict = dict(risk_row) if risk_row is not None else None
        cohort.append({
            "user_id":           student["user_id"],
            "student_id":        student["student_id"],
            "full_name":         student["full_name"],
            "email":             student["email"],
            "quiz_scores":       progress["quiz_scores"],
            "quizzes_submitted": progress["quizzes_submitted"],
            "assignment_1":      progress["assignment_1"],
            "assignment_2":      progress["assignment_2"],
            "risk":              risk_dict,
        })
    return cohort


def get_available_units_for_student(user_id: int):
    """
    Units that are active but the student is NOT yet enrolled in.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.*, uc.quiz_count, uc.quiz_max_marks, uc.quiz_weight_pct,
               uc.a1_weight_pct, uc.a2_weight_pct, uc.final_weight_pct,
               uc.pass_mark_pct
        FROM units u
        JOIN unit_config uc ON uc.unit_id = u.unit_id
        WHERE uc.is_active = 1
          AND u.unit_id NOT IN (
              SELECT unit_id
              FROM enrollments
              WHERE user_id = ?
          )
        ORDER BY u.unit_code
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def create_unit(unit_code: str, unit_name: str, trimester: str, created_by: int):
    """
    Create a new unit record and return its generated unit_id.
    unit_code is normalised to uppercase before insertion.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO units (unit_code, unit_name, trimester, created_by)
        VALUES (?, ?, ?, ?)
    """, (unit_code.strip().upper(), unit_name.strip(), trimester.strip(), created_by))
    conn.commit()
    unit_id = cursor.lastrowid
    conn.close()
    return unit_id


def update_unit(unit_code: str, unit_name: str, trimester: str):
    """
    Update the name and trimester of an existing unit identified by unit_code.
    The unit code itself remains unchanged.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE units
        SET unit_name = ?, trimester = ?
        WHERE unit_code = ?
    """, (unit_name.strip(), trimester.strip(), unit_code.strip().upper()))
    conn.commit()
    conn.close()


def delete_unit(unit_code: str):
    """
    Delete a unit and all its child data in the correct FK order:
      risk_results -> assignment_submissions -> quiz_submissions
      -> enrollments -> unit_config -> units
    FK constraints are ON, so we must delete children before parent.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Get unit_id first
        cursor.execute("SELECT unit_id FROM units WHERE unit_code = ?",
                       (unit_code.strip().upper(),))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return   # Nothing to delete

        uid = row["unit_id"]

        # Delete children in dependency order
        cursor.execute("DELETE FROM risk_results          WHERE unit_id = ?", (uid,))
        cursor.execute("DELETE FROM assignment_submissions WHERE unit_id = ?", (uid,))
        cursor.execute("DELETE FROM quiz_submissions       WHERE unit_id = ?", (uid,))
        cursor.execute("DELETE FROM enrollments            WHERE unit_id = ?", (uid,))
        cursor.execute("DELETE FROM unit_config            WHERE unit_id = ?", (uid,))
        cursor.execute("DELETE FROM units                  WHERE unit_id = ?", (uid,))

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def enroll_student_by_unit_code(user_id: int, unit_code: str):
    """
    Enrol a student in a unit using the unit code instead of the numeric unit_id.
    Raises ValueError if the unit code does not exist.
    """
    unit = get_unit_by_code(unit_code)
    if not unit:
        raise ValueError("Unit not found")
    enroll_student(user_id, unit["unit_id"])


def get_student_dashboard_data(user_id: int):
    """
    Return full unit data (config + progress + risk) for the student dashboard.
    Includes all assessment config fields so the frontend can build grade forms.
    """
    units = get_active_units_for_student(user_id)
    result = []
    for unit in units:
        progress = get_student_progress(user_id, unit["unit_id"])
        result.append({
            # Unit identity
            "unit_id":            unit["unit_id"],
            "unit_code":          unit["unit_code"],
            "unit_name":          unit["unit_name"],
            "trimester":          unit["trimester"],
            # Assessment configuration (from unit_config)
            "quiz_count":         unit["quiz_count"],
            "quiz_max_marks":     unit["quiz_max_marks"],
            "quiz_weight_pct":    unit["quiz_weight_pct"],
            "a1_max_marks":       unit["a1_max_marks"],
            "a1_weight_pct":      unit["a1_weight_pct"],
            "a2_max_marks":       unit["a2_max_marks"],
            "a2_weight_pct":      unit["a2_weight_pct"],
            "final_weight_pct":   unit["final_weight_pct"],
            "pass_mark_pct":      unit["pass_mark_pct"],
            # Student progress
            "quizzes_submitted":  progress["quizzes_submitted"],
            "quiz_scores":        progress["quiz_scores"],      # list of raw scores
            "assignment_1":       progress["assignment_1"],
            "assignment_2":       progress["assignment_2"],
            "risk":               dict(progress["risk"]) if progress["risk"] else None,
        })
    return result