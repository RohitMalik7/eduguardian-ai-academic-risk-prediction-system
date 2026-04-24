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
# app/routes.py
# ICT304 - EduGuardian AI - Academic Risk Prediction System
# All URL routes and API endpoints for the Flask application.
# =============================================================================

from flask import (
    Blueprint, render_template, request,
    redirect, url_for, jsonify, current_app
)
from .auth import login_user, logout_user, current_user, login_required
from database.models import (
    get_all_units,
    get_students_in_unit,
    create_unit,
    update_unit,
    delete_unit,
    get_unit_by_code,
    save_unit_config,
    set_unit_active,
    get_active_units_for_student,
    get_available_units_for_student,
    enroll_student_by_unit_code,
    get_student_dashboard_data,
    get_unit_config,
    get_next_quiz_number,
    save_quiz_score,
    save_assignment_score,
    get_assignment_scores,
    get_quiz_scores_list,
    get_full_cohort_progress,
    get_risk_summary_for_unit,
)
from .services import run_prediction_for_unit
from database.db import get_connection
from flask import session

main = Blueprint("main", __name__)


# =============================================================================
# PAGE ROUTES
# =============================================================================

@main.route("/", methods=["GET"])
def index():
    session.clear()
    return redirect(url_for("main.login"))

@main.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        role     = request.form.get("role", "student").strip().lower()

        if login_user(email, password, role):
            if role == "staff":
                return redirect(url_for("main.staff_portal"))
            return redirect(url_for("main.student_portal"))

        return render_template("login.html",
                               error="Incorrect credentials. Please try again.",
                               role=role)

    return render_template("login.html", error=None, role="student")


@main.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("main.login"))


@main.route("/staff")
@login_required(role="staff")
def staff_portal():
    user = current_user()
    return render_template("staff_portal.html", user=user)


@main.route("/student")
@login_required(role="student")
def student_portal():
    user = current_user()
    return render_template("student_portal.html", user=user)


# =============================================================================
# UNIT API - Staff only
# =============================================================================

@main.route("/api/units", methods=["GET"])
@login_required()
def api_get_units():
    user = current_user()
    if user["role"] == "staff":
        rows = get_all_units()
    else:
        rows = get_active_units_for_student(user["id"])
    return jsonify([dict(r) for r in rows])


@main.route("/api/units", methods=["POST"])
@login_required(role="staff")
def api_create_unit():
    user = current_user()
    data = request.get_json(force=True)
    try:
        unit_id = create_unit(
            unit_code  = data["unit_code"],
            unit_name  = data["unit_name"],
            trimester  = data["trimester"],
            created_by = user["id"]
        )
        save_unit_config(
            unit_id          = unit_id,
            quiz_count       = int(data.get("quiz_count",       10)),
            quiz_max_marks   = float(data.get("quiz_max_marks",  20)),
            quiz_weight_pct  = float(data.get("quiz_weight_pct", 20)),
            a1_max_marks     = float(data.get("a1_max_marks",   100)),
            a1_weight_pct    = float(data.get("a1_weight_pct",   15)),
            a2_max_marks     = float(data.get("a2_max_marks",   100)),
            a2_weight_pct    = float(data.get("a2_weight_pct",   15)),
            final_weight_pct = float(data.get("final_weight_pct",50)),
            pass_mark_pct    = float(data.get("pass_mark_pct",   50)),
        )
        return jsonify({"status": "ok", "unit_id": unit_id}), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@main.route("/api/units/<unit_code>", methods=["PUT"])
@login_required(role="staff")
def api_update_unit(unit_code):
    data = request.get_json(force=True)
    unit = get_unit_by_code(unit_code)
    if not unit:
        return jsonify({"status": "error", "message": "Unit not found"}), 404

    update_unit(
        unit_code = unit_code,
        unit_name = data["unit_name"],
        trimester = data["trimester"]
    )
    save_unit_config(
        unit_id          = unit["unit_id"],
        quiz_count       = int(data.get("quiz_count",       10)),
        quiz_max_marks   = float(data.get("quiz_max_marks",  20)),
        quiz_weight_pct  = float(data.get("quiz_weight_pct", 20)),
        a1_max_marks     = float(data.get("a1_max_marks",   100)),
        a1_weight_pct    = float(data.get("a1_weight_pct",   15)),
        a2_max_marks     = float(data.get("a2_max_marks",   100)),
        a2_weight_pct    = float(data.get("a2_weight_pct",   15)),
        final_weight_pct = float(data.get("final_weight_pct",50)),
        pass_mark_pct    = float(data.get("pass_mark_pct",   50)),
    )
    if "is_active" in data:
        set_unit_active(unit["unit_id"], bool(data["is_active"]))
    return jsonify({"status": "updated"})


@main.route("/api/units/<unit_code>", methods=["DELETE"])
@login_required(role="staff")
def api_delete_unit(unit_code):
    delete_unit(unit_code)
    return jsonify({"status": "deleted"})


@main.route("/api/units/<unit_code>/activate", methods=["POST"])
@login_required(role="staff")
def api_activate_unit(unit_code):
    """Toggle a unit's active state. Students can only see active units."""
    data = request.get_json(force=True)
    unit = get_unit_by_code(unit_code)
    if not unit:
        return jsonify({"status": "error", "message": "Unit not found"}), 404
    is_active = bool(data.get("is_active", True))
    set_unit_active(unit["unit_id"], is_active)
    state = "activated" if is_active else "deactivated"
    return jsonify({"status": "ok", "message": f"{unit_code} {state}"})


@main.route("/api/units/<unit_code>/students", methods=["GET"])
@login_required()
def api_get_students(unit_code):
    unit = get_unit_by_code(unit_code)
    if not unit:
        return jsonify([])
    cohort = get_full_cohort_progress(unit["unit_id"])
    return jsonify(cohort)


# =============================================================================
# STUDENT MANAGEMENT API - Staff only
# =============================================================================

@main.route("/api/student/add", methods=["POST"])
@login_required(role="staff")
def api_add_student():
    """
    Add a new student user and optionally enrol them in a unit.
    Used by the 'Add Student' modal in the Staff Portal.

    Expected JSON body:
        student_id  (str):  e.g. "S1020"
        full_name   (str):  e.g. "Jane Smith"
        email       (str):  e.g. "s1020@student.murdoch.edu.au"
        password    (str):  default "student123"
        unit_code   (str):  unit to enrol them in (optional)
    """
    data = request.get_json(force=True)
    student_id = data.get("student_id", "").strip().upper()
    full_name  = data.get("full_name", "").strip()
    email      = data.get("email", "").strip().lower()
    password   = data.get("password", "student123").strip()
    unit_code  = data.get("unit_code", "").strip().upper()

    if not student_id or not full_name or not email:
        return jsonify({"status": "error",
                        "message": "student_id, full_name, and email are required"}), 400

    conn   = get_connection()
    cursor = conn.cursor()
    try:
        # Insert user - ignore if already exists (idempotent)
        cursor.execute("""
            INSERT OR IGNORE INTO users (student_id, full_name, email, password, role)
            VALUES (?, ?, ?, ?, 'student')
        """, (student_id, full_name, email, password))
        conn.commit()

        # Get user_id (works whether just inserted or already existed)
        cursor.execute("SELECT user_id FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"status": "error", "message": "Could not create user"}), 500
        user_id = row["user_id"]

        # Optionally enrol in a unit
        if unit_code:
            unit = get_unit_by_code(unit_code)
            if unit:
                cursor.execute("""
                    INSERT OR IGNORE INTO enrollments (user_id, unit_id) VALUES (?, ?)
                """, (user_id, unit["unit_id"]))
                conn.commit()

        return jsonify({"status": "ok", "user_id": user_id,
                        "message": f"Student {student_id} added successfully."})
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 400
    finally:
        conn.close()


# =============================================================================
# STUDENT API - Student only
# =============================================================================

@main.route("/api/available-units", methods=["GET"])
@login_required(role="student")
def api_get_available_units():
    user = current_user()
    rows = get_available_units_for_student(user["id"])
    return jsonify([dict(r) for r in rows])


@main.route("/api/enrol/<unit_code>", methods=["POST"])
@login_required(role="student")
def api_enrol_unit(unit_code):
    user = current_user()
    try:
        enroll_student_by_unit_code(user["id"], unit_code)
        return jsonify({"status": "ok", "message": f"Enrolled in {unit_code}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@main.route("/api/student/dashboard", methods=["GET"])
@login_required(role="student")
def api_student_dashboard():
    user = current_user()
    data = get_student_dashboard_data(user["id"])
    return jsonify(data)


@main.route("/api/student/unit/<unit_code>/grades", methods=["POST"])
@login_required(role="student")
def api_save_student_grades(unit_code):
    user = current_user()
    unit = get_unit_by_code(unit_code)
    if not unit:
        return jsonify({"status": "error", "message": "Unit not found"}), 404

    unit_id = unit["unit_id"]
    data    = request.get_json(force=True)

    # Save each quiz score in sequence
    quizzes = data.get("quizzes", [])
    for score in quizzes:
        next_quiz = get_next_quiz_number(user["id"], unit_id)
        if next_quiz is not None:
            save_quiz_score(user["id"], unit_id, next_quiz, float(score))

    if data.get("assignment_1") is not None:
        save_assignment_score(user["id"], unit_id, 1, float(data["assignment_1"]))

    if data.get("assignment_2") is not None:
        save_assignment_score(user["id"], unit_id, 2, float(data["assignment_2"]))

    return jsonify({"status": "saved"})

@main.route('/api/student/my-report-pdf/<unitcode>', methods=['GET'])
@login_required(role='student')
def api_student_my_report_pdf(unitcode):
    import datetime
    from database.models import (
        get_quiz_scores_list,
        get_assignment_scores,
        get_risk_result,
        get_unit_config
    )

    user = current_user()
    uid = user['id']

    unit = get_unit_by_code(unitcode.strip().upper())
    if not unit:
        return jsonify({'status': 'error', 'message': 'Unit not found'}), 404

    unit_id = unit['unit_id']
    cfg = get_unit_config(unit_id)
    quizzes = get_quiz_scores_list(uid, unit_id)
    assigns = get_assignment_scores(uid, unit_id)
    rr = get_risk_result(uid, unit_id)
    risk = dict(rr) if rr else {}
    ml_weight = float(risk.get("ml_weight_used", 0.55))
    engine_weight = float(risk.get("engine_weight_used", 0.45))
    medium_threshold = float(risk.get("medium_threshold_used", 0.35))
    high_threshold = float(risk.get("high_threshold_used", 0.55))

    qmax = cfg['quiz_max_marks'] if cfg else 20
    qweight = cfg['quiz_weight_pct'] if cfg else 20
    a1max = cfg['a1_max_marks'] if cfg else 100
    a1weight = cfg['a1_weight_pct'] if cfg else 15
    a2max = cfg['a2_max_marks'] if cfg else 100
    a2weight = cfg['a2_weight_pct'] if cfg else 15
    finalw = cfg['final_weight_pct'] if cfg else 50
    passmark = cfg['pass_mark_pct'] if cfg else 50
    qcount = cfg['quiz_count'] if cfg else 10

    ml_raw = risk.get('ml_probability')
    ml_pct = f'{ml_raw * 100:.1f}' if ml_raw is not None else 'N/A'

    comb_raw = risk.get('combined_risk_score')
    comb_pct = f'{comb_raw * 100:.1f}' if comb_raw is not None else 'N/A'

    risk_lv = risk.get('risk_level') or 'NOT RUN'
    trend = risk.get('trend_direction') or '-'
    q_avg = risk.get('quiz_avg_pct')
    q_avg_s = f'{q_avg:.1f}%' if q_avg is not None else 'N/A'
    proj = risk.get('projected_avg_pct')
    proj_s = f'{proj:.1f}%' if proj is not None else 'N/A'
    model_u = risk.get('selected_model') or 'Random Forest'
    now = datetime.datetime.now().strftime('%d %B %Y')

    sid = user['student_id']
    sname = user['name']
    semail = user['email']

    a1 = assigns.get(1)
    a2 = assigns.get(2)

    risk_desc = {
        'HIGH': 'You are at HIGH risk of failing this unit. Please contact your unit coordinator immediately.',
        'MEDIUM': 'You are at MEDIUM risk. Increase your study effort and seek support soon.',
        'LOW': 'You are at LOW risk. You are on track - keep up the good work!',
        'NOT RUN': 'Your risk has not been calculated yet. Your coordinator needs to run the AI model first.',
    }.get(risk_lv, 'Please contact your unit coordinator for guidance.')

    recs = {
        'HIGH': [
            'Contact your unit coordinator immediately to discuss your academic standing.',
            'Visit Murdoch University Student Support Services for academic help.',
            'Submit all outstanding assessments as soon as possible.',
            'Form or join a study group to review quiz and assignment material.',
            'Create a personal study plan and stick to it for the remaining weeks.',
        ],
        'MEDIUM': [
            'Increase your weekly study hours dedicated to this unit.',
            'Review topics from quizzes where you scored below 60%.',
            'Ensure all upcoming assignments are submitted on time.',
            "Visit your tutor's consultation hours for targeted feedback.",
        ],
        'LOW': [
            'Keep up your current study routine - you are on track!',
            'Review any quiz topics you found difficult to stay strong.',
            'Stay consistent through the final assessment period.',
        ],
    }.get(risk_lv, ['Contact your unit coordinator to get your risk assessment run.'])

    lines = [
        f"Student ID: {sid}",
        f"Full Name: {sname}",
        f"Email: {semail}",
        f"Unit Code: {unit['unit_code']}",
        f"Unit Name: {unit['unit_name']}",
        f"Trimester: {unit['trimester']}",
        f"Pass Mark: {passmark}%",
        f"Report Date: {now}",
        "",
        "MY AI RISK ASSESSMENT",
        f"Risk Level: {risk_lv}",
        f"Combined Risk Score: {comb_pct}%",
        f"ML Probability: {ml_pct}%",
        f"Quiz Average: {q_avg_s}",
        f"Projected Average: {proj_s}",
        f"Trend Direction: {trend}",
        f"AI Model Used: {model_u}",
        "",
        risk_desc,
        "",
        "MY ASSESSMENT SCORE BREAKDOWN",
    ]

    for i, sc in enumerate(quizzes, start=1):
        pct = float(sc) / float(qmax) * 100.0 if qmax else 0.0
        lines.append(f"Quiz {i}: {float(sc):.1f}/{float(qmax):.1f} ({pct:.1f}%)")

    if quizzes:
        quiz_total_raw = sum(quizzes)
        quiz_total_max = qmax * len(quizzes)
        quiz_pct = (quiz_total_raw / quiz_total_max * 100) if quiz_total_max else 0
        quiz_weighted = quiz_pct * qweight / 100
        lines.append(f"Quiz Average ({len(quizzes)}/{qcount} submitted): {quiz_pct:.1f}% | Weighted: {quiz_weighted:.1f}")
    else:
        lines.append("Quiz Average: No quiz submissions yet")

    if a1 is not None:
        a1_pct = (a1 / a1max * 100) if a1max else 0
        a1_weighted = a1_pct * a1weight / 100
        lines.append(f"Assignment 1: {float(a1):.1f}/{float(a1max):.1f} ({a1_pct:.1f}%) | Weighted: {a1_weighted:.1f}")
    else:
        lines.append(f"Assignment 1: Not submitted | Weight: {a1weight}%")

    if a2 is not None:
        a2_pct = (a2 / a2max * 100) if a2max else 0
        a2_weighted = a2_pct * a2weight / 100
        lines.append(f"Assignment 2: {float(a2):.1f}/{float(a2max):.1f} ({a2_pct:.1f}%) | Weighted: {a2_weighted:.1f}")
    else:
        lines.append(f"Assignment 2: Not submitted | Weight: {a2weight}%")

    lines.extend([
        "",
        "HOW MY RISK SCORE WAS CALCULATED",
        (
            "EduGuardian AI uses a hybrid risk pipeline. First, a deterministic risk engine "
            "analyses your quiz trend and assignment performance to produce an interpretable "
            "academic risk score. Second, a trained machine learning model estimates your "
            "at-risk probability using the same academic feature set. These two signals are "
            "then combined using the hybrid configuration selected by staff when the model was run."
        ),
        "",
        (
            f"Hybrid Configuration Used: ML Weight {ml_weight:.2f} | "
            f"Deterministic Weight {engine_weight:.2f}"
        ),
        (
            f"Risk Thresholds Used: LOW < {medium_threshold:.2f} | "
            f"MEDIUM {medium_threshold:.2f} to {high_threshold:.2f} | "
            f"HIGH >= {high_threshold:.2f}"
        ),
        "",
        (
            f"The ML model ({model_u}) estimated an at-risk probability of {ml_pct}%. "
            f"The final combined risk score for this report is {comb_pct}%."
        ),
        "",
        "RECOMMENDED ACTIONS FOR YOU",
    ])

    for rec in recs:
        lines.append(f"- {rec}")

    lines.extend([
        "",
        "This report was generated by EduGuardian AI.",
        f"Report date: {now}",
        "These results are generated using a hybrid AI model combining a trained machine learning model and a deterministic rule-based engine.",
        "The final risk classification may vary depending on the configuration settings selected during the model run.",
        "These are AI-assisted estimates and should be used for academic decision support only.",
        "Murdoch University.",
    ])

    return build_pdf_response(
        f'EduGuardian_{sid}_{unitcode}_MyReport.pdf',
        f'My Risk Report - {sid} | {unitcode}',
        lines
    )

# =============================================================================
# AI MODEL API - Staff only
# =============================================================================

@main.route("/api/predict", methods=["POST"])
@login_required(role="staff")
def api_run_model():
    """
    Run the hybrid AI prediction pipeline for all students in a unit.

    The staff portal may optionally send runtime hybrid configuration values:
      - ml_weight
      - engine_weight
      - medium_threshold
      - high_threshold

    If these are not provided, predict.py will fall back to the validated defaults.
    """
    data = request.get_json(force=True)
    unit_code = (data.get("unit_code") or "").strip().upper()

    if not unit_code:
        return jsonify({"status": "error", "message": "unit_code required"}), 400

    hybrid_config = {
        "ml_weight": data.get("ml_weight", 0.55),
        "engine_weight": data.get("engine_weight", 0.45),
        "medium_threshold": data.get("medium_threshold", 0.35),
        "high_threshold": data.get("high_threshold", 0.55),
    }

    try:
        result = run_prediction_for_unit(unit_code, hybrid_config=hybrid_config)
        summary = result.get("summary", {})

        return jsonify({
            "status": "ok",
            "high": summary.get("HIGH", 0),
            "medium": summary.get("MEDIUM", 0),
            "low": summary.get("LOW", 0),
            "total": summary.get("total", 0),
            **result
        })
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@main.route("/api/unit/<unit_code>/cohort", methods=["GET"])
@login_required()
def api_get_unit_cohort(unit_code):
    """Return full cohort progress + risk summary for a unit."""
    unit = get_unit_by_code(unit_code)
    if not unit:
        return jsonify({"summary": {}, "cohort": []})
    cohort  = get_full_cohort_progress(unit["unit_id"])
    summary = get_risk_summary_for_unit(unit["unit_id"])
    return jsonify({"summary": summary, "cohort": cohort})


@main.route("/api/predict/save-reports", methods=["POST"])
@login_required(role="staff")
def api_save_reports():
    """
    Save JSON audit reports for all students in a unit to the reports/ folder.
    Called when staff clicks 'Export JSON' - reports are NOT saved automatically
    on model run, only saved on demand via this endpoint.
    """
    data = request.get_json(force=True)
    unit_code = (data.get("unit_code") or "").strip().upper()

    if not unit_code:
        return jsonify({"status": "error", "message": "unit_code required"}), 400

    hybrid_config = {
        "ml_weight": data.get("ml_weight", 0.55),
        "engine_weight": data.get("engine_weight", 0.45),
        "medium_threshold": data.get("medium_threshold", 0.35),
        "high_threshold": data.get("high_threshold", 0.55),
    }

    try:
        from .services import run_prediction_for_unit_with_reports
        result = run_prediction_for_unit_with_reports(
            unit_code,
            hybrid_config=hybrid_config
        )
        return jsonify({"status": "ok", "saved": result.get("saved", 0)})
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@main.route("/api/grades/import", methods=["POST"])
@login_required(role="staff")
def api_import_grades():
    """
    Bulk-import grades for a student into a unit.
    Called by the Import Grades CSV feature in the Staff Portal.

    Expected JSON:
        student_id   (str):        e.g. "S1000"
        unit_code    (str):        e.g. "ICT304"
        quizzes      (list[float]): list of quiz scores in order
        assignment_1 (float|None)
        assignment_2 (float|None)
    """
    data       = request.get_json(force=True)
    student_id = data.get("student_id", "").strip().upper()
    unit_code  = data.get("unit_code",  "").strip().upper()
    quizzes    = data.get("quizzes",    [])
    a1         = data.get("assignment_1")
    a2         = data.get("assignment_2")

    if not student_id or not unit_code:
        return jsonify({"status": "error", "message": "student_id and unit_code required"}), 400

    conn   = get_connection()
    cursor = conn.cursor()
    try:
        # Get user_id from student_id
        cursor.execute("SELECT user_id FROM users WHERE student_id = ?", (student_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"status": "error",
                            "message": f"Student {student_id} not found in DB"}), 404
        user_id = row["user_id"]

        # Get unit_id
        unit = get_unit_by_code(unit_code)
        if not unit:
            return jsonify({"status": "error", "message": f"Unit {unit_code} not found"}), 404
        unit_id = unit["unit_id"]

        # Save each quiz score
        for i, score in enumerate(quizzes):
            quiz_number = i + 1
            cursor.execute("""
                INSERT INTO quiz_submissions (user_id, unit_id, quiz_number, score)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, unit_id, quiz_number)
                DO UPDATE SET score = excluded.score
            """, (user_id, unit_id, quiz_number, float(score)))

        # Save assignments
        if a1 is not None:
            cursor.execute("""
                INSERT INTO assignment_submissions (user_id, unit_id, assignment_number, score)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(user_id, unit_id, assignment_number)
                DO UPDATE SET score = excluded.score
            """, (user_id, unit_id, float(a1)))

        if a2 is not None:
            cursor.execute("""
                INSERT INTO assignment_submissions (user_id, unit_id, assignment_number, score)
                VALUES (?, ?, 2, ?)
                ON CONFLICT(user_id, unit_id, assignment_number)
                DO UPDATE SET score = excluded.score
            """, (user_id, unit_id, float(a2)))

        conn.commit()
        return jsonify({"status": "ok", "student_id": student_id})

    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()



@main.route("/api/staff/grades/save", methods=["POST"])
@login_required(role="staff")
def api_staff_save_grades():
    """Staff enters grades for a specific student in a unit."""
    data       = request.get_json(force=True)
    student_id = data.get("student_id","").strip().upper()
    unit_code  = data.get("unit_code","").strip().upper()
    quizzes    = data.get("quizzes", [])   # list of {number, score}
    a1         = data.get("assignment_1")
    a2         = data.get("assignment_2")

    unit = get_unit_by_code(unit_code)
    if not unit:
        return jsonify({"status":"error","message":"Unit not found"}), 404

    conn   = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_id FROM users WHERE student_id=?", (student_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"status":"error","message":f"Student {student_id} not found"}), 404
        uid     = row["user_id"]
        unit_id = unit["unit_id"]

        for q in quizzes:
            num   = int(q.get("number", 0))
            score = float(q.get("score", 0))
            if num > 0:
                cursor.execute("""
                    INSERT INTO quiz_submissions (user_id, unit_id, quiz_number, score)
                    VALUES (?,?,?,?)
                    ON CONFLICT(user_id, unit_id, quiz_number)
                    DO UPDATE SET score=excluded.score
                """, (uid, unit_id, num, score))

        if a1 is not None:
            cursor.execute("""
                INSERT INTO assignment_submissions (user_id, unit_id, assignment_number, score)
                VALUES (?,?,1,?)
                ON CONFLICT(user_id, unit_id, assignment_number)
                DO UPDATE SET score=excluded.score
            """, (uid, unit_id, float(a1)))

        if a2 is not None:
            cursor.execute("""
                INSERT INTO assignment_submissions (user_id, unit_id, assignment_number, score)
                VALUES (?,?,2,?)
                ON CONFLICT(user_id, unit_id, assignment_number)
                DO UPDATE SET score=excluded.score
            """, (uid, unit_id, float(a2)))

        conn.commit()
        return jsonify({"status":"ok"})
    except Exception as e:
        conn.rollback()
        return jsonify({"status":"error","message":str(e)}), 500
    finally:
        conn.close()


@main.route("/api/unenrol", methods=["POST"])
@login_required(role="staff")
def api_unenrol_student():
    """Remove a student from a unit."""
    data       = request.get_json(force=True)
    student_id = data.get("student_id","").strip().upper()
    unit_code  = data.get("unit_code","").strip().upper()

    unit = get_unit_by_code(unit_code)
    if not unit:
        return jsonify({"status":"error","message":"Unit not found"}), 404

    conn   = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_id FROM users WHERE student_id=?", (student_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"status":"error","message":"Student not found"}), 404
        uid     = row["user_id"]
        unit_id = unit["unit_id"]
        cursor.execute("DELETE FROM enrollments WHERE user_id=? AND unit_id=?", (uid, unit_id))
        conn.commit()
        return jsonify({"status":"ok"})
    except Exception as e:
        conn.rollback()
        return jsonify({"status":"error","message":str(e)}), 500
    finally:
        conn.close()


@main.route("/api/student/profile/<student_id>/<unit_code>", methods=["GET"])
@login_required(role="staff")
def api_student_profile(student_id, unit_code):
    """Full profile for a student in a unit - grades, risk, explanation."""
    unit = get_unit_by_code(unit_code)
    if not unit:
        return jsonify({"status":"error","message":"Unit not found"}), 404

    conn   = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE student_id=?",
                       (student_id.strip().upper(),))
        user = cursor.fetchone()
        if not user:
            return jsonify({"status":"error","message":"Student not found"}), 404

        uid     = user["user_id"]
        unit_id = unit["unit_id"]

        from database.models import (get_quiz_scores_list, get_assignment_scores,
                                      get_risk_result, get_unit_config)
        cfg      = get_unit_config(unit_id)
        quizzes  = get_quiz_scores_list(uid, unit_id)
        assigns  = get_assignment_scores(uid, unit_id)
        risk_row = get_risk_result(uid, unit_id)
        risk     = dict(risk_row) if risk_row else None

        return jsonify({
            "status":     "ok",
            "student_id": user["student_id"],
            "full_name":  user["full_name"],
            "email":      user["email"],
            "unit_code":  unit["unit_code"],
            "unit_name":  unit["unit_name"],
            "config": {
                "quiz_count":      cfg["quiz_count"]      if cfg else 10,
                "quiz_max_marks":  cfg["quiz_max_marks"]  if cfg else 20,
                "quiz_weight_pct": cfg["quiz_weight_pct"] if cfg else 20,
                "a1_max_marks":    cfg["a1_max_marks"]    if cfg else 100,
                "a1_weight_pct":   cfg["a1_weight_pct"]   if cfg else 15,
                "a2_max_marks":    cfg["a2_max_marks"]    if cfg else 100,
                "a2_weight_pct":   cfg["a2_weight_pct"]   if cfg else 15,
                "final_weight_pct":cfg["final_weight_pct"]if cfg else 50,
                "pass_mark_pct":   cfg["pass_mark_pct"]   if cfg else 50,
            },
            "quiz_scores":   quizzes,
            "assignment_1":  assigns.get(1),
            "assignment_2":  assigns.get(2),
            "risk":          risk,
        })
    finally:
        conn.close()


def build_pdf_response(filename: str, title: str, lines: list[str]):
    try:
        from fpdf import FPDF  # type: ignore

        NAVY   = (15, 40, 80)
        GREY   = (90, 95, 105)
        LGREY  = (150, 155, 165)
        DIVIDER= (210, 213, 218)
        BG     = (247, 248, 250)
        WHITE  = (255, 255, 255)
        BLACK  = (30, 30, 35)
        HIGH_C = (180, 30, 30)
        MED_C  = (160, 100, 0)
        LOW_C  = (20, 120, 60)

        def safe(value):
            text = '' if value is None else str(value)
            for src, dst in [('\u2013','-'),('\u2014','-'),('\u2018',"'"),('\u2019',"'"),
                             ('\u201c','"'),('\u201d','"'),('\u2022','-'),('\u2265','>='),('\u2264','<=')]:
                text = text.replace(src, dst)
            return text.encode('latin-1', 'replace').decode('latin-1')

        def risk_color(level):
            l = (level or '').upper()
            if l == 'HIGH':   return HIGH_C
            if l == 'MEDIUM': return MED_C
            if l == 'LOW':    return LOW_C
            return GREY

        # -- Header height constant - MUST match what header() draws ------
        HEADER_H = 16  # mm - navy bar height

        class PDF(FPDF):
            def header(self):
                self.set_fill_color(*NAVY)
                self.rect(0, 0, 210, HEADER_H, 'F')
                self.set_y(3)
                self.set_font('Helvetica', 'B', 10)
                self.set_text_color(*WHITE)
                self.cell(0, 5, safe('EduGuardian AI   |   Murdoch University   |   ICT304 - T1 2026'), align='C')
                self.set_y(9)
                self.set_font('Helvetica', '', 7)
                self.set_text_color(180, 190, 210)
                self.cell(0, 4, safe('AI-Powered Student Academic Risk Prediction System'), align='C')
                # Reset to just below the header band - all content starts here
                self.set_y(HEADER_H + 4)
                self.set_text_color(*BLACK)

            def footer(self):
                self.set_y(-12)
                self.set_draw_color(*DIVIDER)
                self.set_line_width(0.3)
                self.line(15, self.get_y(), 195, self.get_y())
                self.ln(1)
                self.set_font('Helvetica', '', 7)
                self.set_text_color(*LGREY)
                self.cell(120, 4, safe('EduGuardian AI - Murdoch University Dubai - ICT304'))
                self.cell(0, 4, safe(f'Page {self.page_no()}'), align='R')

        pdf = PDF()
        # top_margin must be > HEADER_H so body text never overlaps the header
        TOP_MARGIN = HEADER_H + 6
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.set_margins(15, TOP_MARGIN, 15)
        pdf.add_page()
        W = 180  # usable width

        # -- Report title (sits below header band, no overlap) -------------
        pdf.set_font('Helvetica', 'B', 15)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 9, safe(title), align='L', ln=True)
        pdf.ln(1)
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.5)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(5)
        pdf.set_text_color(*BLACK)

        # -- Helpers -------------------------------------------------------
        def section_heading(text):
            if pdf.get_y() > 255:
                pdf.add_page()
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_text_color(*NAVY)
            pdf.set_fill_color(*BG)
            pdf.rect(15, pdf.get_y(), W, 6.5, 'F')
            pdf.set_x(16)
            pdf.cell(W - 1, 6.5, safe(text.upper()), ln=True)
            pdf.set_text_color(*BLACK)
            pdf.ln(1)

        def label_value(label, value, value_color=None):
            if pdf.get_y() > 272:
                pdf.add_page()
            pdf.set_font('Helvetica', 'B', 8.5)
            pdf.set_text_color(*GREY)
            pdf.set_x(15)
            pdf.cell(52, 6, safe(label))
            pdf.set_font('Helvetica', '', 8.5)
            pdf.set_text_color(*(value_color or BLACK))
            pdf.multi_cell(W - 52, 6, safe(value))

        def thin_divider():
            pdf.set_draw_color(*DIVIDER)
            pdf.set_line_width(0.2)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.set_draw_color(*BLACK)
            pdf.ln(1.5)

        def table_header(cols):
            pdf.set_fill_color(*NAVY)
            pdf.set_font('Helvetica', 'B', 7.5)
            pdf.set_text_color(*WHITE)
            for label, w, align in cols:
                pdf.cell(w, 7, safe(label), border=0, fill=True, align=align)
            pdf.ln(7)
            pdf.set_text_color(*BLACK)

        def table_row(vals, cols, i):
            if pdf.get_y() > 272:
                pdf.add_page()
            fill = (i % 2 == 0)
            pdf.set_fill_color(*BG) if fill else pdf.set_fill_color(*WHITE)
            pdf.set_font('Helvetica', '', 8)
            pdf.set_text_color(*BLACK)
            row_y = pdf.get_y()
            x = 15
            for val, (_, w, align) in zip(vals, cols):
                pdf.set_xy(x, row_y)
                pdf.cell(w, 6.5, safe(val), border=0, fill=True, align=align)
                x += w
            pdf.ln(6.5)

        # -- Parse lines ---------------------------------------------------
        quiz_rows    = []
        student_rows = []
        in_students  = False
        info_block   = []
        risk_level_found = None

        for line in lines:
            stripped = (line or '').strip()
            if not stripped:
                continue
            if stripped == 'Student Details':
                in_students = True
                continue
            if in_students:
                # stop collecting student rows on footer lines
                if stripped.startswith('Generated by') or stripped.startswith('CONFIDENTIAL'):
                    in_students = False
                    # fall through to render as plain note below
                else:
                    student_rows.append(stripped)
                    continue
            if stripped.startswith('Quiz ') and ':' in stripped:
                quiz_rows.append(stripped)
                continue
            if ':' in stripped:
                key, _, val = stripped.partition(':')
                key = key.strip(); val = val.strip()
                if key == 'Risk Level':
                    risk_level_found = val
                info_block.append((key, val))

        # -- Categorise info_block -----------------------------------------
        student_info_keys = {'Student ID', 'Name', 'Email', 'Unit', 'Trimester', 'Generated', 'Pass Mark'}
        risk_keys         = {'Risk Level', 'Combined Score', 'ML Probability', 'Trend',
                             'Quiz Average', 'Projected Average'}
        assign_keys       = {'Assignment 1', 'Assignment 2'}
        dist_keys         = {'HIGH', 'MEDIUM', 'LOW', 'NO DATA'}
        skip_keys         = {'Quizzes Submitted', 'Assessment Weights',
                             'Total Students', 'Risk Level   SEE BELOW'}

        student_info  = [(k, v) for k, v in info_block if k in student_info_keys]
        risk_info     = [(k, v) for k, v in info_block if k in risk_keys]
        assign_info   = [(k, v) for k, v in info_block if k in assign_keys]
        dist_info     = [(k, v) for k, v in info_block if k in dist_keys]
        unit_meta     = [(k, v) for k, v in info_block
                         if k not in student_info_keys | risk_keys | assign_keys | dist_keys | skip_keys
                         and not k.startswith('Risk Level')]
        plain_lines   = [v for k, v in info_block if k in skip_keys or k.startswith('Assessment Weights')]

        # -- Render: Report info -------------------------------------------
        if student_info:
            section_heading('Report Information')
            for lbl, val in student_info:
                label_value(lbl, val)
                thin_divider()

        if unit_meta:
            section_heading('Unit Information')
            for lbl, val in unit_meta:
                label_value(lbl, val)
                thin_divider()

        # -- Render: plain meta lines (weights, totals) --------------------
        for lbl, val in info_block:
            if lbl == 'Assessment Weights':
                section_heading('Assessment Weighting')
                label_value('Weights', val)
                thin_divider()
                pdf.ln(1)
            if lbl == 'Total Students':
                label_value('Total Students', val)
                thin_divider()

        # -- Render: Risk distribution (cohort) ----------------------------
        if dist_info:
            section_heading('Risk Distribution')
            for k, v in dist_info:
                label_value(k + ' Risk', v + ' student(s)', risk_color(k))
                thin_divider()

        # -- Render: AI result (student report) ----------------------------
        if risk_info:
            section_heading('AI Risk Assessment Result')
            for lbl, val in risk_info:
                vc = risk_color(val) if lbl == 'Risk Level' else None
                label_value(lbl, val, vc)
                thin_divider()

        # -- Render: How the prediction was made ---------------------------
        # Triggered for student reports (has risk_info) AND cohort reports
        # (which embed the explanation text directly in lines)
        explanation_lines = []
        in_explanation = False
        for line in lines:
            stripped = (line or '').strip()
            if stripped.startswith('EduGuardian AI uses a three-stage'):
                in_explanation = True
            if in_explanation:
                if stripped == 'Student Details':
                    break
                explanation_lines.append(stripped)

        if risk_info or explanation_lines:
            section_heading('How This Prediction Was Made')
            if explanation_lines:
                # cohort - use the embedded explanation
                pdf.set_font('Helvetica', '', 8.5)
                pdf.set_text_color(*GREY)
                for el in explanation_lines:
                    if pdf.get_y() > 272:
                        pdf.add_page()
                    pdf.set_x(15)
                    pdf.multi_cell(W, 5, safe(el))
            else:
                # student - render inline explanation
                pdf.set_font('Helvetica', '', 8.5)
                pdf.set_text_color(*GREY)
                for el in [
                    'EduGuardian AI uses a three-stage hybrid pipeline to determine each student\'s risk level:',
                    '',
                    '  Stage 1 - Quiz Performance:  The average of all submitted quiz scores (as a % of max',
                    '  mark) is calculated. A declining trend applies an additional risk penalty.',
                    '',
                    '  Stage 2 - Assignment Performance:  Assignment 1 and Assignment 2 scores are converted',
                    '  to percentages and weighted by their configured unit contribution.',
                    '',
                    '  Stage 3 - ML Ensemble Classifier:  Three models (Logistic Regression, Random Forest,',
                    '  XGBoost) trained on 5 SMOTE-balanced Kaggle datasets produce a probability of failure.',
                    '  This ML Probability is the primary driver of the final risk classification.',
                    '',
                    '  Final Classification is based on the hybrid configuration used during the model run.',
                    '  The configured thresholds stored with this result determine whether the final score',
                    '  is classified as HIGH, MEDIUM, or LOW.',
                    '  Please refer to the Hybrid Configuration and Thresholds shown earlier in this report.',
                ]:
                    if pdf.get_y() > 272:
                        pdf.add_page()
                    pdf.set_x(15)
                    pdf.multi_cell(W, 5, safe(el))
            pdf.ln(2)

        # -- Render: Quiz table --------------------------------------------
        if quiz_rows:
            section_heading('Quiz Scores')
            cols = [('Quiz', 18, 'C'), ('Score', 28, 'C'), ('Out Of', 28, 'C'),
                    ('Percentage', 32, 'C'), ('Performance', 74, 'L')]
            table_header(cols)
            for i, qline in enumerate(quiz_rows):
                try:
                    num_part  = qline.split(':')[0].replace('Quiz', '').strip()
                    score_raw = qline.split(':')[1].strip()
                    raw       = score_raw.split('/')[0].strip()
                    rest      = score_raw.split('/')[1].strip()
                    mx        = rest.split('(')[0].strip()
                    pct_str   = rest.split('(')[1].replace(')', '').replace('%', '').strip()
                    pct       = float(pct_str)
                    perf      = ('Excellent' if pct >= 80 else
                                 'Good'      if pct >= 65 else
                                 'Pass'      if pct >= 50 else 'Below Pass')
                    table_row([num_part, raw, mx, f'{pct:.1f}%', perf], cols, i)
                except Exception:
                    pdf.set_x(15)
                    pdf.set_font('Helvetica', '', 8)
                    pdf.cell(0, 6, safe(qline), ln=True)
            pdf.ln(2)

        # -- Render: Assignment scores -------------------------------------
        if assign_info:
            section_heading('Assignment Scores')
            for lbl, val in assign_info:
                label_value(lbl, val)
                thin_divider()

        # -- Render: Student cohort table ----------------------------------
        if student_rows:
            section_heading('Student Cohort Overview')
            cols = [('Student ID', 22, 'L'), ('Name', 38, 'L'), ('Quizzes', 20, 'C'),
                    ('Quiz Avg', 20, 'C'), ('A1', 16, 'C'), ('A2', 16, 'C'),
                    ('Trend', 16, 'C'), ('Combined', 18, 'C'), ('ML Prob', 14, 'C')]
            table_header(cols)
            for i, sline in enumerate(student_rows):
                try:
                    parts = sline.split(',', 2)
                    sid   = parts[0].strip() if len(parts) > 0 else '-'
                    sname = parts[1].strip()[:16] if len(parts) > 1 else '-'
                    rest  = parts[2].strip() if len(parts) > 2 else sline

                    def extract(tag, nxt):
                        if tag not in rest: return '-'
                        chunk = rest.split(tag, 1)[1].strip()
                        for n in nxt:
                            if n in chunk:
                                chunk = chunk.split(n)[0].strip()
                                break
                        return chunk.split()[0] if chunk else '-'

                    tags_order = ['Quizzes','Avg','A1','A2','Trend','Risk','ML','Combined Score']
                    qz   = extract('Quizzes',        ['Avg','A1','A2','Trend','Risk','ML','Combined Score'])
                    avg  = extract('Avg',             ['A1','A2','Trend','Risk','ML','Combined Score'])
                    a1v  = extract('A1',              ['A2','Trend','Risk','ML','Combined Score'])
                    a2v  = extract('A2',              ['Trend','Risk','ML','Combined Score'])
                    trnd = extract('Trend',           ['Risk','ML','Combined Score'])
                    comb = extract('Combined Score',  [])
                    ml   = extract('ML',              ['Combined Score'])

                    avg_disp  = (avg  + '%') if avg  not in ('-', 'N/A') else avg
                    comb_disp = comb  if comb in ('-', 'N/A') else comb

                    table_row([sid, sname, qz, avg_disp, a1v, a2v, trnd, comb_disp, ml],
                              cols, i)
                except Exception:
                    pdf.set_x(15)
                    pdf.set_font('Helvetica', '', 7.5)
                    pdf.cell(0, 6, safe(sline[:110]), ln=True)
            pdf.ln(2)

        # -- Render: Recommendations ---------------------------------------
        if risk_level_found:
            recs = {
                'HIGH': [
                    'Schedule an urgent meeting with the student and their academic advisor.',
                    'Refer the student to Murdoch University Student Support Services.',
                    'Review all submitted assessments and ensure detailed feedback has been given.',
                    'Set up weekly check-ins until the next major assessment.',
                    'Create a personalised catch-up study plan with clear milestones.',
                ],
                'MEDIUM': [
                    'Increase check-in frequency - aim for fortnightly contact.',
                    'Encourage the student to attend all remaining classes and labs.',
                    'Provide targeted written feedback on their weakest quiz topics.',
                    'Remind the student of their current standing and the pass mark.',
                ],
                'LOW': [
                    'Maintain current engagement - student is performing well.',
                    'Encourage consistent study habits heading into final assessments.',
                    "Acknowledge the student's effort positively at the next interaction.",
                ],
            }.get(risk_level_found.upper(), [])
            if recs:
                section_heading('Recommended Actions')
                for j, rec in enumerate(recs, 1):
                    if pdf.get_y() > 272:
                        pdf.add_page()
                    pdf.set_font('Helvetica', 'B', 8.5)
                    pdf.set_text_color(*NAVY)
                    pdf.set_x(15)
                    pdf.cell(8, 6, f'{j}.')
                    pdf.set_font('Helvetica', '', 8.5)
                    pdf.set_text_color(*BLACK)
                    pdf.multi_cell(W - 8, 6, safe(rec))
                pdf.ln(2)

        # -- Footer note ---------------------------------------------------
        pdf.ln(4)
        pdf.set_draw_color(*DIVIDER)
        pdf.set_line_width(0.3)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(3)
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.set_text_color(*LGREY)
        pdf.multi_cell(W, 5, safe(
            'This report was automatically generated by EduGuardian AI. '
            'Risk predictions are based on statistical models and should be used as a '
            'supportive tool alongside professional academic judgement. '
            'Murdoch University - ICT304 AI System Development, T1 2026.'
        ))

        raw = pdf.output(dest='S')
        pdf_bytes = bytes(raw) if isinstance(raw, (bytes, bytearray)) else raw.encode('latin-1')
        return current_app.response_class(
            pdf_bytes,
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )

    except ModuleNotFoundError:
        try:
            from io import BytesIO
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.units import mm
            buffer = BytesIO()
            c = rl_canvas.Canvas(buffer, pagesize=A4)
            w, h = A4
            c.setFillColorRGB(15/255, 40/255, 80/255)
            c.rect(0, h - 16*mm, w, 16*mm, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1)
            c.setFont('Helvetica-Bold', 10)
            c.drawCentredString(w/2, h - 8*mm, 'EduGuardian AI  |  Murdoch University  |  ICT304 T1 2026')
            c.setFont('Helvetica', 7)
            c.drawCentredString(w/2, h - 13*mm, 'AI-Powered Student Academic Risk Prediction System')
            c.setFillColorRGB(0.12, 0.12, 0.14)
            y = h - 24*mm
            c.setFont('Helvetica-Bold', 13)
            c.drawString(15*mm, y, str(title)[:80])
            y -= 8*mm
            c.setFont('Helvetica', 8.5)
            for line in lines:
                text = '' if line is None else str(line)
                text = text.encode('ascii', 'replace').decode('ascii')
                if not text.strip():
                    y -= 3*mm
                    continue
                if ':' in text and len(text.split(':')[0]) < 30:
                    parts = text.split(':', 1)
                    c.setFont('Helvetica-Bold', 8.5)
                    c.drawString(15*mm, y, parts[0].strip() + ':')
                    c.setFont('Helvetica', 8.5)
                    c.drawString(70*mm, y, parts[1].strip()[:80])
                else:
                    c.setFont('Helvetica', 8.5)
                    c.drawString(15*mm, y, text[:100])
                y -= 6*mm
                if y < 20*mm:
                    c.showPage()
                    y = h - 20*mm
            c.save()
            pdf_bytes = buffer.getvalue()
            buffer.close()
            return current_app.response_class(
                pdf_bytes,
                mimetype='application/pdf',
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
        except ModuleNotFoundError:
            return jsonify({'status': 'error', 'message': 'PDF export requires fpdf2 or reportlab. Run: pip install fpdf2'}), 500

@main.route('/api/export/student-pdf/<studentid>/<unitcode>', methods=['GET'])
@login_required(role='staff')
def api_export_student_pdf(studentid, unitcode):
    import datetime
    from database.models import get_quiz_scores_list, get_assignment_scores, get_risk_result, get_unit_config

    unit = get_unit_by_code(unitcode)
    if not unit:
        return jsonify({'status': 'error', 'message': 'Unit not found'}), 404

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM users WHERE student_id=?', (studentid.strip().upper(),))
        user = cursor.fetchone()
        if not user:
            return jsonify({'status': 'error', 'message': 'Student not found'}), 404

        uid    = user['user_id']
        unitid = unit['unit_id']
        cfg    = get_unit_config(unitid)
        quizzes = get_quiz_scores_list(uid, unitid)
        assigns = get_assignment_scores(uid, unitid)
        rr      = get_risk_result(uid, unitid)
        risk    = dict(rr) if rr else {}

        qmax     = cfg['quiz_max_marks']   if cfg else 20
        qweight  = cfg['quiz_weight_pct']  if cfg else 20
        a1weight = cfg['a1_weight_pct']    if cfg else 15
        a2weight = cfg['a2_weight_pct']    if cfg else 15
        passmark = cfg['pass_mark_pct']    if cfg else 50

        ml_raw   = risk.get('ml_probability')
        ml_pct   = f'{ml_raw * 100:.1f}%' if ml_raw is not None else 'Not available'
        comb_raw = risk.get('combined_risk_score')
        comb_pct = f'{comb_raw * 100:.1f}%' if comb_raw is not None else 'Not available'
        risk_lvl = risk.get('risk_level') or 'NOT RUN'
        trend    = risk.get('trend_direction') or 'N/A'
        q_avg    = risk.get('quiz_avg_pct')
        proj     = risk.get('projected_avg_pct')
        model    = risk.get('selected_model') or 'Ensemble (LR + RF + XGB)'
        now      = datetime.datetime.now().strftime('%d %B %Y, %H:%M')

        lines = [
            f'Student ID:    {user["student_id"]}',
            f'Name:          {user["full_name"]}',
            f'Email:         {user["email"]}',
            f'Unit:          {unit["unit_code"]} - {unit["unit_name"]}',
            f'Trimester:     {unit["trimester"]}',
            f'Pass Mark:     {passmark}%',
            f'Generated:     {now}',
            '',
            f'Risk Level:          {risk_lvl}',
            f'Combined Score:      {comb_pct}',
            f'ML Probability:      {ml_pct}  (model: {model})',
            f'Quiz Average:        {f"{q_avg:.1f}%" if q_avg is not None else "N/A"}',
            f'Projected Average:   {f"{proj:.1f}%" if proj is not None else "N/A"}',
            f'Trend:               {trend}',
            '',
            f'Quizzes Submitted:   {len(quizzes)} of {cfg["quiz_count"] if cfg else 10}  (each worth {qmax} marks, total weight {qweight}%)',
        ]

        for i, sc in enumerate(quizzes, start=1):
            pct = (float(sc) / float(qmax) * 100.0) if qmax else 0.0
            lines.append(f'Quiz {i}: {int(sc)}/{int(qmax)} ({pct:.1f}%)')

        a1 = assigns.get(1)
        a2 = assigns.get(2)
        lines.extend([
            '',
            f'Assignment 1:  {int(a1) if a1 is not None else "Not submitted"}  (weight: {a1weight}%)',
            f'Assignment 2:  {int(a2) if a2 is not None else "Not submitted"}  (weight: {a2weight}%)',
        ])

        return build_pdf_response(
            f"EduGuardian_{user['student_id']}_{unitcode}_MyReport.pdf", 
            f"My Risk Report  -  {user['student_id']}  |  {unitcode}",     
            lines
        )
    finally:
        conn.close()


@main.route('/api/student/my-report/<unitcode>', methods=['GET'])
@login_required(role='student')
def api_student_self_pdf(unitcode):
    import datetime
    from database.models import get_quiz_scores_list, get_assignment_scores, get_risk_result, get_unit_config

    user = current_user()
    unit = get_unit_by_code(unitcode.strip().upper())
    if not unit:
        return jsonify({'status': 'error', 'message': 'Unit not found'}), 404

    conn = get_connection()
    try:
        uid    = user['id']
        unitid = unit['unit_id']

        cfg     = get_unit_config(unitid)
        quizzes = get_quiz_scores_list(uid, unitid)
        assigns = get_assignment_scores(uid, unitid)
        rr      = get_risk_result(uid, unitid)
        risk    = dict(rr) if rr else {}
        ml_weight = float(risk.get("ml_weight_used", 0.55))
        engine_weight = float(risk.get("engine_weight_used", 0.45))
        medium_threshold = float(risk.get("medium_threshold_used", 0.35))
        high_threshold = float(risk.get("high_threshold_used", 0.55))

        qmax     = cfg['quiz_max_marks']  if cfg else 20
        passmark = cfg['pass_mark_pct']   if cfg else 50

        # Pre-compute to avoid nested f-string SyntaxError on Python < 3.12
        ml_prob    = f"{risk['ml_probability'] * 100:.1f}"   if risk.get('ml_probability')     is not None else 'N/A'
        comb_score = f"{risk['combined_risk_score'] * 100:.1f}" if risk.get('combined_risk_score') is not None else 'N/A'
        quiz_avg   = f"{risk['quiz_avg_pct']:.1f}%"          if risk.get('quiz_avg_pct')        is not None else 'N/A'
        proj_avg   = f"{risk['projected_avg_pct']:.1f}%"     if risk.get('projected_avg_pct')   is not None else 'N/A'
        trend      = risk.get('trend_direction') or '-'
        risk_level = risk.get('risk_level') or 'NOT RUN'
        a1         = assigns.get(1)
        a2         = assigns.get(2)
        a1_str     = str(a1) if a1 is not None else 'Not submitted'
        a2_str     = str(a2) if a2 is not None else 'Not submitted'
        now        = datetime.datetime.now().strftime('%d %B %Y %H:%M')

        lines = [
            f"Student ID: {user['student_id']}",
            f"Name: {user['name']}",
            f"Email: {user['email']}",
            f"Unit: {unit['unit_code']} - {unit['unit_name']}",
            f"Trimester: {unit['trimester']}",
            f"Generated: {now}",
            f"Pass Mark: {passmark}%",
            '',
            f"Risk Level: {risk_level}",
            f"Combined Score: {comb_score}",
            f"ML Probability: {ml_prob}",
            f"Trend: {trend}",
            f"Quiz Average: {quiz_avg}",
            f"Projected Average: {proj_avg}",
            '',
            f"Quizzes Submitted: {len(quizzes)}",
        ]

        for i, sc in enumerate(quizzes, start=1):
            pct = float(sc) / float(qmax) * 100.0 if qmax else 0.0
            lines.append(f"Quiz {i}: {int(sc)}/{int(qmax)} ({pct:.1f}%)")

        lines += [
            '',
            f"Assignment 1: {a1_str}",
            f"Assignment 2: {a2_str}",
        ]

        return build_pdf_response(
            f"EduGuardian_{user['student_id']}_{unitcode}_MyReport.pdf",
            f"My Risk Report  -  {user['student_id']}  |  {unitcode}",
            lines
        )
    finally:
        conn.close()

@main.route('/api/export/cohort-pdf/<unitcode>', methods=['GET'])
@login_required(role='staff')
def api_export_cohort_pdf(unitcode):
    import datetime
    from database.models import get_full_cohort_progress, get_unit_config

    unit = get_unit_by_code(unitcode)
    if not unit:
        return jsonify({'status': 'error', 'message': 'Unit not found'}), 404

    unitid   = unit['unit_id']
    cohort   = get_full_cohort_progress(unitid)
    cfg      = get_unit_config(unitid)
    qcount   = cfg['quiz_count']       if cfg else 10
    qweight  = cfg['quiz_weight_pct']  if cfg else 20
    a1weight = cfg['a1_weight_pct']    if cfg else 15
    a2weight = cfg['a2_weight_pct']    if cfg else 15
    finalw   = cfg['final_weight_pct'] if cfg else 50
    passmark = cfg['pass_mark_pct']    if cfg else 50
    now      = datetime.datetime.now().strftime('%d %B %Y, %H:%M')
    cfg_source = next((s.get("risk") for s in cohort if s.get("risk")), {}) or {}
    ml_weight = float(cfg_source.get("ml_weight_used", 0.55))
    engine_weight = float(cfg_source.get("engine_weight_used", 0.45))
    medium_threshold = float(cfg_source.get("medium_threshold_used", 0.35))
    high_threshold = float(cfg_source.get("high_threshold_used", 0.55))

    high = sum(1 for s in cohort if s.get('risk') and s['risk'].get('risk_level') == 'HIGH')
    med  = sum(1 for s in cohort if s.get('risk') and s['risk'].get('risk_level') == 'MEDIUM')
    low  = sum(1 for s in cohort if s.get('risk') and s['risk'].get('risk_level') == 'LOW')
    none = len(cohort) - high - med - low

    lines = [
        f'Unit:            {unit["unit_code"]} - {unit["unit_name"]}',
        f'Trimester:       {unit["trimester"]}',
        f'Total Students:  {len(cohort)}',
        f'Pass Mark:       {passmark}%',
        f'Generated:       {now}',
        '',
        f'Assessment Weights:   Quizzes {qweight}%  |  Assignment 1 {a1weight}%  |  Assignment 2 {a2weight}%  |  Final Exam {finalw}%',
        '',
        f'HIGH:      {high}',
        f'MEDIUM:    {med}',
        f'LOW:       {low}',
        f'NO DATA:   {none}',
        '',
        # -- How the prediction was made -----------------------------------
        'Risk Level:   SEE BELOW - included per student',
        '',
        'EduGuardian AI uses a three-stage hybrid pipeline to determine each student\'s risk level:',
        '',
        '  Stage 1 - Quiz Performance:  The average of all submitted quiz scores (as a % of',
        '  max mark) is calculated. A declining trend applies an additional risk penalty.',
        '',
        '  Stage 2 - Assignment Performance:  Assignment 1 and Assignment 2 scores are',
        '  converted to percentages and weighted by their configured unit contribution.',
        '',
        '  Stage 3 - ML Ensemble Classifier:  Three models (Logistic Regression, Random',
        '  Forest, XGBoost) trained on 5 SMOTE-balanced Kaggle datasets produce a probability',
        '  of failure. This ML Probability is the primary driver of the final risk level.',
        '',
                f'  Combined Risk Score Formula:',
        f'  Score = (1 - Quiz%) x {qweight}%  +  (1 - A1%) x {a1weight}%  +',
        f'          (1 - A2%) x {a2weight}%  +  (1 - Projected%) x {finalw}%',
        '',
        f'  Hybrid Configuration Used:  ML {ml_weight:.2f}  |  Deterministic {engine_weight:.2f}',
        f'  Thresholds:  LOW < {medium_threshold * 100:.0f}   |   MEDIUM {medium_threshold * 100:.0f} to {high_threshold * 100:.0f}   |   HIGH >= {high_threshold * 100:.0f}',
        '',
        'Student Details',
    ]

    for s in cohort:
        r         = s.get('risk') or {}
        ml_raw    = r.get('ml_probability')
        comb_raw  = r.get('combined_risk_score')
        ml_str    = f'{ml_raw * 100:.1f}%'   if ml_raw   is not None else 'N/A'
        comb_str  = f'{comb_raw * 100:.1f}%' if comb_raw is not None else 'N/A'
        qavg      = r.get('quiz_avg_pct')
        qstr      = f'{qavg:.0f}' if qavg is not None else 'N/A'
        lines.append(
            f'{s.get("student_id", "-")}, '
            f'{s.get("full_name", "")}, '
            f'Quizzes {s.get("quizzes_submitted", 0)}/{qcount} '
            f'Avg {qstr} '
            f'A1 {s.get("assignment_1") if s.get("assignment_1") is not None else "-"} '
            f'A2 {s.get("assignment_2") if s.get("assignment_2") is not None else "-"} '
            f'Trend {r.get("trend_direction", "N/A")} '
            f'Risk {r.get("risk_level", "N/A")} '
            f'ML {ml_str} '
            f'Combined Score {comb_str}'
        )

    lines.extend([
        '',
        'Generated by EduGuardian AI - Murdoch University - CONFIDENTIAL',
    ])

    return build_pdf_response(
        f'EduGuardian_{unitcode}_CohortReport.pdf',
        f'Cohort Risk Report - {unitcode} | {unit["unit_name"]}',
        lines
    )

@main.route("/api/pipeline/status", methods=["GET"])
@login_required(role="staff")
def api_pipeline_status():
    """
    Return the status of the AI pipeline components.
    Tells the frontend whether the model is trained and ready,
    or needs preprocessing / training first.
    """
    import os
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path  = os.path.join(base_dir, "models", "risk_model.joblib")
    scaler_path = os.path.join(base_dir, "models", "scaler.joblib")
    data_path   = os.path.join(base_dir, "data", "processed", "ds1_train_smote.csv")

    return jsonify({
        "model_ready":      os.path.exists(model_path),
        "scaler_ready":     os.path.exists(scaler_path),
        "data_processed":   os.path.exists(data_path),
        "model_path":       model_path  if os.path.exists(model_path)  else None,
        "scaler_path":      scaler_path if os.path.exists(scaler_path) else None,
    })