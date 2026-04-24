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
# database/db.py
# ICT304 AI Academic Risk Prediction System
# Database Schema Layer - SQLite connection management and table creation
# =============================================================================

import sqlite3
import os

# Path to the SQLite database file (sits inside /database/ folder)
DB_PATH = os.path.join(os.path.dirname(__file__), "academic_system.db")


def get_connection():
    """Return a connection to the SQLite database with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # Allows column-name access: row["email"]
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """
    Create all required tables if they do not already exist.
    Safe to call multiple times - uses IF NOT EXISTS on every table.

    Tables
    ------
    users               : All system accounts (students + staff)
    units               : Configured teaching units (e.g. ICT304)
    unit_config         : Staff-controlled settings per unit (quiz count, max marks, etc.)
    enrollments         : Which students are enrolled in which unit
    quiz_submissions    : Individual quiz scores submitted by a student for a unit
    assignment_submissions : Assignment scores submitted by a student for a unit
    risk_results        : Cached risk prediction results per student per unit
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ------------------------------------------------------------------
    # TABLE: users
    # Stores all accounts - role distinguishes staff from student
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      TEXT UNIQUE,          -- e.g. S1000, S1001 (NULL for staff)
            full_name       TEXT NOT NULL,
            email           TEXT NOT NULL UNIQUE,
            password        TEXT NOT NULL,        -- Plain text for demo (hash in production)
            role            TEXT NOT NULL CHECK(role IN ('student', 'staff')),
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # ------------------------------------------------------------------
    # TABLE: units
    # A unit is a teaching course like ICT304
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS units (
            unit_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_code       TEXT NOT NULL UNIQUE, -- e.g. ICT304
            unit_name       TEXT NOT NULL,        -- e.g. AI System Development
            trimester       TEXT NOT NULL,        -- e.g. T1 2026
            created_by      INTEGER NOT NULL,     -- FK -> users.user_id (staff only)
            created_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (created_by) REFERENCES users(user_id)
        )
    """)

    # ------------------------------------------------------------------
    # TABLE: unit_config
    # Staff configures the assessment structure for a unit
    # Matches the ICT304 prototype structure exactly:
    #   Quizzes  : 10 x 20 marks  = 20% of unit
    #   Assign 1 : out of 100     = 15% of unit
    #   Assign 2 : out of 100     = 15% of unit
    #   Final    : out of 100     = 50% of unit
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS unit_config (
            config_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id             INTEGER NOT NULL UNIQUE,
            quiz_count          INTEGER NOT NULL DEFAULT 10,   -- Total quizzes in trimester
            quiz_max_marks      REAL    NOT NULL DEFAULT 20,   -- Max marks per quiz
            quiz_weight_pct     REAL    NOT NULL DEFAULT 20,   -- % of final unit mark
            a1_max_marks        REAL    NOT NULL DEFAULT 100,
            a1_weight_pct       REAL    NOT NULL DEFAULT 15,
            a2_max_marks        REAL    NOT NULL DEFAULT 100,
            a2_weight_pct       REAL    NOT NULL DEFAULT 15,
            final_weight_pct    REAL    NOT NULL DEFAULT 50,
            pass_mark_pct       REAL    NOT NULL DEFAULT 50,   -- Pass threshold
            is_active           INTEGER NOT NULL DEFAULT 0,    -- 1 = students can see & submit
            configured_at       TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (unit_id) REFERENCES units(unit_id)
        )
    """)

    # ------------------------------------------------------------------
    # TABLE: enrollments
    # Links students to units they are enrolled in
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS enrollments (
            enrollment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,   -- FK -> users.user_id (student)
            unit_id         INTEGER NOT NULL,   -- FK -> units.unit_id
            enrolled_at     TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, unit_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (unit_id) REFERENCES units(unit_id)
        )
    """)

    # ------------------------------------------------------------------
    # TABLE: quiz_submissions
    # Each row is one quiz attempt by one student in one unit
    # quiz_number: 1 to quiz_count (e.g. 1–10 for ICT304)
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_submissions (
            submission_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            unit_id         INTEGER NOT NULL,
            quiz_number     INTEGER NOT NULL,   -- Which quiz: 1, 2, 3 ... 10
            score           REAL    NOT NULL,   -- Raw score out of quiz_max_marks
            submitted_at    TEXT    DEFAULT (datetime('now')),
            UNIQUE(user_id, unit_id, quiz_number),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (unit_id) REFERENCES units(unit_id)
        )
    """)

    # ------------------------------------------------------------------
    # TABLE: assignment_submissions
    # assignment_number: 1 or 2
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assignment_submissions (
            submission_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id             INTEGER NOT NULL,
            unit_id             INTEGER NOT NULL,
            assignment_number   INTEGER NOT NULL CHECK(assignment_number IN (1, 2)),
            score               REAL    NOT NULL,
            submitted_at        TEXT    DEFAULT (datetime('now')),
            UNIQUE(user_id, unit_id, assignment_number),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (unit_id) REFERENCES units(unit_id)
        )
    """)

    # ------------------------------------------------------------------
    # TABLE: risk_results
    # Stores the last calculated risk prediction for each student per unit
    # Updated every time a student submits a quiz or assignment
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_results (
            result_id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id                 INTEGER NOT NULL,
            unit_id                 INTEGER NOT NULL,
            quiz_avg_pct            REAL,               -- Current quiz average as %
            projected_avg_pct       REAL,               -- Projected quiz average as %
            trend_direction         TEXT,               -- Improving / Stable / Declining
            trend_slope             REAL,
            a1_score                REAL,               -- NULL if not submitted
            a2_score                REAL,               -- NULL if not submitted
            combined_risk_score     REAL,               -- 0.0 to 1.0
            risk_level              TEXT CHECK(risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
            ml_probability          REAL,               -- ML model risk probability
            selected_model          TEXT,               -- Logistic Regression / Random Forest
            marks_earned            REAL,               -- Marks earned so far
            marks_available         REAL,               -- Marks from submitted assessments

            ml_weight_used          REAL DEFAULT 0.55,
            engine_weight_used      REAL DEFAULT 0.45,
            medium_threshold_used   REAL DEFAULT 0.35,
            high_threshold_used     REAL DEFAULT 0.55,

            calculated_at           TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, unit_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (unit_id) REFERENCES units(unit_id)
        )
    """)

    # ------------------------------------------------------------------
    # MIGRATION: ensure new hybrid configuration columns exist in
    # existing databases created before runtime config was added.
    # ------------------------------------------------------------------
    cursor.execute("PRAGMA table_info(risk_results)")
    existing_cols = {row["name"] for row in cursor.fetchall()}

    if "ml_weight_used" not in existing_cols:
        cursor.execute("ALTER TABLE risk_results ADD COLUMN ml_weight_used REAL DEFAULT 0.55")

    if "engine_weight_used" not in existing_cols:
        cursor.execute("ALTER TABLE risk_results ADD COLUMN engine_weight_used REAL DEFAULT 0.45")

    if "medium_threshold_used" not in existing_cols:
        cursor.execute("ALTER TABLE risk_results ADD COLUMN medium_threshold_used REAL DEFAULT 0.35")

    if "high_threshold_used" not in existing_cols:
        cursor.execute("ALTER TABLE risk_results ADD COLUMN high_threshold_used REAL DEFAULT 0.55")

    conn.commit()
    conn.close()
    print("[db.py] Database initialised successfully.")
    print(f"[db.py] Location: {DB_PATH}")


def drop_all_tables():
    """
    Drop all application tables.
    Used only during development when a full database reset is required.
    Do not call this in production.
    """
    conn = get_connection()
    cursor = conn.cursor()
    tables = [
        "risk_results",
        "assignment_submissions",
        "quiz_submissions",
        "enrollments",
        "unit_config",
        "units",
        "users"
    ]
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    conn.close()
    print("[db.py] All tables dropped.")


if __name__ == "__main__":
    init_db()
