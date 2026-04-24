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
# database/seed_db.py
# ICT304 - EduGuardian AI - Database Seeder
#
# Run before first launch to create demo data:
#     python database/seed_db.py
#
# Safe to re-run any number of times - INSERT OR IGNORE means existing
# data is never overwritten, no duplicates are created.
#
# PERSISTENCE:
#   All data (units, students, grades, risk results) persists in
#   academic_system.db between Flask restarts. Nothing is deleted when
#   you stop and restart run.py.
#
# STUDENT ACCOUNTS:
#   full_name is set to the Student ID (e.g. "S1000") - no random
#   fictional names. The portal sidebar and student table show IDs only.
#   To use real names, change full_name values below and re-run.
#   Re-running seed_db.py will NOT update existing rows (INSERT OR IGNORE).
#   To force-update names, run:
#       UPDATE users SET full_name='Real Name' WHERE student_id='S1000';
#   directly against academic_system.db using a SQLite browser.
#
# ICT304 DEFAULT UNIT:
#   ICT304 is seeded but NOT active (is_active = 0) by default.
#   Staff must activate it: Staff Portal -> My Units -> ICT304 -> Edit -> Activate.
#   You can create as many additional units as you want through the Staff Portal.
#
# MULTIPLE UNITS & ENROLLMENTS:
#   Students can be enrolled in as many activated units as they want.
#   There is no limit on units, enrollments, or model runs.
#
# MODEL RUNS:
#   Run Model as many times as you want - each run overwrites the previous
#   risk result for each student (no accumulation, always shows latest).
# =============================================================================

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.db import init_db, get_connection


# =============================================================================
# SEED DATA
# =============================================================================

# -- Staff Account -------------------------------------------------------------
STAFF_USERS = [
    {
        "student_id": None,
        "full_name":  "Staff Account",
        "email":      "staff@murdoch.edu.au",
        "password":   "staff123",
        "role":       "staff"
    }
]

# -- Student Accounts (S1000 – S1019) -----------------------------------------
# full_name = Student ID - no random fictional names.
# Student demo accounts use student IDs as display names so the interface
# consistently shows academic identifiers during demonstrations.
STUDENT_USERS = [
    {"student_id": "S1000", "full_name": "S1000", "email": "s1000@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1001", "full_name": "S1001", "email": "s1001@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1002", "full_name": "S1002", "email": "s1002@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1003", "full_name": "S1003", "email": "s1003@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1004", "full_name": "S1004", "email": "s1004@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1005", "full_name": "S1005", "email": "s1005@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1006", "full_name": "S1006", "email": "s1006@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1007", "full_name": "S1007", "email": "s1007@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1008", "full_name": "S1008", "email": "s1008@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1009", "full_name": "S1009", "email": "s1009@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1010", "full_name": "S1010", "email": "s1010@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1011", "full_name": "S1011", "email": "s1011@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1012", "full_name": "S1012", "email": "s1012@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1013", "full_name": "S1013", "email": "s1013@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1014", "full_name": "S1014", "email": "s1014@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1015", "full_name": "S1015", "email": "s1015@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1016", "full_name": "S1016", "email": "s1016@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1017", "full_name": "S1017", "email": "s1017@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1018", "full_name": "S1018", "email": "s1018@student.murdoch.edu.au", "password": "student123", "role": "student"},
    {"student_id": "S1019", "full_name": "S1019", "email": "s1019@student.murdoch.edu.au", "password": "student123", "role": "student"},
]

# -- Units ---------------------------------------------------------------------
# ICT304 is the only pre-seeded unit. Create more via Staff Portal -> Add Unit.
UNITS = [
    {
        "unit_code": "ICT304",
        "unit_name": "AI System Development",
        "trimester": "T1 2026",
    }
]

# -- Unit Configuration --------------------------------------------------------
# Quizzes  : 10 x 20 marks  -> 20% of unit
# Assign 1 : /100           -> 15%
# Assign 2 : /100           -> 15%
# Final    : /100           -> 50%
# Pass mark: 50%
# is_active = 0 -> must be activated by staff before students can see it
UNIT_CONFIGS = {
    "ICT304": {
        "quiz_count":        10,
        "quiz_max_marks":    20.0,
        "quiz_weight_pct":   20.0,
        "a1_max_marks":     100.0,
        "a1_weight_pct":     15.0,
        "a2_max_marks":     100.0,
        "a2_weight_pct":     15.0,
        "final_weight_pct":  50.0,
        "pass_mark_pct":     50.0,
        "is_active":          0,
    }
}


# =============================================================================
# SEEDING FUNCTIONS
# =============================================================================

def seed_users(cursor):
    """Insert staff and student accounts. Skips rows that already exist."""
    all_users = STAFF_USERS + STUDENT_USERS
    inserted = skipped = 0
    for u in all_users:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO users
                    (student_id, full_name, email, password, role)
                VALUES (?, ?, ?, ?, ?)
            """, (u["student_id"], u["full_name"],
                  u["email"].lower(), u["password"], u["role"]))
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [WARN] Could not insert user {u['email']}: {e}")
    print(f"  Users       - inserted: {inserted}, skipped (already exist): {skipped}")


def seed_units(cursor):
    """Insert unit records. Skips units that already exist."""
    cursor.execute("SELECT user_id FROM users WHERE role = 'staff' LIMIT 1")
    staff_row = cursor.fetchone()
    if staff_row is None:
        print("  [ERROR] No staff user found - run seed_users() first.")
        return
    staff_id = staff_row[0]
    inserted = skipped = 0
    for u in UNITS:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO units (unit_code, unit_name, trimester, created_by)
                VALUES (?, ?, ?, ?)
            """, (u["unit_code"], u["unit_name"], u["trimester"], staff_id))
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [WARN] Could not insert unit {u['unit_code']}: {e}")
    print(f"  Units       - inserted: {inserted}, skipped (already exist): {skipped}")


def seed_unit_configs(cursor):
    """Insert unit_config rows. Uses INSERT OR IGNORE - never overwrites."""
    inserted = skipped = 0
    for unit_code, cfg in UNIT_CONFIGS.items():
        cursor.execute("SELECT unit_id FROM units WHERE unit_code = ?", (unit_code,))
        unit_row = cursor.fetchone()
        if unit_row is None:
            print(f"  [WARN] Unit {unit_code} not found - skipping config.")
            continue
        unit_id = unit_row[0]
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO unit_config (
                    unit_id, quiz_count, quiz_max_marks, quiz_weight_pct,
                    a1_max_marks, a1_weight_pct,
                    a2_max_marks, a2_weight_pct,
                    final_weight_pct, pass_mark_pct, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (unit_id, cfg["quiz_count"], cfg["quiz_max_marks"],
                  cfg["quiz_weight_pct"], cfg["a1_max_marks"], cfg["a1_weight_pct"],
                  cfg["a2_max_marks"], cfg["a2_weight_pct"],
                  cfg["final_weight_pct"], cfg["pass_mark_pct"], cfg["is_active"]))
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [WARN] Could not insert config for {unit_code}: {e}")
    print(f"  Unit config - inserted: {inserted}, skipped (already exist): {skipped}")


def seed_enrollments(cursor):
    """Enrol all 20 students into ICT304. Skips existing enrolments."""
    cursor.execute("SELECT unit_id FROM units WHERE unit_code = 'ICT304'")
    unit_row = cursor.fetchone()
    if unit_row is None:
        print("  [ERROR] ICT304 unit not found - run seed_units() first.")
        return
    unit_id = unit_row[0]
    cursor.execute("SELECT user_id FROM users WHERE role = 'student' ORDER BY student_id")
    students = cursor.fetchall()
    inserted = skipped = 0
    for student in students:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO enrollments (user_id, unit_id) VALUES (?, ?)
            """, (student[0], unit_id))
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [WARN] Could not enrol user_id {student[0]}: {e}")
    print(f"  Enrollments - inserted: {inserted}, skipped (already exist): {skipped}")


# =============================================================================
# MAIN SEED RUNNER
# =============================================================================

def run_seed():
    print("=" * 60)
    print("  EduGuardian AI - Database Seeder")
    print("=" * 60)

    print("\n[1/5] Initialising database tables...")
    init_db()

    conn   = get_connection()
    cursor = conn.cursor()

    print("\n[2/5] Seeding users (1 staff + 20 students)...")
    seed_users(cursor)
    conn.commit()

    print("\n[3/5] Seeding units...")
    seed_units(cursor)
    conn.commit()

    print("\n[4/5] Seeding unit configurations...")
    seed_unit_configs(cursor)
    conn.commit()

    print("\n[5/5] Enrolling all students into ICT304...")
    seed_enrollments(cursor)
    conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print("  Seed complete.")
    print("=" * 60)
    print()
    print("  STAFF LOGIN")
    print("    Email    : staff@murdoch.edu.au")
    print("    Password : staff123")
    print()
    print("  STUDENT LOGINS  (S1000 to S1019 - 20 accounts)")
    print("    Email    : s[ID]@student.murdoch.edu.au")
    print("    Example  : s1000@student.murdoch.edu.au")
    print("    Password : student123  (all accounts)")
    print()
    print("  WHAT TO DO NEXT:")
    print("    1.  python run.py")
    print("    2.  Log in as staff and activate ICT304 from Staff Portal -> Units -> ICT304 -> Edit Unit")
    print("    3.  Optionally create additional units through Staff Portal -> Units -> Create Unit")
    print("    4.  Activate any additional units you want students to access")
    print("    5.  Log in as a student to enrol in active units and submit demo assessment scores")
    print("    6.  Return to the Staff Portal and run the AI model to generate risk results")
    print("    7.  Review cohort results and export PDF, CSV, or JSON reports as needed")
    print()
    print("  KEY FACTS:")
    print("    - Data persists across restarts (nothing deleted by run.py)")
    print("    - Re-running seed_db.py is safe (INSERT OR IGNORE)")
    print("    - Create unlimited extra units via Staff Portal -> Add Unit")
    print("    - Students can enrol in as many activated units as needed")
    print("    - Run Model as many times as you want (always overwrites)")
    print("=" * 60)


if __name__ == "__main__":
    run_seed()