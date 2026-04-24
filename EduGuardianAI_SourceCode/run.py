# =============================================================================
# run.py
# ICT304 - EduGuardian AI - Application Entry Point
#
# HOW TO RUN (from project root):
#   1. First time only - seed the database:
#        python database/seed_db.py
#
#   2. Start the Flask server:
#        python run.py
#
#   3. Open your browser at:
#        http://127.0.0.1:5000
#
# DEMO CREDENTIALS:
#   Staff  : staff@murdoch.edu.au  / staff123
#   Student: s1000@student.murdoch.edu.au / student123  (S1000–S1019)
#
# AI PIPELINE:
#   The model is trained automatically the first time you click "Run Model"
#   in the Staff Portal. Or you can pre-train manually:
#       python src/preprocess.py
#       python src/train_model.py
# =============================================================================

import os
import sys

# Ensure the project root is on sys.path so all imports resolve correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import create_app
from database.db import init_db

# -- Startup checks ------------------------------------------------------------

def check_database():
    """Ensure the database and tables exist before starting."""
    db_path = os.path.join(BASE_DIR, "database", "academic_system.db")
    if not os.path.exists(db_path):
        print("[startup] Database not found - initialising tables...")
        init_db()
        print("[startup] Tables created. Run 'python database/seed_db.py' to add demo data.")
    else:
        # Always ensure tables exist (safe to call multiple times)
        init_db()


def print_startup_banner():
    print()
    print("=" * 60)
    print("  EduGuardian AI - Academic Risk Prediction System")
    print("  ICT304 · Murdoch University · T1 2026")
    print("=" * 60)
    print()
    print("  URL         : http://127.0.0.1:5000")
    print()
    print("  Staff login : staff@murdoch.edu.au  / staff123")
    print("  Student     : s1000@student.murdoch.edu.au / student123")
    print("              : (S1000–S1019 all use password: student123)")
    print()
    print("  AI Model    : Will train automatically on first 'Run Model'")
    print("                OR pre-train with:")
    print("                  python src/preprocess.py")
    print("                  python src/train_model.py")
    print()
    model_path = os.path.join(BASE_DIR, "models", "risk_model.joblib")
    if os.path.exists(model_path):
        print("  Model status: READY (risk_model.joblib found)")
    else:
        print("  Model status: NOT TRAINED YET (will auto-train on first run)")
    print()
    print("=" * 60)
    print()


# -- Main ----------------------------------------------------------------------

check_database()
print_startup_banner()

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)