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
# app/__init__.py
# ICT304 - EduGuardian AI - Flask Application Factory
# =============================================================================

from flask import Flask
from .utils import register_jinja_filters


def create_app():
    app = Flask(__name__)

    # Secret key for session signing - change this in production
    app.secret_key = "eduguardian-ai-ict304-secret-key-2026"

    # Session configuration
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # Register custom Jinja2 filters
    register_jinja_filters(app)

    # Register all routes
    from .routes import main
    app.register_blueprint(main)

    return app