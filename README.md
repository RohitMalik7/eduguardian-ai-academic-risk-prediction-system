# EduGuardian AI - Academic Risk Prediction System

> **Author:** Rohit Malik | **Domain:** Artificial Intelligence / Full-Stack Development | **Type:** Group Academic Project | **Purpose:** Portfolio & Educational

---

## Overview

EduGuardian AI is a full-stack, end-to-end artificial intelligence system designed to identify at-risk students early in the semester using partial academic data.

The system combines machine learning models with a rule-based risk engine to deliver accurate and interpretable predictions, enabling academic staff to take timely intervention actions.

---

## Project Type

Group Project - 3 Members

- Rohit Kumar Malik
- Izaan Shumaiz
- Mohamed Sinan

---

## Problem Statement

Universities often identify at-risk students only after major assessments, which results in delayed intervention and reduced chances of academic recovery.

There is a need for a data-driven system that can detect early signs of risk using partial semester data and provide actionable insights.

---

## Proposed Solution

EduGuardian AI is a hybrid AI system that:

- Analyses early academic performance data
- Predicts probability of student failure
- Classifies students into LOW, MEDIUM, HIGH risk categories
- Provides interpretable insights for academic decision-making

---

## Key Features

| Feature | Description |
|---|---|
| Hybrid AI Model | Machine Learning combined with a Rule-Based Engine |
| Early Prediction | Risk detection using partial semester data |
| Web Application | Full-stack Flask application with dashboards |
| Data Pipeline | Preprocessing, feature engineering, and transformation |
| Model Evaluation | ROC-AUC, F1-score, precision, and recall |
| Class Imbalance Handling | SMOTE applied during training |
| Dashboards | Separate interfaces for staff and student interaction |
| Exportable Reports | Output available in PDF, CSV, and JSON formats |

---

## Repository Structure

```text
EduGuardian-AI/
|
+-- EduGuardianAI_SourceCode/
|   |
|   +-- app/              - Flask web application (UI + routing)
|   +-- data/             - Input datasets (excluded / placeholder only)
|   +-- database/         - Database integration (SQLite)
|   +-- notebooks/        - Experimentation and analysis
|   +-- reports/plots     - Generated outputs and visualizations
|   +-- src/              - Core system logic and pipeline
|   +-- tests/            - Testing modules
|   |
|   +-- README.md         - Internal project documentation
|   +-- requirements.txt  - Project dependencies
|   +-- run.py            - Application entry point
|
+-- docs/
|   +-- EduGuardian_User_Guide.docx
|   +-- groupDeclarationSheet.docx
|   +-- Presentation.pptx
|   +-- Report_EduGuardianAI.docx
|
+-- README.md             - Main project overview
```
---

## System Architecture

The system follows a modular layered design:

| Layer | Description |
|---|---|
| Presentation Layer | Flask web interface |
| Application Layer | Business logic and API handling |
| Data Layer | SQLite database |
| AI Processing Layer | ML models combined with rule-based engine |

---

## AI Approach

### Machine Learning Models

| Model | Role |
|---|---|
| Logistic Regression | Baseline model |
| Random Forest | Best performing model |
| XGBoost | Advanced model |

### Rule-Based Risk Engine
- Detects academic trends such as declining performance
- Provides interpretable reasoning alongside ML predictions

### Hybrid Model
- Combines ML output with rule-based evaluation
- Improves both accuracy and interpretability of results

---

## Data Pipeline

1. Data input via dashboard
2. Data cleaning and preprocessing
3. Feature scaling and transformation
4. Handling class imbalance using SMOTE
5. Machine learning prediction
6. Rule-based evaluation
7. Hybrid risk scoring
8. Results displayed and stored

---

## How to Run

Install dependencies:
pip install -r requirements.txt

Run the application:
python run.py

Open in browser:
http://localhost:5000

---

## Documentation

Project documentation is available in the docs/ folder:

- User Guide
- System Report
- Presentation Slides
- Group Declaration

---

## My Contribution

- Designed hybrid AI architecture
- Implemented machine learning models and evaluation
- Developed data preprocessing pipeline
- Contributed to Flask application and system integration
- Assisted in testing and validation

---

## Skills Demonstrated

- Machine Learning
- Python
- Flask
- Data Analysis
- Model Evaluation
- System Design
- Full-Stack Development
- Problem Solving

---

## Limitations

- Limited real-world dataset
- Manual data input (CSV-based)
- Basic authentication system
- SQLite limits scalability

---

## Future Improvements

- Integration with LMS platforms such as Moodle
- Cloud deployment
- Real-time data ingestion
- Improved security and scalability

---

## Usage and Credit

This project is shared for portfolio and educational purposes. If you use or reference any part of this work, please provide appropriate credit to the author.

---

## Author

Rohit Malik
Email: rohitmalik180904@gmail.com
GitHub: https://github.com/RohitMalik7
Location: Dubai, UAE
