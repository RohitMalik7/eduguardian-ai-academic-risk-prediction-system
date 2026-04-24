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
# FILE          : train_model.py
# PURPOSE       : Train, compare, and save three AI classification models:
#                   1. Logistic Regression  (interpretable linear baseline)
#                   2. Random Forest        (non-linear ensemble)
#                   3. XGBoost              (gradient boosting - HD addition)
#                 Each model is optimised using GridSearchCV with 5-fold
#                 stratified cross-validation. The best model is selected
#                 based on F1 score for the at-risk class (label=1) and
#                 saved to models/risk_model.joblib for use in the live app.
#
# INPUT         : data/processed/ds1_train_smote.csv   (SMOTE-balanced, scaled)
#                 data/processed/ds5_validation.csv    (OULAD held-out validation)
#                 models/scaler.joblib                 (fitted StandardScaler)
#
# OUTPUT        : models/risk_model.joblib             (best model only)
#                 models/model_metrics.json            (all metrics, all 3 models)
#                 models/feature_importance.csv        (per-model importances)
#                 reports/plots/confusion_matrices.png
#                 reports/plots/roc_curves.png
#                 reports/plots/feature_importance.png
#
# WHY THREE TECHNIQUES:
#   Logistic Regression - linear, interpretable baseline. Coefficients show
#     exactly how each assessment feature contributes to risk probability.
#     Suitable when risk increases linearly with lower scores.
#   Random Forest - non-linear ensemble of decision trees. Captures complex
#     feature interactions (e.g., low quizzes AND low assignments = very high
#     risk). Robust to outliers across merged multi-source datasets.
#   XGBoost - sequential gradient boosting. Each tree corrects errors of the
#     previous one. Consistently highest performance on tabular/structured data.
#     Included to demonstrate awareness of state-of-the-art ML methods.
#
# MODEL SELECTION CRITERION - F1 SCORE (not Accuracy):
#   Accuracy can be misleadingly high if one class dominates. F1 combines
#   Precision (avoiding unnecessary false alarms for staff) and Recall
#   (not missing real at-risk students). In an academic early-warning system,
#   missing a real at-risk student (low Recall) is more harmful than sending
#   one extra alert. F1 is therefore the most appropriate selection criterion.
# =============================================================================

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     ConfusionMatrixDisplay, roc_curve)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data",    "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")

TRAIN_FILE  = os.path.join(PROCESSED_DIR, "ds1_train_smote.csv")
VALID_FILE  = os.path.join(PROCESSED_DIR, "ds5_validation.csv")
SCALER_FILE = os.path.join(MODELS_DIR,    "scaler.joblib")

MODEL_OUTPUT      = os.path.join(MODELS_DIR, "risk_model.joblib")
METRICS_OUTPUT    = os.path.join(MODELS_DIR, "model_metrics.json")
IMPORTANCE_OUTPUT = os.path.join(MODELS_DIR, "feature_importance.csv")
PLOTS_DIR         = os.path.join(REPORTS_DIR, "plots")

# =============================================================================
# FEATURE DEFINITIONS
# Must match preprocess.py exactly - same names, same order.
# Final_Score is intentionally excluded (data leakage - not available mid-sem).
# =============================================================================

FINAL_FEATURES = [
    "Quizzes_Avg",
    "Assignments_Avg",
    "Midterm_Score",
    "Participation_Score",
    "Projects_Score",
]

TARGET_COL    = "At_Risk"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5
ML_THRESHOLD  = 0.45    # Below 0.5 to favour Recall - catching more at-risk students


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def ensure_dirs():
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)


def print_section(title):
    print("\n" + "="*70)
    print("  " + title)
    print("="*70)


def compute_metrics(model, X, y, threshold=ML_THRESHOLD, label=""):
    """
    Compute full classification metrics using a custom probability threshold.
    Using threshold=0.45 instead of default 0.50 increases Recall (catches
    more at-risk students at the cost of slightly more false alarms).
    """
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)
    cm    = confusion_matrix(y, pred)

    TP = int(cm[1, 1]) if cm.shape == (2, 2) else 0
    FP = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    TN = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    FN = int(cm[1, 0]) if cm.shape == (2, 2) else 0

    return {
        "label":            label,
        "accuracy":         round(float(accuracy_score(y, pred)),                   4),
        "precision":        round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall":           round(float(recall_score(y, pred, zero_division=0)),    4),
        "f1":               round(float(f1_score(y, pred, zero_division=0)),        4),
        "roc_auc":          round(float(roc_auc_score(y, proba)),                  4),
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "confusion_matrix": cm.tolist(),
    }


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    print_section("LOADING DATA")

    if not os.path.exists(TRAIN_FILE):
        print("  [ERROR] Training file not found: " + TRAIN_FILE)
        print("          Run preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(TRAIN_FILE)
    print("  Training file loaded : " + str(len(df)) + " records")

    X = df[FINAL_FEATURES].values
    y = df[TARGET_COL].values

    # Stratified split preserves At_Risk class ratio in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("  Train set : " + str(len(X_train)) + " | Test set : " + str(len(X_test)))
    print("  At-Risk in train : " + str(y_train.sum()) +
          " | At-Risk in test : " + str(y_test.sum()))

    # Load OULAD validation set if available
    X_val, y_val = None, None
    if os.path.exists(VALID_FILE):
        df_val = pd.read_csv(VALID_FILE)

        # Fill any missing feature columns with 50.0 (neutral midpoint)
        for col in FINAL_FEATURES:
            if col not in df_val.columns:
                df_val[col] = 50.0

        df_val[FINAL_FEATURES] = df_val[FINAL_FEATURES].clip(0, 100)
        df_val.dropna(subset=[TARGET_COL], inplace=True)

        # Apply the saved scaler - OULAD data is unscaled, must match training scale
        scaler = joblib.load(SCALER_FILE)
        X_val  = scaler.transform(df_val[FINAL_FEATURES].values)
        y_val  = df_val[TARGET_COL].values

        # KNOWN LIMITATION - print explicit warning for report documentation
        print("\n  [WARNING] OULAD VALIDATION SET - KNOWN LIMITATION:")
        print("  DS5 (OULAD) only contains Quizzes_Avg and Assignments_Avg.")
        print("  Midterm_Score, Participation_Score, Projects_Score are filled")
        print("  with 50.0 (neutral placeholder) because OULAD does not provide")
        print("  these fields. As a result, validation F1/Recall will be")
        print("  artificially suppressed and should be interpreted with caution.")
        print("  This limitation is documented in the project report.")
        print("  OULAD validation records loaded : " + str(len(df_val)))
    else:
        print("  [INFO] OULAD validation file not found - skipping held-out eval.")

    return X_train, X_test, y_train, y_test, X_val, y_val


# =============================================================================
# MODEL 1 - LOGISTIC REGRESSION
#
# WHY LOGISTIC REGRESSION:
#   - Interpretable: coefficients directly show how each feature affects risk.
#     A negative coefficient on Quizzes_Avg means higher quizzes -> lower risk.
#   - Strong linear baseline: if the data is approximately linearly separable,
#     LR will perform close to more complex models at far less cost.
#   - Required from Assignment 1 for consistency and comparison.
#
# HYPERPARAMETERS TUNED:
#   C      : Inverse regularisation strength. Smaller = stronger penalty on
#            large coefficients, reducing overfitting. Tested: 0.01 to 10.
#   solver : Optimisation algorithm. 'lbfgs' is stable and memory-efficient.
#            'saga' is faster on larger datasets and supports L1 penalty.
#   max_iter: Convergence limit. Increased to 2000 to ensure full convergence.
#
# GRID SIZE: 4 × 2 × 3 = 24 combinations × 5 folds = 120 fits  <- fast
# =============================================================================

def train_logistic_regression(X_train, y_train):
    print_section("MODEL 1 - LOGISTIC REGRESSION (GridSearchCV)")

    param_grid = {
        "C":        [0.01, 0.1, 1.0, 10.0],
        "solver":   ["lbfgs", "saga"],
        "max_iter": [500, 1000, 2000],
    }

    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE),
        param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    print("  Best params : " + str(grid.best_params_))
    print("  Best CV F1  : " + str(round(grid.best_score_, 4)))
    return grid.best_estimator_


# =============================================================================
# MODEL 2 - RANDOM FOREST CLASSIFIER
#
# WHY RANDOM FOREST:
#   - Captures non-linear patterns and feature interactions that LR misses.
#     Example: a student with low Quizzes_Avg AND low Assignments_Avg may be
#     much higher risk than either signal alone suggests.
#   - Ensemble of many trees: each tree sees a random subset of data and
#     features. Final prediction is majority vote - reduces overfitting.
#   - Feature importance scores: shows which assessment component (quizzes,
#     assignments, midterm) contributes most to risk prediction.
#   - class_weight="balanced" adjusts sample weights inversely to class freq.
#     This handles any residual class imbalance after SMOTE.
#
# HYPERPARAMETERS TUNED (reduced grid for reasonable runtime):
#   n_estimators     : 100 or 200 trees (300 removed - marginal gain, slow)
#   max_depth        : 4, 6, or None (unlimited). Controls overfitting.
#   min_samples_leaf : 2 or 5. Minimum samples at a leaf node.
#
# GRID SIZE: 2 × 3 × 2 = 12 combinations × 5 folds = 60 fits  <- fast
# =============================================================================

def train_random_forest(X_train, y_train):
    print_section("MODEL 2 - RANDOM FOREST CLASSIFIER (GridSearchCV)")

    param_grid = {
        "n_estimators":     [100, 200],
        "max_depth":        [4, 6, None],
        "min_samples_leaf": [2, 5],
    }

    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    print("  Best params : " + str(grid.best_params_))
    print("  Best CV F1  : " + str(round(grid.best_score_, 4)))
    return grid.best_estimator_


# =============================================================================
# MODEL 3 - XGBOOST CLASSIFIER
#
# WHY XGBOOST:
#   - Gradient boosting: trees built sequentially, each correcting the errors
#     of the previous one. Converges to a highly accurate model.
#   - Consistently outperforms Random Forest on structured tabular data in
#     real-world benchmarks and Kaggle competitions.
#   - Handles missing values natively (though we already imputed them).
#   - eval_metric="logloss": uses log-loss for internal tree building - stable
#     and appropriate for binary classification probability calibration.
#
# NOTE ON use_label_encoder:
#   This parameter was deprecated in XGBoost 1.6 and REMOVED in XGBoost 2.0+.
#   It must NOT be passed - doing so crashes the training pipeline with a
#   TypeError on any modern Python installation. It is intentionally omitted.
#
# NOTE ON scale_pos_weight:
#   After SMOTE in preprocess.py, classes are already balanced (50/50).
#   Therefore neg/pos ≈ 1.0 and scale_pos_weight has no effect.
#   It is intentionally NOT set here to avoid misleading the reader into
#   thinking the data is still imbalanced. SMOTE has already handled this.
#
# HYPERPARAMETERS TUNED (compact grid - fast runtime):
#   n_estimators  : 100 or 200 boosting rounds.
#   max_depth     : 3 or 6. Shallow trees generalise better.
#   learning_rate : 0.05 or 0.1. Step size per round. Smaller = more robust.
#
# GRID SIZE: 2 × 2 × 2 = 8 combinations × 5 folds = 40 fits  <- fast
# Total fits across all 3 models: 120 + 60 + 40 = 220 fits  <- acceptable
# =============================================================================

def train_xgboost(X_train, y_train):
    print_section("MODEL 3 - XGBOOST CLASSIFIER (GridSearchCV)")

    param_grid = {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 6],
        "learning_rate": [0.05, 0.1],
    }

    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        XGBClassifier(
            eval_metric="logloss",      # use_label_encoder removed - crashes XGBoost 2.0+
            random_state=RANDOM_STATE
        ),
        param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    print("  Best params : " + str(grid.best_params_))
    print("  Best CV F1  : " + str(round(grid.best_score_, 4)))
    return grid.best_estimator_


# =============================================================================
# EVALUATE AND COMPARE ALL THREE MODELS
# =============================================================================

def evaluate_all(models_dict, X_test, y_test, X_val, y_val):
    print_section("MODEL EVALUATION - TEST SET RESULTS")

    results = {}
    for name, model in models_dict.items():
        m = compute_metrics(model, X_test, y_test, label=name + " (test)")
        results[name] = {"test": m}

        print("\n  --- " + name + " ---")
        print("  Accuracy  : " + str(m["accuracy"]))
        print("  Precision : " + str(m["precision"]))
        print("  Recall    : " + str(m["recall"]))
        print("  F1 Score  : " + str(m["f1"]))
        print("  ROC-AUC   : " + str(m["roc_auc"]))
        print("  TP=" + str(m["TP"]) + "  FP=" + str(m["FP"]) +
              "  TN=" + str(m["TN"]) + "  FN=" + str(m["FN"]))

        if X_val is not None:
            mv = compute_metrics(model, X_val, y_val,
                                 label=name + " (OULAD validation)")
            results[name]["oulad_validation"] = mv
            # Remind reader that OULAD metrics are suppressed due to missing features
            print("  OULAD Val  F1=" + str(mv["f1"]) +
                  "  Recall=" + str(mv["recall"]) +
                  "  ROC-AUC=" + str(mv["roc_auc"]) +
                  "  [NOTE: 3 features filled with 50.0 - see report limitation]")

    return results


# =============================================================================
# SELECT BEST MODEL
# =============================================================================

def select_best_model(models_dict, results):
    print_section("MODEL SELECTION - BEST MODEL BY F1 SCORE")

    best_name = None
    best_f1   = -1
    for name in models_dict:
        f1 = results[name]["test"]["f1"]
        print("  " + name.ljust(25) + " F1 = " + str(f1))
        if f1 > best_f1:
            best_f1   = f1
            best_name = name

    print("\n  [SELECTED] " + best_name + " | F1 = " + str(best_f1))
    print("  Justification: Highest F1 on at-risk class. F1 chosen over")
    print("  accuracy because it balances Precision and Recall - critical")
    print("  for an early-warning system where missing at-risk students")
    print("  is more harmful than occasional false alarms.")
    return best_name, models_dict[best_name]


# =============================================================================
# SAVE FEATURE IMPORTANCE
# =============================================================================

def save_feature_importance(models_dict):
    print_section("FEATURE IMPORTANCE")
    rows = []
    for name, model in models_dict.items():
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
        elif hasattr(model, "coef_"):
            imps = np.abs(model.coef_[0])
        else:
            continue
        for feat, imp in zip(FINAL_FEATURES, imps):
            rows.append({"model": name, "feature": feat,
                         "importance": round(float(imp), 6)})
        print("  " + name + ": " +
              str({f: round(float(i), 4) for f, i in zip(FINAL_FEATURES, imps)}))

    if rows:
        pd.DataFrame(rows).to_csv(IMPORTANCE_OUTPUT, index=False)
        print("  [SAVED] " + IMPORTANCE_OUTPUT)


# =============================================================================
# GENERATE EVALUATION PLOTS
# =============================================================================

def generate_plots(models_dict, X_test, y_test):
    print_section("GENERATING EVALUATION PLOTS")
    names  = list(models_dict.keys())
    models = list(models_dict.values())
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    # Plot 1 - Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices - All Three Models", fontsize=14, fontweight="bold")
    for i, (name, model) in enumerate(zip(names, models)):
        proba = model.predict_proba(X_test)[:, 1]
        pred  = (proba >= ML_THRESHOLD).astype(int)
        cm    = confusion_matrix(y_test, pred)
        disp  = ConfusionMatrixDisplay(cm, display_labels=["Not At-Risk", "At-Risk"])
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(name, fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("  [SAVED] " + p)

    # Plot 2 - ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, model) in enumerate(zip(names, models)):
        proba       = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val     = round(roc_auc_score(y_test, proba), 4)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=name + " (AUC = " + str(auc_val) + ")")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves - Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    p = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("  [SAVED] " + p)

    # Plot 3 - Feature Importance
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Feature Importance - All Three Models", fontsize=14, fontweight="bold")
    for i, (name, model) in enumerate(zip(names, models)):
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
        elif hasattr(model, "coef_"):
            imps = np.abs(model.coef_[0])
        else:
            continue
        idx   = np.argsort(imps)
        feats = [FINAL_FEATURES[j] for j in idx]
        axes[i].barh(feats, imps[idx], color=colors[i], edgecolor="white")
        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Importance", fontsize=10)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("  [SAVED] " + p)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  ICT304 - AI Academic Risk Prediction System")
    print("  MODEL TRAINING PIPELINE - Starting...")
    print("="*70)

    ensure_dirs()

    # Step 1: Load data
    X_train, X_test, y_train, y_test, X_val, y_val = load_data()

    # Step 2: Train all three models
    lr_model  = train_logistic_regression(X_train, y_train)
    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    models_dict = {
        "Logistic Regression": lr_model,
        "Random Forest":       rf_model,
        "XGBoost":             xgb_model,
    }

    # Step 3: Evaluate all models
    results = evaluate_all(models_dict, X_test, y_test, X_val, y_val)

    # Step 4: Select best model
    best_name, best_model = select_best_model(models_dict, results)

    # Step 5: Save BEST MODEL ONLY
    # NOTE: all_models is intentionally NOT saved here.
    # Saving all 3 models (especially RF with 200 trees) creates a much larger
    # file and slows down application startup and prediction loading.
    # Only the best model is saved for live predictions.
    # All 3 models remain available during this training session via models_dict.
    model_package = {
        "model":      best_model,
        "model_name": best_name,
        "features":   FINAL_FEATURES,
        "threshold":  ML_THRESHOLD,
    }
    joblib.dump(model_package, MODEL_OUTPUT)
    print("\n  [SAVED] Best model only -> " + MODEL_OUTPUT)
    print("  (all_models excluded from package - keeps the saved model lightweight)")

    # Step 6: Save all metrics to JSON
    def to_serialisable(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return obj

    metrics_out = {}
    for name, res in results.items():
        metrics_out[name] = {}
        for split, m in res.items():
            metrics_out[name][split] = {k: to_serialisable(v) for k, v in m.items()}
    metrics_out["best_model"]    = best_name
    metrics_out["ml_threshold"]  = ML_THRESHOLD
    metrics_out["features_used"] = FINAL_FEATURES
    metrics_out["oulad_warning"] = (
        "OULAD validation metrics are suppressed: Midterm_Score, "
        "Participation_Score, and Projects_Score were filled with 50.0 "
        "(neutral placeholder) because OULAD does not provide these fields. "
        "Treat OULAD validation F1/Recall as a lower-bound estimate only."
    )

    with open(METRICS_OUTPUT, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("  [SAVED] All model metrics -> " + METRICS_OUTPUT)

    # Step 7: Feature importance
    save_feature_importance(models_dict)

    # Step 8: Generate plots
    generate_plots(models_dict, X_test, y_test)

    # Final summary
    print_section("TRAINING COMPLETE - SUMMARY")
    print("  Models trained      : Logistic Regression, Random Forest, XGBoost")
    print("  GridSearchCV fits   : ~220 total (LR=120, RF=60, XGB=40)")
    print("  Best model          : " + best_name)
    print("  Selection criterion : F1 score on at-risk class (label=1)")
    print("  Threshold used      : " + str(ML_THRESHOLD))
    print("  Outputs:")
    print("    - " + MODEL_OUTPUT)
    print("    - " + METRICS_OUTPUT)
    print("    - " + IMPORTANCE_OUTPUT)
    print("    - " + PLOTS_DIR + "/confusion_matrices.png")
    print("    - " + PLOTS_DIR + "/roc_curves.png")
    print("    - " + PLOTS_DIR + "/feature_importance.png")
    print("\n  [DONE] Run evaluate_model.py or launch the system with python run.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
