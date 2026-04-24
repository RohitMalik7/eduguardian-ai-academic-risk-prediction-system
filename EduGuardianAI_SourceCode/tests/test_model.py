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
# tests/test_model.py
# ICT304 AI Academic Risk Prediction System
#
# Unit & Integration Tests - ML Prediction Pipeline (src/predict.py)
#
# Coverage:
#   - build_feature_vector()       neutral defaults, participation scaling, schema
#   - fuse_signals()               weighted fusion, threshold classification,
#                                  confidence and agreement logic
#   - build_confidence_message()   plain-language explanation output
#   - ml_predict()                 ML inference using saved or synthetic model packages
#   - predict_student_risk()       full hybrid pipeline output schema,
#                                  graceful degradation, and known scenarios
#   - load_model() / load_scaler() caching and missing-file handling
#   - Trained model package        structure, feature count, threshold integrity
#   - FINAL_FEATURES               exact 5-feature schema and ordering
#
# Run with:
#   pytest tests/test_model.py -v
#
# Design notes:
#   - Tests that need a real trained model are marked with
#     @pytest.mark.skipif(not MODEL_AVAILABLE) so the suite always
#     runs cleanly even before train_model.py has been executed.
#   - Synthetic sklearn models (RandomForestClassifier trained on tiny
#     balanced data) are used for structural tests that do not need
#     the real production model.
#   - predict_student_risk() is tested end-to-end with save_report=False
#     so no files are written to disk during testing.
# =============================================================================

import os
import sys
import warnings
import pytest
import numpy as np
import joblib
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)

for p in [BASE_DIR, os.path.join(BASE_DIR, "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import predict as pred
import risk_engine as re

# ---------------------------------------------------------------------------
# Detect whether the real trained model exists
# ---------------------------------------------------------------------------
MODEL_FILE  = os.path.join(BASE_DIR, "models", "risk_model.joblib")
SCALER_FILE = os.path.join(BASE_DIR, "models", "scaler.joblib")
MODEL_AVAILABLE  = os.path.exists(MODEL_FILE)
SCALER_AVAILABLE = os.path.exists(SCALER_FILE)
BOTH_AVAILABLE   = MODEL_AVAILABLE and SCALER_AVAILABLE

SKIP_NO_MODEL  = pytest.mark.skipif(
    not BOTH_AVAILABLE,
    reason="Trained model not found - run preprocess.py then train_model.py first"
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
FINAL_FEATURES = ["Quizzes_Avg", "Assignments_Avg", "Midterm_Score",
                  "Participation_Score", "Projects_Score"]

DEFAULT_CFG = re.DEFAULT_UNIT_CONFIG

ML_WEIGHT     = pred.ML_WEIGHT       # 0.55
ENGINE_WEIGHT = pred.ENGINE_WEIGHT   # 0.45
HIGH_THRESH   = pred.FINAL_HIGH_THRESHOLD    # 0.55
MEDIUM_THRESH = pred.FINAL_MEDIUM_THRESHOLD  # 0.35


def _make_synthetic_package(tmp_path):
    """
    Build a minimal sklearn RandomForestClassifier package that mirrors
    the structure expected by predict.py:
      { model, model_name, features, threshold }
    Trained on 40 balanced synthetic samples - enough for predict_proba.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    np.random.seed(0)
    n = 40
    X = np.random.uniform(0, 100, (n, 5))
    y = np.array([0] * 20 + [1] * 20)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_scaled, y)

    package = {
        "model":      model,
        "model_name": "Random Forest",
        "features":   FINAL_FEATURES,
        "threshold":  0.45,
    }

    model_path  = str(tmp_path / "risk_model.joblib")
    scaler_path = str(tmp_path / "scaler.joblib")
    joblib.dump(package, model_path)
    joblib.dump(scaler, scaler_path)

    return model_path, scaler_path, package, scaler


def _trend(scores=None):
    """Return a trend result for given scores (default improving)."""
    scores = scores or [10, 12, 14, 16, 18]
    return re.analyse_quiz_trend(scores, DEFAULT_CFG)


def _assign(a1=75.0, a2=80.0):
    return re.score_assignment_risk(a1, a2, DEFAULT_CFG)


# =============================================================================
# SECTION 1 - FINAL_FEATURES Constant Integrity
# =============================================================================

class TestFinalFeaturesConstant:
    """Verify the feature list in predict.py matches the 5-feature schema."""

    EXPECTED = ["Quizzes_Avg", "Assignments_Avg", "Midterm_Score",
                "Participation_Score", "Projects_Score"]

    def test_feature_count_is_5(self):
        assert len(pred.FINAL_FEATURES) == 5, (
            f"Expected 5 features, got {len(pred.FINAL_FEATURES)}"
        )

    def test_feature_names_exact(self):
        assert list(pred.FINAL_FEATURES) == self.EXPECTED, (
            f"Feature mismatch.\nExpected: {self.EXPECTED}\n"
            f"Got     : {list(pred.FINAL_FEATURES)}"
        )

    def test_ml_weight_plus_engine_weight_equals_1(self):
        """ML_WEIGHT + ENGINE_WEIGHT must sum to exactly 1.0."""
        total = ML_WEIGHT + ENGINE_WEIGHT
        assert abs(total - 1.0) < 1e-9, (
            f"ML_WEIGHT ({ML_WEIGHT}) + ENGINE_WEIGHT ({ENGINE_WEIGHT}) = {total}, "
            f"expected 1.0"
        )

    def test_ml_weight_is_0_55(self):
        assert ML_WEIGHT == pytest.approx(0.55, abs=1e-9)

    def test_engine_weight_is_0_45(self):
        assert ENGINE_WEIGHT == pytest.approx(0.45, abs=1e-9)

    def test_high_threshold_greater_than_medium(self):
        assert HIGH_THRESH > MEDIUM_THRESH, (
            "HIGH_THRESH must be greater than MEDIUM_THRESH"
        )

    def test_thresholds_in_valid_range(self):
        assert 0.0 < MEDIUM_THRESH < HIGH_THRESH < 1.0


# =============================================================================
# SECTION 2 - build_feature_vector()
# =============================================================================

class TestBuildFeatureVector:
    """Tests for feature vector construction from trend + assessment data."""

    def test_returns_tuple_of_two(self):
        result = pred.build_feature_vector(_trend())
        assert isinstance(result, tuple) and len(result) == 2

    def test_vector_shape_is_1x5(self):
        vector, _ = pred.build_feature_vector(_trend())
        assert vector.shape == (1, 5), (
            f"Expected shape (1, 5), got {vector.shape}"
        )

    def test_feature_values_dict_has_5_keys(self):
        _, feat_vals = pred.build_feature_vector(_trend())
        assert len(feat_vals) == 5

    def test_feature_values_dict_has_correct_keys(self):
        _, feat_vals = pred.build_feature_vector(_trend())
        for key in FINAL_FEATURES:
            assert key in feat_vals, f"Missing key '{key}' in feature_values"

    def test_none_a1_a2_fills_with_50(self):
        """No assignment data -> Assignments_Avg should be neutral 50.0."""
        _, feat_vals = pred.build_feature_vector(_trend(), a1=None, a2=None)
        assert feat_vals["Assignments_Avg"] == pytest.approx(50.0, abs=0.1), (
            f"Expected Assignments_Avg=50.0 (neutral), got {feat_vals['Assignments_Avg']}"
        )

    def test_both_assignments_averaged_correctly(self):
        """Assignments_Avg = mean(a1, a2)."""
        _, feat_vals = pred.build_feature_vector(_trend(), a1=60.0, a2=80.0)
        assert feat_vals["Assignments_Avg"] == pytest.approx(70.0, abs=0.1)

    def test_only_a1_used_when_a2_none(self):
        _, feat_vals = pred.build_feature_vector(_trend(), a1=70.0, a2=None)
        assert feat_vals["Assignments_Avg"] == pytest.approx(70.0, abs=0.1)

    def test_none_midterm_fills_with_50(self):
        _, feat_vals = pred.build_feature_vector(_trend(), midterm=None)
        assert feat_vals["Midterm_Score"] == pytest.approx(50.0, abs=0.1)

    def test_midterm_used_when_provided(self):
        _, feat_vals = pred.build_feature_vector(_trend(), midterm=72.0)
        assert feat_vals["Midterm_Score"] == pytest.approx(72.0, abs=0.1)

    def test_none_participation_fills_with_50(self):
        _, feat_vals = pred.build_feature_vector(_trend(), participation=None)
        assert feat_vals["Participation_Score"] == pytest.approx(50.0, abs=0.1)

    def test_participation_normalised_0_10_to_0_100(self):
        """Participation is on 0–10 scale and must be normalised to 0–100."""
        _, feat_vals = pred.build_feature_vector(_trend(), participation=8.0)
        assert feat_vals["Participation_Score"] == pytest.approx(80.0, abs=0.1), (
            f"participation=8.0 should normalise to 80.0, got {feat_vals['Participation_Score']}"
        )

    def test_participation_clamped_above_10(self):
        """Participation > 10 should be clamped to 10 before normalising."""
        _, feat_vals = pred.build_feature_vector(_trend(), participation=15.0)
        assert feat_vals["Participation_Score"] <= 100.0

    def test_none_projects_fills_with_50(self):
        _, feat_vals = pred.build_feature_vector(_trend(), projects=None)
        assert feat_vals["Projects_Score"] == pytest.approx(50.0, abs=0.1)

    def test_projects_used_when_provided(self):
        _, feat_vals = pred.build_feature_vector(_trend(), projects=88.0)
        assert feat_vals["Projects_Score"] == pytest.approx(88.0, abs=0.1)

    def test_quizzes_avg_uses_projected_avg_pct(self):
        """Quizzes_Avg must use projected_avg_pct from the trend result (not raw avg)."""
        trend = _trend([14, 16, 18, 20])
        _, feat_vals = pred.build_feature_vector(trend)
        expected = trend["projected_avg_pct"]
        assert feat_vals["Quizzes_Avg"] == pytest.approx(expected, abs=0.1), (
            f"Quizzes_Avg should use projected_avg_pct={expected}, "
            f"got {feat_vals['Quizzes_Avg']}"
        )

    def test_all_feature_values_are_floats(self):
        _, feat_vals = pred.build_feature_vector(
            _trend(), a1=70.0, a2=80.0, midterm=65.0,
            participation=7.0, projects=75.0
        )
        for k, v in feat_vals.items():
            assert isinstance(v, float), f"Feature '{k}' value {v} is not float"

    def test_vector_dtype_is_float(self):
        vector, _ = pred.build_feature_vector(_trend())
        assert vector.dtype in (np.float32, np.float64)


# =============================================================================
# SECTION 3 - fuse_signals()
# =============================================================================

class TestFuseSignals:
    """Tests for the ML + engine hybrid fusion function."""

    def test_returns_tuple_of_four(self):
        result = pred.fuse_signals(0.6, 0.6)
        assert isinstance(result, tuple) and len(result) == 4

    def test_final_score_formula(self):
        """final_score = ml_prob * ML_WEIGHT + engine_score * ENGINE_WEIGHT."""
        ml, eng = 0.7, 0.5
        expected = round(ml * ML_WEIGHT + eng * ENGINE_WEIGHT, 4)
        score, _, _, _ = pred.fuse_signals(ml, eng)
        assert score == pytest.approx(expected, abs=0.001)

    def test_final_score_always_in_0_1(self):
        for ml, eng in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5),
                        (0.0, 1.0), (1.0, 0.0)]:
            score, _, _, _ = pred.fuse_signals(ml, eng)
            assert 0.0 <= score <= 1.0

    # -- Risk level classification ---------------------------------------------

    def test_high_inputs_produce_HIGH_risk(self):
        score, level, _, _ = pred.fuse_signals(0.9, 0.9)
        assert score > HIGH_THRESH
        assert level == "HIGH"

    def test_medium_inputs_produce_MEDIUM_risk(self):
        # Target fused score around 0.45 (between 0.35 and 0.55)
        ml, eng = 0.45, 0.45
        score, level, _, _ = pred.fuse_signals(ml, eng)
        assert MEDIUM_THRESH < score <= HIGH_THRESH
        assert level == "MEDIUM"

    def test_low_inputs_produce_LOW_risk(self):
        score, level, _, _ = pred.fuse_signals(0.1, 0.1)
        assert score < MEDIUM_THRESH
        assert level == "LOW"

    def test_risk_level_is_valid_string(self):
        for ml, eng in [(0.1, 0.1), (0.45, 0.45), (0.9, 0.9)]:
            _, level, _, _ = pred.fuse_signals(ml, eng)
            assert level in ("HIGH", "MEDIUM", "LOW")

    # -- Agreement logic -------------------------------------------------------

    def test_both_high_produces_agree(self):
        """Both signals above 0.5 -> agreement = 'Agree'."""
        _, _, _, agreement = pred.fuse_signals(0.8, 0.7)
        assert agreement == "Agree"

    def test_both_low_produces_agree(self):
        """Both signals below 0.5 -> agreement = 'Agree'."""
        _, _, _, agreement = pred.fuse_signals(0.2, 0.3)
        assert agreement == "Agree"

    def test_disagreeing_signals_produces_disagree(self):
        """ML high, engine low (or vice versa) -> 'Disagree'."""
        _, _, _, agreement = pred.fuse_signals(0.8, 0.2)
        assert agreement == "Disagree"

    # -- Confidence logic ------------------------------------------------------

    def test_agreeing_signals_produce_high_confidence(self):
        _, _, confidence, agreement = pred.fuse_signals(0.8, 0.75)
        assert agreement == "Agree"
        assert confidence == "High"

    def test_large_gap_produces_low_confidence(self):
        """Gap > 0.3 between signals -> 'Low' confidence."""
        _, _, confidence, _ = pred.fuse_signals(0.9, 0.1)
        assert confidence == "Low"

    def test_moderate_gap_produces_moderate_confidence(self):
        """Gap around 0.2 -> 'Moderate' confidence."""
        _, _, confidence, _ = pred.fuse_signals(0.6, 0.35)
        assert confidence == "Moderate", (
            f"Gap of 0.25 should produce Moderate confidence, got {confidence}"
        )


# =============================================================================
# SECTION 4 - build_confidence_message()
# =============================================================================

class TestBuildConfidenceMessage:
    """Tests for the plain-language confidence explanation builder."""

    def test_high_agree_contains_both_agree(self):
        msg = pred.build_confidence_message("High", "Agree", 0.75, 0.70)
        assert "agree" in msg.lower() or "high confidence" in msg.lower()

    def test_moderate_message_mentions_review(self):
        msg = pred.build_confidence_message("Moderate", "Disagree", 0.6, 0.4)
        assert "review" in msg.lower() or "different" in msg.lower()

    def test_low_message_mentions_additional_data(self):
        msg = pred.build_confidence_message("Low", "Disagree", 0.9, 0.1)
        assert "additional" in msg.lower() or "disagree" in msg.lower() or "strongly" in msg.lower()

    def test_message_contains_ml_percentage(self):
        """Message should include the ML probability as a percentage."""
        msg = pred.build_confidence_message("High", "Agree", 0.75, 0.70)
        assert "75" in msg or "ML" in msg

    def test_message_contains_engine_percentage(self):
        msg = pred.build_confidence_message("High", "Agree", 0.75, 0.70)
        assert "70" in msg or "Engine" in msg

    def test_returns_non_empty_string(self):
        for conf, agree, ml, eng in [
            ("High",     "Agree",    0.8, 0.75),
            ("Moderate", "Disagree", 0.6, 0.4),
            ("Low",      "Disagree", 0.9, 0.1),
        ]:
            msg = pred.build_confidence_message(conf, agree, ml, eng)
            assert isinstance(msg, str) and len(msg) > 0


# =============================================================================
# SECTION 5 - load_model() / load_scaler() - Caching & Error Handling
# =============================================================================

class TestLoadModelScaler:
    """Tests for model and scaler loading with caching and error handling."""

    def test_missing_model_raises_file_not_found(self, tmp_path, monkeypatch):
        """load_model() must raise FileNotFoundError when file is missing."""
        monkeypatch.setattr(pred, "MODEL_FILE",  str(tmp_path / "missing.joblib"))
        monkeypatch.setattr(pred, "_model_cache", None)
        with pytest.raises(FileNotFoundError):
            pred.load_model()

    def test_missing_scaler_raises_file_not_found(self, tmp_path, monkeypatch):
        """load_scaler() must raise FileNotFoundError when file is missing."""
        monkeypatch.setattr(pred, "SCALER_FILE",  str(tmp_path / "missing.joblib"))
        monkeypatch.setattr(pred, "_scaler_cache", None)
        with pytest.raises(FileNotFoundError):
            pred.load_scaler()

    def test_load_model_uses_cache_on_second_call(self, tmp_path, monkeypatch):
        """load_model() must return the cached object on second call."""
        sentinel = {"model": "cached_sentinel"}
        monkeypatch.setattr(pred, "_model_cache", sentinel)
        result = pred.load_model()
        assert result is sentinel, "load_model() should return cached object"

    def test_load_scaler_uses_cache_on_second_call(self, tmp_path, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(pred, "_scaler_cache", sentinel)
        result = pred.load_scaler()
        assert result is sentinel

    @SKIP_NO_MODEL
    def test_real_model_loads_successfully(self):
        """If model exists, it must load without error."""
        pred._model_cache  = None
        pred._scaler_cache = None
        package = pred.load_model()
        assert package is not None

    @SKIP_NO_MODEL
    def test_real_model_package_has_required_keys(self):
        pred._model_cache = None
        package = pred.load_model()
        for key in ("model", "model_name", "features", "threshold"):
            assert key in package, f"Model package missing key: '{key}'"

    @SKIP_NO_MODEL
    def test_real_model_features_match_final_features(self):
        pred._model_cache = None
        package = pred.load_model()
        assert list(package["features"]) == FINAL_FEATURES

    @SKIP_NO_MODEL
    def test_real_model_threshold_is_0_45(self):
        pred._model_cache = None
        package = pred.load_model()
        assert package["threshold"] == pytest.approx(0.45, abs=1e-6)

    @SKIP_NO_MODEL
    def test_real_scaler_loads_successfully(self):
        pred._scaler_cache = None
        scaler = pred.load_scaler()
        assert scaler is not None

    @SKIP_NO_MODEL
    def test_real_scaler_has_correct_feature_count(self):
        pred._scaler_cache = None
        scaler = pred.load_scaler()
        assert len(scaler.mean_) == 5


# =============================================================================
# SECTION 6 - ml_predict() with Synthetic Model
# =============================================================================

class TestMlPredictSynthetic:
    """
    Tests for ml_predict() using a synthetic sklearn model.
    These always run - no real trained model required.
    """

    def test_returns_tuple_of_four(self, tmp_path, monkeypatch):
        model_path, scaler_path, package, scaler = _make_synthetic_package(tmp_path)
        monkeypatch.setattr(pred, "MODEL_FILE",   model_path)
        monkeypatch.setattr(pred, "SCALER_FILE",  scaler_path)
        monkeypatch.setattr(pred, "_model_cache",  None)
        monkeypatch.setattr(pred, "_scaler_cache", None)

        vector, _ = pred.build_feature_vector(_trend())
        result = pred.ml_predict(vector)
        assert isinstance(result, tuple) and len(result) == 4

    def test_ml_probability_in_0_1(self, tmp_path, monkeypatch):
        model_path, scaler_path, _, _ = _make_synthetic_package(tmp_path)
        monkeypatch.setattr(pred, "MODEL_FILE",   model_path)
        monkeypatch.setattr(pred, "SCALER_FILE",  scaler_path)
        monkeypatch.setattr(pred, "_model_cache",  None)
        monkeypatch.setattr(pred, "_scaler_cache", None)

        vector, _ = pred.build_feature_vector(_trend())
        ml_prob, _, _, _ = pred.ml_predict(vector)
        assert 0.0 <= ml_prob <= 1.0, f"ml_prob={ml_prob} out of [0,1]"

    def test_ml_label_is_valid(self, tmp_path, monkeypatch):
        model_path, scaler_path, _, _ = _make_synthetic_package(tmp_path)
        monkeypatch.setattr(pred, "MODEL_FILE",   model_path)
        monkeypatch.setattr(pred, "SCALER_FILE",  scaler_path)
        monkeypatch.setattr(pred, "_model_cache",  None)
        monkeypatch.setattr(pred, "_scaler_cache", None)

        vector, _ = pred.build_feature_vector(_trend())
        _, ml_label, _, _ = pred.ml_predict(vector)
        assert ml_label in ("At-Risk", "Not At-Risk"), (
            f"Invalid ml_label: '{ml_label}'"
        )

    def test_ml_label_consistent_with_probability(self, tmp_path, monkeypatch):
        """Label must match threshold: prob >= threshold -> At-Risk."""
        model_path, scaler_path, package, _ = _make_synthetic_package(tmp_path)
        monkeypatch.setattr(pred, "MODEL_FILE",   model_path)
        monkeypatch.setattr(pred, "SCALER_FILE",  scaler_path)
        monkeypatch.setattr(pred, "_model_cache",  None)
        monkeypatch.setattr(pred, "_scaler_cache", None)

        vector, _ = pred.build_feature_vector(_trend())
        ml_prob, ml_label, _, threshold = pred.ml_predict(vector)
        expected_label = "At-Risk" if ml_prob >= threshold else "Not At-Risk"
        assert ml_label == expected_label, (
            f"prob={ml_prob} threshold={threshold} -> expected '{expected_label}', "
            f"got '{ml_label}'"
        )

    def test_model_name_is_string(self, tmp_path, monkeypatch):
        model_path, scaler_path, _, _ = _make_synthetic_package(tmp_path)
        monkeypatch.setattr(pred, "MODEL_FILE",   model_path)
        monkeypatch.setattr(pred, "SCALER_FILE",  scaler_path)
        monkeypatch.setattr(pred, "_model_cache",  None)
        monkeypatch.setattr(pred, "_scaler_cache", None)

        vector, _ = pred.build_feature_vector(_trend())
        _, _, model_name, _ = pred.ml_predict(vector)
        assert isinstance(model_name, str) and len(model_name) > 0


# =============================================================================
# SECTION 7 - predict_student_risk() - Full Hybrid Pipeline
# =============================================================================

class TestPredictStudentRisk:
    """
    End-to-end tests for the full hybrid prediction pipeline.
    Uses synthetic model via monkeypatching - always runs.
    """

    def _patch(self, monkeypatch, tmp_path):
        model_path, scaler_path, _, _ = _make_synthetic_package(tmp_path)
        monkeypatch.setattr(pred, "MODEL_FILE",   model_path)
        monkeypatch.setattr(pred, "SCALER_FILE",  scaler_path)
        monkeypatch.setattr(pred, "_model_cache",  None)
        monkeypatch.setattr(pred, "_scaler_cache", None)

    # -- Output schema ---------------------------------------------------------

    def test_returns_required_keys(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[12, 14, 16], save_report=False
        )
        required = {
            "student_id", "student_name", "unit_name",
            "ml_probability", "ml_label", "ml_model_name", "ml_threshold",
            "ml_available", "engine_score", "engine_risk_level",
            "final_score", "final_risk_level", "confidence", "agreement",
            "confidence_message", "feature_values", "trend",
            "assignment", "alert", "report_path",
        }
        assert required.issubset(set(result.keys())), (
            f"Missing keys: {required - set(result.keys())}"
        )

    def test_final_risk_level_is_valid(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        assert result["final_risk_level"] in ("HIGH", "MEDIUM", "LOW")

    def test_ml_probability_in_0_1(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        assert 0.0 <= result["ml_probability"] <= 1.0

    def test_final_score_in_0_1(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        assert 0.0 <= result["final_score"] <= 1.0

    def test_student_id_preserved(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S9042", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        assert result["student_id"] == "S9042"

    def test_unit_name_preserved(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        assert result["unit_name"] == "ICT304"

    def test_save_report_false_gives_none_path(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        assert result["report_path"] is None

    def test_ml_label_matches_probability_and_threshold(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        expected = "At-Risk" if result["ml_probability"] >= result["ml_threshold"] else "Not At-Risk"
        assert result["ml_label"] == expected

    def test_feature_values_has_5_entries(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        assert len(result["feature_values"]) == 5

    def test_alert_has_reasons_and_action(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        assert "reasons" in result["alert"]
        assert "action" in result["alert"]

    def test_confidence_is_valid_string(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        assert result["confidence"] in ("High", "Moderate", "Low")

    def test_agreement_is_valid_string(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        assert result["agreement"] in ("Agree", "Disagree")

    # -- Graceful degradation - model absent -----------------------------------

    def test_missing_model_degrades_gracefully(self, tmp_path, monkeypatch):
        """
        If riskmodel.joblib is missing, predict_student_risk must NOT crash.
        It should fall back to engine-only mode and still return a valid result.
        """
        monkeypatch.setattr(pred, "MODEL_FILE",   str(tmp_path / "missing.joblib"))
        monkeypatch.setattr(pred, "SCALER_FILE",  str(tmp_path / "missing.joblib"))
        monkeypatch.setattr(pred, "_model_cache",  None)
        monkeypatch.setattr(pred, "_scaler_cache", None)

        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_report=False
        )
        assert result["final_risk_level"] in ("HIGH", "MEDIUM", "LOW"), (
            "Engine-only fallback must still produce a valid risk level"
        )
        assert result["ml_available"] is False, (
            "ml_available must be False when model is missing"
        )

    # -- Empty input ------------------------------------------------------------

    def test_empty_quiz_scores_no_crash(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        result = pred.predict_student_risk(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[], save_report=False
        )
        assert result["final_risk_level"] in ("HIGH", "MEDIUM", "LOW")


# =============================================================================
# SECTION 8 - Trained Model Quality (skipped if model not present)
# =============================================================================

class TestTrainedModelQuality:
    """
    Validate the real production model's output quality on known inputs.
    All tests in this class require train_model.py to have been run first.
    """

    @SKIP_NO_MODEL
    def test_model_predict_proba_returns_probabilities(self):
        """model.predict_proba() must return values in [0, 1]."""
        pred._model_cache  = None
        pred._scaler_cache = None
        package = pred.load_model()
        scaler  = pred.load_scaler()
        model   = package["model"]

        X_raw = pd.DataFrame([[70, 75, 68, 60, 72]], columns=FINAL_FEATURES)
        X_scaled = scaler.transform(X_raw)
        proba = model.predict_proba(X_scaled)

        assert proba.shape[1] == 2, "predict_proba must return 2 columns (0 and 1)"
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    @SKIP_NO_MODEL
    def test_model_output_class_count(self):
        """Model must output exactly 2 classes: 0 (not at risk) and 1 (at risk)."""
        pred._model_cache = None
        package = pred.load_model()
        model   = package["model"]
        assert len(model.classes_) == 2
        assert set(model.classes_) == {0, 1}

    @SKIP_NO_MODEL
    def test_high_risk_student_scores_high_probability(self):
        """
        A student with very low scores should receive a high at-risk probability.
        This validates the model has learned meaningful patterns.
        """
        pred._model_cache  = None
        pred._scaler_cache = None
        package = pred.load_model()
        scaler  = pred.load_scaler()
        model   = package["model"]
        threshold = package["threshold"]

        # All features near zero = clearly failing student
        X_raw    = pd.DataFrame([[5.0, 8.0, 10.0, 15.0, 5.0]], columns=FINAL_FEATURES)
        X_scaled = scaler.transform(X_raw)
        proba    = model.predict_proba(X_scaled)[0][1]

        assert proba > 0.3, (
            f"Failing student (all low scores) should have at-risk prob > 0.3, "
            f"got {proba:.3f}"
        )

    @SKIP_NO_MODEL
    def test_low_risk_student_scores_low_probability(self):
        """
        A student with high scores should receive a low at-risk probability.
        """
        pred._model_cache  = None
        pred._scaler_cache = None
        package = pred.load_model()
        scaler  = pred.load_scaler()
        model   = package["model"]

        # All features high = clearly passing student
        X_raw    = pd.DataFrame([[90.0, 85.0, 88.0, 82.0, 92.0]], columns=FINAL_FEATURES)
        X_scaled = scaler.transform(X_raw)
        proba    = model.predict_proba(X_scaled)[0][1]

        assert proba < 0.7, (
            f"High-scoring student should have at-risk prob < 0.7, got {proba:.3f}"
        )

    @SKIP_NO_MODEL
    def test_scaler_transform_changes_values(self):
        """StandardScaler must actually transform the values (not pass through)."""
        pred._scaler_cache = None
        scaler = pred.load_scaler()

        raw = pd.DataFrame([[50.0, 50.0, 50.0, 50.0, 50.0]], columns=FINAL_FEATURES)
        scaled = scaler.transform(raw)
        assert not np.allclose(raw, scaled), (
            "Scaler must transform feature values, not return them unchanged"
        )
