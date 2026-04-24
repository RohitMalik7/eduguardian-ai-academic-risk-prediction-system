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
# tests/test_risk_engine.py
# ICT304 AI Academic Risk Prediction System
#
# Unit Tests - Deterministic Risk Engine (src/risk_engine.py)
#
# Coverage:
#   - analyse_quiz_trend()      slope, trend direction, projection, risk score
#   - score_assignment_risk()   risk scoring, unit marks, edge cases
#   - compute_combined_risk()   fusion weights, threshold classification
#   - generate_alert()          reason generation, action text, colour coding
#   - run_risk_assessment()     full pipeline integration, output schema
#   - save_report()             JSON file creation and content validation
#   - DEFAULT_UNIT_CONFIG       constant integrity
#   - Boundary / edge cases     empty input, single quiz, perfect scores, zeros
#
# Run with:
#   pytest tests/test_risk_engine.py -v
#
# Design notes:
#   - No real database or ML model required - pure deterministic logic.
#   - save_report() tests use tmp_path fixture to avoid polluting reports/.
#   - All numeric thresholds are imported from the module itself to stay
#     in sync if constants are ever changed.
# =============================================================================

import os
import sys
import json
import warnings
import pytest
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)

for p in [BASE_DIR, os.path.join(BASE_DIR, "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import risk_engine as re

# ---------------------------------------------------------------------------
# Convenience aliases - imported from the module so tests stay in sync
# ---------------------------------------------------------------------------
DEFAULT_CFG       = re.DEFAULT_UNIT_CONFIG
QUIZ_MAX          = DEFAULT_CFG["quiz_max_marks"]        # 20
QUIZ_COUNT        = DEFAULT_CFG["quiz_count"]            # 10
QUIZ_WEIGHT       = DEFAULT_CFG["quiz_weight"]           # 20.0
ASSIGN_MAX        = DEFAULT_CFG["assignment_max"]        # 100
ASSIGN_WEIGHT     = DEFAULT_CFG["assignment_weight"]     # 30.0
HIGH_THRESH       = re.RISK_HIGH_THRESHOLD               # 0.55
MEDIUM_THRESH     = re.RISK_MEDIUM_THRESHOLD             # 0.35
QUIZ_W            = re.QUIZ_RISK_WEIGHT                  # 0.60
ASSIGN_W          = re.ASSIGNMENT_RISK_WEIGHT            # 0.40
MIN_QUIZZES_TREND = re.MIN_QUIZZES_FOR_TREND             # 3


# =============================================================================
# SECTION 1 - analyse_quiz_trend()
# =============================================================================

class TestAnalyseQuizTrend:
    """Tests for the quiz trend analysis sub-system."""

    # -- Output schema ---------------------------------------------------------

    def test_returns_required_keys(self):
        """Output dict must contain all expected keys."""
        result = re.analyse_quiz_trend([15, 14, 13, 12], DEFAULT_CFG)
        required = {
            "quizzes_entered", "quiz_count_total", "current_avg",
            "current_avg_pct", "current_marks", "slope", "trend",
            "projected_avg", "projected_avg_pct", "projected_marks",
            "quiz_weight", "quiz_risk_score",
        }
        assert required.issubset(set(result.keys())), (
            f"Missing keys: {required - set(result.keys())}"
        )

    # -- Edge case: no quizzes --------------------------------------------------

    def test_empty_scores_returns_neutral(self):
        """Zero quizzes entered must return safe neutral values - no crash."""
        result = re.analyse_quiz_trend([], DEFAULT_CFG)
        assert result["quizzes_entered"] == 0
        assert result["quiz_risk_score"] == 0.5, (
            "No quiz data -> neutral risk score 0.5 expected"
        )
        assert result["trend"] == "Stable"

    def test_single_quiz_returns_stable_trend(self):
        """Only 1 quiz - not enough for trend fitting, should default to Stable."""
        result = re.analyse_quiz_trend([18], DEFAULT_CFG)
        assert result["quizzes_entered"] == 1
        assert result["trend"] == "Stable"
        assert result["slope"] == 0.0

    def test_two_quizzes_returns_stable_trend(self):
        """2 quizzes is below MIN_QUIZZES_FOR_TREND - trend must be Stable."""
        result = re.analyse_quiz_trend([15, 12], DEFAULT_CFG)
        assert result["trend"] == "Stable"

    # -- Trend direction classification ----------------------------------------

    def test_clearly_declining_scores_trend_declining(self):
        """Strongly declining scores should produce trend='Declining'."""
        result = re.analyse_quiz_trend([18, 16, 14, 12, 10, 8], DEFAULT_CFG)
        assert result["trend"] == "Declining", (
            f"Expected Declining, got {result['trend']}"
        )
        assert result["slope"] < -0.5

    def test_clearly_improving_scores_trend_improving(self):
        """Strongly improving scores should produce trend='Improving'."""
        result = re.analyse_quiz_trend([8, 10, 12, 14, 16, 18], DEFAULT_CFG)
        assert result["trend"] == "Improving", (
            f"Expected Improving, got {result['trend']}"
        )
        assert result["slope"] > 0.5

    def test_flat_scores_trend_stable(self):
        """Scores with no slope should produce trend='Stable'."""
        result = re.analyse_quiz_trend([10, 10, 10, 10, 10], DEFAULT_CFG)
        assert result["trend"] == "Stable"
        assert abs(result["slope"]) <= 0.5

    def test_slight_improvement_still_stable(self):
        """A very small positive slope (< 0.5) should remain Stable."""
        result = re.analyse_quiz_trend([10, 10, 11, 10, 11, 10], DEFAULT_CFG)
        assert result["trend"] == "Stable"

    # -- Numerical accuracy ----------------------------------------------------

    def test_current_avg_pct_correct(self):
        """current_avg_pct = (mean of scores / quiz_max) * 100."""
        scores = [10, 10, 10]   # avg = 10, max = 20 -> 50%
        result = re.analyse_quiz_trend(scores, DEFAULT_CFG)
        assert result["current_avg_pct"] == pytest.approx(50.0, abs=0.1)

    def test_projected_marks_within_bounds(self):
        """projected_marks must always be between 0 and quiz_weight."""
        for scores in [
            [20, 20, 20, 20, 20],    # perfect
            [0,  0,  0,  0,  0],     # zero
            [10, 10, 10],            # mid
            [18, 16, 14, 12, 10, 8], # declining
        ]:
            result = re.analyse_quiz_trend(scores, DEFAULT_CFG)
            assert 0.0 <= result["projected_marks"] <= QUIZ_WEIGHT, (
                f"projected_marks={result['projected_marks']} out of bounds "
                f"for input {scores}"
            )

    def test_quiz_risk_score_between_0_and_1(self):
        """quiz_risk_score must always be in [0, 1]."""
        for scores in [[20]*5, [0]*5, [10]*5, [18,16,14,12,10,8], [8,10,12,14,16,18]]:
            result = re.analyse_quiz_trend(scores, DEFAULT_CFG)
            assert 0.0 <= result["quiz_risk_score"] <= 1.0, (
                f"quiz_risk_score={result['quiz_risk_score']} out of [0,1] "
                f"for input {scores}"
            )

    def test_perfect_scores_low_risk_score(self):
        """All-perfect scores should produce very low quiz risk score."""
        result = re.analyse_quiz_trend([20]*6, DEFAULT_CFG)
        assert result["quiz_risk_score"] < 0.2, (
            f"Perfect scores should produce low risk, got {result['quiz_risk_score']}"
        )

    def test_zero_scores_high_risk_score(self):
        """All-zero scores should produce high quiz risk score."""
        result = re.analyse_quiz_trend([0]*6, DEFAULT_CFG)
        assert result["quiz_risk_score"] > 0.8, (
            f"Zero scores should produce high risk, got {result['quiz_risk_score']}"
        )

    def test_quizzes_entered_count_matches_input(self):
        scores = [12, 14, 16]
        result = re.analyse_quiz_trend(scores, DEFAULT_CFG)
        assert result["quizzes_entered"] == 3

    def test_quiz_count_total_from_config(self):
        result = re.analyse_quiz_trend([10], DEFAULT_CFG)
        assert result["quiz_count_total"] == QUIZ_COUNT

    def test_projected_avg_pct_between_0_and_100(self):
        result = re.analyse_quiz_trend([10, 12, 14, 16], DEFAULT_CFG)
        assert 0.0 <= result["projected_avg_pct"] <= 100.0

    # -- Custom unit config -----------------------------------------------------

    def test_custom_unit_config_respected(self):
        """Custom config with different max marks should be used correctly."""
        custom_cfg = dict(DEFAULT_CFG)
        custom_cfg["quiz_max_marks"] = 10
        custom_cfg["quiz_weight"]    = 10.0
        scores = [10, 10, 10, 10]   # perfect on 10-mark scale
        result = re.analyse_quiz_trend(scores, custom_cfg)
        assert result["current_avg_pct"] == pytest.approx(100.0, abs=0.1)


# =============================================================================
# SECTION 2 - score_assignment_risk()
# =============================================================================

class TestScoreAssignmentRisk:
    """Tests for the assignment risk scoring sub-system."""

    # -- Output schema ---------------------------------------------------------

    def test_returns_required_keys(self):
        result = re.score_assignment_risk(70.0, 80.0, DEFAULT_CFG)
        required = {
            "assignments_submitted", "assignment_avg_pct",
            "a1_unit_marks", "a2_unit_marks",
            "assignment_risk_score", "note",
        }
        assert required.issubset(set(result.keys()))

    # -- No submissions --------------------------------------------------------

    def test_no_assignments_returns_neutral_score(self):
        """No submissions -> neutral risk score 0.5."""
        result = re.score_assignment_risk(None, None, DEFAULT_CFG)
        assert result["assignments_submitted"] == 0
        assert result["assignment_risk_score"] == 0.5
        assert result["assignment_avg_pct"] is None

    def test_no_assignments_a1_marks_none(self):
        result = re.score_assignment_risk(None, None, DEFAULT_CFG)
        assert result["a1_unit_marks"] is None
        assert result["a2_unit_marks"] is None

    # -- Single submission ------------------------------------------------------

    def test_only_a1_submitted(self):
        result = re.score_assignment_risk(80.0, None, DEFAULT_CFG)
        assert result["assignments_submitted"] == 1
        assert result["a1_unit_marks"] is not None
        assert result["a2_unit_marks"] is None

    def test_only_a2_submitted(self):
        result = re.score_assignment_risk(None, 75.0, DEFAULT_CFG)
        assert result["assignments_submitted"] == 1
        assert result["a2_unit_marks"] is not None

    # -- Risk score correctness ------------------------------------------------

    def test_high_scores_produce_low_risk(self):
        """Scores well above the 65% benchmark should produce low risk."""
        result = re.score_assignment_risk(85.0, 90.0, DEFAULT_CFG)
        assert result["assignment_risk_score"] == 0.0, (
            f"High scores should produce risk=0, got {result['assignment_risk_score']}"
        )

    def test_scores_at_65_produce_zero_risk(self):
        """Exactly 65% average = risk benchmark -> risk score = 0.0."""
        result = re.score_assignment_risk(65.0, 65.0, DEFAULT_CFG)
        assert result["assignment_risk_score"] == pytest.approx(0.0, abs=0.01)

    def test_zero_scores_produce_maximum_risk(self):
        """Zero marks should produce maximum risk score = 1.0."""
        result = re.score_assignment_risk(0.0, 0.0, DEFAULT_CFG)
        assert result["assignment_risk_score"] == pytest.approx(1.0, abs=0.01)

    def test_low_scores_produce_high_risk(self):
        result = re.score_assignment_risk(30.0, 35.0, DEFAULT_CFG)
        assert result["assignment_risk_score"] >= 0.5

    def test_risk_score_always_in_0_1(self):
        """Risk score must be clamped to [0, 1] for all inputs."""
        for a1, a2 in [(0, 0), (100, 100), (50, 50), (None, None), (80, None)]:
            result = re.score_assignment_risk(a1, a2, DEFAULT_CFG)
            assert 0.0 <= result["assignment_risk_score"] <= 1.0

    # -- Unit marks calculation ------------------------------------------------

    def test_a1_unit_marks_calculation(self):
        """a1_unit_marks = a1 / assign_max * (assign_weight / assign_count)."""
        weight_each = ASSIGN_WEIGHT / DEFAULT_CFG["assignment_count"]   # 15.0
        result = re.score_assignment_risk(80.0, None, DEFAULT_CFG)
        expected = round(80.0 / ASSIGN_MAX * weight_each, 2)
        assert result["a1_unit_marks"] == pytest.approx(expected, abs=0.01)

    def test_a2_unit_marks_calculation(self):
        weight_each = ASSIGN_WEIGHT / DEFAULT_CFG["assignment_count"]
        result = re.score_assignment_risk(None, 60.0, DEFAULT_CFG)
        expected = round(60.0 / ASSIGN_MAX * weight_each, 2)
        assert result["a2_unit_marks"] == pytest.approx(expected, abs=0.01)

    def test_both_submitted_count_is_2(self):
        result = re.score_assignment_risk(70.0, 80.0, DEFAULT_CFG)
        assert result["assignments_submitted"] == 2

    def test_avg_pct_correct_both_submitted(self):
        result = re.score_assignment_risk(60.0, 80.0, DEFAULT_CFG)
        expected_avg = (60.0 + 80.0) / 2
        assert result["assignment_avg_pct"] == pytest.approx(expected_avg, abs=0.1)


# =============================================================================
# SECTION 3 - compute_combined_risk()
# =============================================================================

class TestComputeCombinedRisk:
    """Tests for the hybrid signal fusion and risk classification."""

    # -- Output schema ---------------------------------------------------------

    def test_returns_tuple_of_two(self):
        result = re.compute_combined_risk(0.6, 0.6, True)
        assert isinstance(result, tuple) and len(result) == 2

    def test_combined_score_is_float(self):
        score, _ = re.compute_combined_risk(0.5, 0.5, True)
        assert isinstance(score, float)

    def test_risk_level_is_valid_string(self):
        _, level = re.compute_combined_risk(0.5, 0.5, True)
        assert level in ("HIGH", "MEDIUM", "LOW")

    # -- Threshold classification -----------------------------------------------

    def test_high_scores_produce_HIGH_risk(self):
        """Combined score > HIGH_THRESH -> 'HIGH'."""
        score, level = re.compute_combined_risk(0.9, 0.9, True)
        assert score > HIGH_THRESH
        assert level == "HIGH"

    def test_medium_scores_produce_MEDIUM_risk(self):
        """Combined score in (MEDIUM_THRESH, HIGH_THRESH] -> 'MEDIUM'."""
        # Target a score around 0.45 (between 0.35 and 0.55)
        score, level = re.compute_combined_risk(0.45, 0.45, True)
        assert MEDIUM_THRESH < score <= HIGH_THRESH
        assert level == "MEDIUM"

    def test_low_scores_produce_LOW_risk(self):
        """Combined score < MEDIUM_THRESH -> 'LOW'."""
        score, level = re.compute_combined_risk(0.1, 0.1, True)
        assert score < MEDIUM_THRESH
        assert level == "LOW"

    # -- Weight application ----------------------------------------------------

    def test_combined_score_with_assignments(self):
        """With assignments: score = quiz*0.60 + assign*0.40."""
        q, a = 0.6, 0.8
        expected = round(q * QUIZ_W + a * ASSIGN_W, 4)
        score, _ = re.compute_combined_risk(q, a, True)
        assert score == pytest.approx(expected, abs=0.001)

    def test_combined_score_without_assignments(self):
        """Without assignments: full weight on quiz signal only."""
        q = 0.7
        score, _ = re.compute_combined_risk(q, 0.5, False)
        assert score == pytest.approx(q, abs=0.001), (
            "No assignment data -> combined score must equal quiz risk"
        )

    # -- Boundary values -------------------------------------------------------

    def test_exactly_at_high_threshold(self):
        """Score exactly at HIGH_THRESH should be HIGH."""
        # Set quiz and assign so combined = HIGH_THRESH exactly
        q = HIGH_THRESH / QUIZ_W
        a = 0.0
        score, level = re.compute_combined_risk(min(q, 1.0), 0.0, True)
        assert level in ("HIGH", "MEDIUM"), (
            "Boundary classification may vary slightly due to rounding"
        )

    def test_combined_score_clamped_to_0_1(self):
        """Combined score must always be in [0, 1]."""
        for q, a, has in [(1.0, 1.0, True), (0.0, 0.0, True),
                          (0.5, 0.5, False), (1.0, 0.0, False)]:
            score, _ = re.compute_combined_risk(q, a, has)
            assert 0.0 <= score <= 1.0, (
                f"Score {score} out of [0,1] for q={q}, a={a}, has={has}"
            )


# =============================================================================
# SECTION 4 - generate_alert()
# =============================================================================

class TestGenerateAlert:
    """Tests for staff alert generation and reason building."""

    def _declining_trend(self):
        return re.analyse_quiz_trend([18, 16, 14, 12, 10, 8], DEFAULT_CFG)

    def _improving_trend(self):
        return re.analyse_quiz_trend([8, 10, 12, 14, 16, 18], DEFAULT_CFG)

    def _flat_trend(self):
        return re.analyse_quiz_trend([10, 10, 10, 10, 10], DEFAULT_CFG)

    def _bad_assignment(self):
        return re.score_assignment_risk(30.0, 35.0, DEFAULT_CFG)

    def _good_assignment(self):
        return re.score_assignment_risk(80.0, 85.0, DEFAULT_CFG)

    def _no_assignment(self):
        return re.score_assignment_risk(None, None, DEFAULT_CFG)

    # -- Output schema ---------------------------------------------------------

    def test_returns_required_keys(self):
        result = re.generate_alert("HIGH", self._declining_trend(),
                                   self._bad_assignment(), "Test Student")
        required = {"risk_level", "colour", "reasons", "action",
                    "email_subject", "student_name"}
        assert required.issubset(set(result.keys()))

    def test_reasons_is_non_empty_list(self):
        result = re.generate_alert("HIGH", self._declining_trend(),
                                   self._bad_assignment(), "Omar")
        assert isinstance(result["reasons"], list)
        assert len(result["reasons"]) >= 1, "At least one reason must always be present"

    # -- Risk level routing ----------------------------------------------------

    def test_HIGH_risk_produces_urgent_action(self):
        result = re.generate_alert("HIGH", self._declining_trend(),
                                   self._bad_assignment(), "Student A")
        assert "URGENT" in result["action"].upper() or "Contact" in result["action"]
        assert result["colour"] == "red"

    def test_MEDIUM_risk_produces_recommended_action(self):
        result = re.generate_alert("MEDIUM", self._flat_trend(),
                                   self._no_assignment(), "Student B")
        assert result["colour"] == "orange"
        assert "check" in result["action"].lower() or "schedule" in result["action"].lower()

    def test_LOW_risk_produces_monitoring_action(self):
        result = re.generate_alert("LOW", self._improving_trend(),
                                   self._good_assignment(), "Student C")
        assert result["colour"] == "green"
        assert result["risk_level"] == "LOW"

    # -- Reason triggers -------------------------------------------------------

    def test_declining_trend_triggers_decline_reason(self):
        result = re.generate_alert("HIGH", self._declining_trend(),
                                   self._no_assignment(), "Omar")
        reasons_text = " ".join(result["reasons"]).lower()
        assert "declining" in reasons_text, (
            "Declining trend should trigger a decline-related reason"
        )

    def test_low_projected_avg_triggers_reason(self):
        """Projected avg below 50% should appear in reasons."""
        low_trend = re.analyse_quiz_trend([5, 4, 3, 2, 1], DEFAULT_CFG)
        result = re.generate_alert("HIGH", low_trend,
                                   self._no_assignment(), "Student")
        reasons_text = " ".join(result["reasons"]).lower()
        assert "projected" in reasons_text or "benchmark" in reasons_text

    def test_failing_assignment_triggers_reason(self):
        """Assignment average below 50% should appear in reasons."""
        result = re.generate_alert("HIGH", self._flat_trend(),
                                   self._bad_assignment(), "Student")
        reasons_text = " ".join(result["reasons"]).lower()
        assert "assignment" in reasons_text

    def test_student_name_in_email_subject(self):
        result = re.generate_alert("HIGH", self._declining_trend(),
                                   self._bad_assignment(), "Omar Al-Rashid")
        assert "Omar Al-Rashid" in result["email_subject"]

    def test_default_reason_always_provided(self):
        """Even when no specific triggers fire, a fallback reason must be present."""
        flat_trend  = self._flat_trend()
        no_assign   = self._no_assignment()
        result = re.generate_alert("LOW", flat_trend, no_assign, "Student")
        assert len(result["reasons"]) >= 1


# =============================================================================
# SECTION 5 - run_risk_assessment() - Full Pipeline Integration
# =============================================================================

class TestRunRiskAssessment:
    """Integration tests for the full deterministic pipeline."""

    # -- Output schema ---------------------------------------------------------

    def test_returns_required_top_level_keys(self):
        result = re.run_risk_assessment(
            student_id="TEST001", unit_name="ICT304",
            quiz_scores=[15, 14, 13], a1=70.0, a2=75.0,
            save_json=False
        )
        required = {
            "student_id", "student_name", "unit_name",
            "trend", "assignment", "combined_score",
            "risk_level", "alert", "ml_signal", "report_path",
        }
        assert required.issubset(set(result.keys()))

    def test_risk_level_is_valid(self):
        result = re.run_risk_assessment(
            student_id="TEST001", unit_name="ICT304",
            quiz_scores=[10, 10, 10], save_json=False
        )
        assert result["risk_level"] in ("HIGH", "MEDIUM", "LOW")

    def test_combined_score_is_in_range(self):
        result = re.run_risk_assessment(
            student_id="TEST001", unit_name="ICT304",
            quiz_scores=[12, 12, 12], save_json=False
        )
        assert 0.0 <= result["combined_score"] <= 1.0

    # -- Known scenario outcomes -----------------------------------------------

    def test_declining_student_is_HIGH_risk(self):
        """Strongly declining quizzes + failing assignment -> HIGH risk."""
        result = re.run_risk_assessment(
            student_id="TEST_HIGH",
            unit_name="ICT304",
            quiz_scores=[18, 16, 14, 12, 10, 8, 6, 4],
            a1=20.0,
            a2=None,
            save_json=False
        )
        assert result["risk_level"] == "HIGH", (
            f"Declining student should be HIGH risk, got {result['risk_level']} "
            f"(score={result['combined_score']})"
        )

    def test_improving_student_is_LOW_risk(self):
        """Strongly improving quizzes + good assignments -> LOW risk."""
        result = re.run_risk_assessment(
            student_id="TEST_LOW",
            unit_name="ICT304",
            quiz_scores=[10, 12, 14, 16, 18, 19],
            a1=78.0,
            a2=82.0,
            save_json=False
        )
        assert result["risk_level"] == "LOW", (
            f"Improving student should be LOW risk, got {result['risk_level']} "
            f"(score={result['combined_score']})"
        )

    def test_stable_low_no_assignments_MEDIUM_risk(self):
        """Stable low scores with no assignments -> MEDIUM risk."""
        result = re.run_risk_assessment(
            student_id="TEST_MED",
            unit_name="ICT304",
            quiz_scores=[9, 10, 9, 11, 10],
            a1=None,
            a2=None,
            save_json=False
        )
        assert result["risk_level"] in ("MEDIUM", "HIGH"), (
            f"Stable low student with no assignments should be MEDIUM or HIGH, "
            f"got {result['risk_level']}"
        )

    def test_no_quizzes_no_assignments_returns_result(self):
        """Completely empty input should not crash - must return a valid result."""
        result = re.run_risk_assessment(
            student_id="TEST_EMPTY",
            unit_name="ICT304",
            quiz_scores=[],
            save_json=False
        )
        assert result["risk_level"] in ("HIGH", "MEDIUM", "LOW")
        assert result["combined_score"] is not None

    def test_student_id_preserved_in_output(self):
        result = re.run_risk_assessment(
            student_id="S1042", unit_name="ICT304",
            quiz_scores=[12, 13, 14], save_json=False
        )
        assert result["student_id"] == "S1042"

    def test_unit_name_preserved_in_output(self):
        result = re.run_risk_assessment(
            student_id="S1001", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_json=False
        )
        assert result["unit_name"] == "ICT304"

    def test_save_json_false_report_path_none(self):
        result = re.run_risk_assessment(
            student_id="TEST", unit_name="ICT304",
            quiz_scores=[10, 12, 14], save_json=False
        )
        assert result["report_path"] is None

    def test_alert_keys_present_in_output(self):
        result = re.run_risk_assessment(
            student_id="TEST", unit_name="ICT304",
            quiz_scores=[10, 10, 10], save_json=False
        )
        assert "reasons" in result["alert"]
        assert "action" in result["alert"]

    def test_trend_keys_present_in_output(self):
        result = re.run_risk_assessment(
            student_id="TEST", unit_name="ICT304",
            quiz_scores=[12, 14, 16], save_json=False
        )
        assert "slope" in result["trend"]
        assert "trend" in result["trend"]

    def test_custom_unit_config_respected(self):
        """Custom config should override defaults."""
        custom = dict(DEFAULT_CFG)
        custom["quiz_max_marks"] = 10
        custom["quiz_weight"]    = 10.0
        result = re.run_risk_assessment(
            student_id="TEST", unit_name="ICT304",
            quiz_scores=[10, 10, 10],
            unit_config=custom,
            save_json=False
        )
        assert result["trend"]["quiz_weight"] == 10.0


# =============================================================================
# SECTION 6 - save_report()
# =============================================================================

class TestSaveReport:
    """Tests for JSON audit report file creation and content."""

    def _get_trend(self):
        return re.analyse_quiz_trend([15, 14, 13, 12], DEFAULT_CFG)

    def _get_assign(self):
        return re.score_assignment_risk(70.0, 75.0, DEFAULT_CFG)

    def _get_alert(self, level="MEDIUM"):
        return re.generate_alert(level, self._get_trend(),
                                 self._get_assign(), "Test Student")

    def test_creates_file(self, tmp_path, monkeypatch):
        """save_report() must create a .json file in the reports directory."""
        monkeypatch.setattr(re, "REPORTS_DIR", str(tmp_path))
        path = re.save_report(
            student_id="S1001", unit_name="ICT304",
            trend_result=self._get_trend(),
            assign_result=self._get_assign(),
            combined_score=0.42,
            risk_level="MEDIUM",
            alert=self._get_alert(),
        )
        assert os.path.exists(path), f"Report file not created at {path}"

    def test_file_is_valid_json(self, tmp_path, monkeypatch):
        """The saved file must be valid, parseable JSON."""
        monkeypatch.setattr(re, "REPORTS_DIR", str(tmp_path))
        path = re.save_report(
            student_id="S1001", unit_name="ICT304",
            trend_result=self._get_trend(),
            assign_result=self._get_assign(),
            combined_score=0.42,
            risk_level="MEDIUM",
            alert=self._get_alert(),
        )
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict), "Report file must contain a JSON object"

    def test_report_contains_student_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(re, "REPORTS_DIR", str(tmp_path))
        path = re.save_report(
            student_id="S9999", unit_name="ICT304",
            trend_result=self._get_trend(),
            assign_result=self._get_assign(),
            combined_score=0.6,
            risk_level="HIGH",
            alert=self._get_alert("HIGH"),
        )
        with open(path) as f:
            data = json.load(f)
        assert data.get("student_id") == "S9999"

    def test_report_contains_risk_level(self, tmp_path, monkeypatch):
        monkeypatch.setattr(re, "REPORTS_DIR", str(tmp_path))
        path = re.save_report(
            student_id="S0001", unit_name="ICT304",
            trend_result=self._get_trend(),
            assign_result=self._get_assign(),
            combined_score=0.6,
            risk_level="HIGH",
            alert=self._get_alert("HIGH"),
        )
        with open(path) as f:
            data = json.load(f)
        assert data["combined_risk"]["risk_level"] == "HIGH"

    def test_report_filename_includes_student_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(re, "REPORTS_DIR", str(tmp_path))
        path = re.save_report(
            student_id="S1234", unit_name="ICT304",
            trend_result=self._get_trend(),
            assign_result=self._get_assign(),
            combined_score=0.3,
            risk_level="LOW",
            alert=self._get_alert("LOW"),
        )
        assert "S1234" in os.path.basename(path)

    def test_returns_string_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(re, "REPORTS_DIR", str(tmp_path))
        path = re.save_report(
            student_id="S0001", unit_name="ICT304",
            trend_result=self._get_trend(),
            assign_result=self._get_assign(),
            combined_score=0.4,
            risk_level="MEDIUM",
            alert=self._get_alert(),
        )
        assert isinstance(path, str)


# =============================================================================
# SECTION 7 - DEFAULT_UNIT_CONFIG Constant Integrity
# =============================================================================

class TestDefaultUnitConfig:
    """Verify DEFAULT_UNIT_CONFIG has all required keys and sensible values."""

    REQUIRED_KEYS = [
        "quiz_count", "quiz_max_marks", "quiz_weight",
        "assignment_count", "assignment_max", "assignment_weight",
        "has_exam", "exam_max", "exam_weight", "pass_mark",
    ]

    def test_all_required_keys_present(self):
        for key in self.REQUIRED_KEYS:
            assert key in DEFAULT_CFG, f"Missing key: '{key}' in DEFAULT_UNIT_CONFIG"

    def test_quiz_weight_is_20(self):
        assert DEFAULT_CFG["quiz_weight"] == 20.0

    def test_assignment_weight_is_30(self):
        assert DEFAULT_CFG["assignment_weight"] == 30.0

    def test_exam_weight_is_50(self):
        assert DEFAULT_CFG["exam_weight"] == 50.0

    def test_weights_sum_to_100(self):
        total = (DEFAULT_CFG["quiz_weight"]
                 + DEFAULT_CFG["assignment_weight"]
                 + DEFAULT_CFG["exam_weight"])
        assert total == pytest.approx(100.0, abs=0.01), (
            f"Assessment weights must sum to 100, got {total}"
        )

    def test_pass_mark_is_50(self):
        assert DEFAULT_CFG["pass_mark"] == 50.0

    def test_quiz_count_is_10(self):
        assert DEFAULT_CFG["quiz_count"] == 10

    def test_quiz_max_marks_is_20(self):
        assert DEFAULT_CFG["quiz_max_marks"] == 20
