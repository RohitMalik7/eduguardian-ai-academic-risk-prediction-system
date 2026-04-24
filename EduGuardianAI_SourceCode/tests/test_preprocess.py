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
# tests/test_preprocess.py
# ICT304 AI Academic Risk Prediction System
#
# Unit Tests - Preprocessing Pipeline (src/preprocess.py)
#
# Coverage:
#   - grade_to_at_risk()        label encoding for both grading systems
#   - score_to_at_risk()        numeric score -> binary label
#   - drop_cols()               safe column dropping
#   - merge_training_datasets() concatenation and schema enforcement
#   - apply_smote()             class balancing via SMOTE
#   - scale_features()          StandardScaler output shape and stats
#   - load_dataset1/2/3/4()     file-absent graceful failure
#   - FINAL_FEATURES schema     all 5 features + At_Risk present after pipeline
#
# Run with:
#   pytest tests/test_preprocess.py -v
#
# Design notes:
#   - All tests use synthetic in-memory DataFrames - no real CSV files required.
#   - File-present tests are skipped with pytest.mark.skipif when datasets
#     are absent so the test suite always runs cleanly in any environment.
#   - SMOTE tests use kneighbors=1 to work with very small synthetic frames.
# =============================================================================

import os
import sys
import warnings
import pytest
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup - allow running from project root OR from tests/ directory
# ---------------------------------------------------------------------------
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(THIS_DIR)

for p in [BASE_DIR, os.path.join(BASE_DIR, "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import the module under test
import preprocess as pp

# ---------------------------------------------------------------------------
# Expected schema constants used for validation against preprocess.py
# ---------------------------------------------------------------------------
FINAL_FEATURES = ["Quizzes_Avg", "Assignments_Avg", "Midterm_Score",
                  "Participation_Score", "Projects_Score"]
TARGET_COL     = "At_Risk"


# =============================================================================
# SECTION 1 - grade_to_at_risk()
# =============================================================================

class TestGradeToAtRisk:
    """Tests for the grade = binary At_Risk label converter."""

    def test_murdoch_system_S_and_F_are_at_risk(self):
        """Murdoch grading: S and F both map to At_Risk=1."""
        grades = pd.Series(["HD", "D", "C", "P", "S", "F"])
        result = pp.grade_to_at_risk(grades, system="murdoch")
        assert result.tolist() == [0, 0, 0, 0, 1, 1], (
            "Murdoch: S and F should both be At_Risk=1"
        )

    def test_abcdf_system_only_F_is_at_risk(self):
        """ABCDF grading: only F maps to At_Risk=1; A/B/C/D all pass."""
        grades = pd.Series(["A", "B", "C", "D", "F"])
        result = pp.grade_to_at_risk(grades, system="abcdf")
        assert result.tolist() == [0, 0, 0, 0, 1], (
            "ABCDF: only F should be At_Risk=1"
        )

    def test_murdoch_uppercase_normalisation(self):
        """grade_to_at_risk should handle mixed case input."""
        grades = pd.Series(["hd", "f", "HD", "F"])
        result = pp.grade_to_at_risk(grades, system="murdoch")
        assert list(result) == [0, 1, 0, 1]

    def test_unknown_grade_fills_zero(self):
        """Unmapped grades should fill with 0 (not at risk by default)."""
        grades = pd.Series(["X", "Y", "Z"])
        result = pp.grade_to_at_risk(grades, system="murdoch")
        assert result.isnull().sum() == 0, "No NaN should remain after fillna(0)"
        assert set(result.tolist()).issubset({0, 1})

    def test_returns_integer_dtype(self):
        """Output must be integer dtype for ML compatibility."""
        grades = pd.Series(["A", "F", "B"])
        result = pp.grade_to_at_risk(grades, system="abcdf")
        assert result.dtype in (int, np.int64, np.int32), (
            f"Expected int dtype, got {result.dtype}"
        )


# =============================================================================
# SECTION 2 - score_to_at_risk()
# =============================================================================

class TestScoreToAtRisk:
    """Tests for numeric score -> binary At_Risk label converter."""

    def test_below_50_is_at_risk(self):
        """Scores strictly below 50 should be At_Risk=1."""
        scores = pd.Series([0.0, 25.0, 49.9])
        result = pp.score_to_at_risk(scores)
        assert all(result == 1), "All scores < 50 should be At_Risk=1"

    def test_exactly_50_is_not_at_risk(self):
        """Score of exactly 50 is the pass mark - should be At_Risk=0."""
        scores = pd.Series([50.0])
        result = pp.score_to_at_risk(scores)
        assert result.iloc[0] == 0, "Score=50 is passing (not at risk)"

    def test_above_50_is_not_at_risk(self):
        """Scores above 50 should all be At_Risk=0."""
        scores = pd.Series([51.0, 75.0, 100.0])
        result = pp.score_to_at_risk(scores)
        assert all(result == 0), "All scores > 50 should be At_Risk=0"

    def test_nan_is_handled(self):
        """NaN values should not crash - filled with neutral 50 (not at risk)."""
        scores = pd.Series([np.nan, 30.0, np.nan])
        result = pp.score_to_at_risk(scores)
        assert result.isnull().sum() == 0, "NaN should be filled before comparison"

    def test_string_coercion(self):
        """Non-numeric strings should be coerced to NaN then filled safely."""
        scores = pd.Series(["invalid", "75", "30"])
        result = pp.score_to_at_risk(scores)
        assert len(result) == 3
        assert set(result.tolist()).issubset({0, 1})

    def test_returns_integer_dtype(self):
        scores = pd.Series([40.0, 60.0, 80.0])
        result = pp.score_to_at_risk(scores)
        assert result.dtype in (int, np.int64, np.int32)


# =============================================================================
# SECTION 3 - drop_cols()
# =============================================================================

class TestDropCols:
    """Tests for the safe column dropping utility."""

    def _sample_df(self):
        return pd.DataFrame({
            "A": [1, 2], "B": [3, 4], "C": [5, 6]
        })

    def test_drops_existing_columns(self):
        df = self._sample_df()
        result = pp.drop_cols(df, ["A", "B"])
        assert list(result.columns) == ["C"]

    def test_ignores_nonexistent_columns(self):
        """Dropping a column that doesn't exist must not raise KeyError."""
        df = self._sample_df()
        result = pp.drop_cols(df, ["A", "Z"])   # Z does not exist
        assert "A" not in result.columns
        assert "B" in result.columns

    def test_empty_list_returns_unchanged(self):
        df = self._sample_df()
        result = pp.drop_cols(df, [])
        assert list(result.columns) == ["A", "B", "C"]

    def test_all_columns_dropped(self):
        df = self._sample_df()
        result = pp.drop_cols(df, ["A", "B", "C"])
        assert result.empty or list(result.columns) == []

    def test_does_not_modify_original(self):
        """drop_cols must return a new frame, not mutate in-place."""
        df = self._sample_df()
        _ = pp.drop_cols(df, ["A"])
        assert "A" in df.columns, "Original DataFrame should not be modified"


# =============================================================================
# SECTION 4 - merge_training_datasets()
# =============================================================================

class TestMergeTrainingDatasets:
    """Tests for multi-dataset concatenation and schema enforcement."""

    def _make_frame(self, n=50, at_risk_ratio=0.3, source="DS1"):
        """Create a synthetic DataFrame matching FINAL_FEATURES schema."""
        np.random.seed(42)
        n_at_risk = int(n * at_risk_ratio)
        df = pd.DataFrame({
            "Quizzes_Avg":       np.random.uniform(20, 90, n),
            "Assignments_Avg":   np.random.uniform(20, 90, n),
            "Midterm_Score":     np.random.uniform(20, 90, n),
            "Participation_Score": np.random.uniform(20, 90, n),
            "Projects_Score":    np.random.uniform(20, 90, n),
            TARGET_COL:         [1]*n_at_risk + [0]*(n - n_at_risk),
            "source":           source,
        })
        return df

    def test_concatenates_multiple_frames(self):
        df1 = self._make_frame(50, source="DS1")
        df2 = self._make_frame(60, source="DS2")
        merged = pp.merge_training_datasets([df1, df2])
        assert len(merged) >= 100, "Merged frame must contain rows from all inputs"

    def test_all_final_features_present(self):
        df1 = self._make_frame(50)
        df2 = self._make_frame(50)
        merged = pp.merge_training_datasets([df1, df2])
        for feat in FINAL_FEATURES:
            assert feat in merged.columns, f"Feature '{feat}' missing from merged output"

    def test_target_column_present(self):
        df1 = self._make_frame(50)
        merged = pp.merge_training_datasets([df1])
        assert TARGET_COL in merged.columns

    def test_target_is_binary(self):
        df1 = self._make_frame(100)
        merged = pp.merge_training_datasets([df1])
        unique_vals = set(merged[TARGET_COL].unique())
        assert unique_vals.issubset({0, 1}), (
            f"At_Risk must be binary (0/1), got {unique_vals}"
        )

    def test_all_values_clipped_0_100(self):
        """No feature value should exceed 100 or fall below 0 after merging."""
        df1 = self._make_frame(200)
        merged = pp.merge_training_datasets([df1])
        for feat in FINAL_FEATURES:
            assert merged[feat].max() <= 100.0, f"{feat} exceeds 100"
            assert merged[feat].min() >= 0.0,   f"{feat} below 0"

    def test_no_nulls_in_final_features(self):
        """merge_training_datasets must fill all NaNs in FINAL_FEATURES."""
        df1 = self._make_frame(50)
        # Introduce NaNs deliberately
        df1.loc[0:5, "Quizzes_Avg"] = np.nan
        merged = pp.merge_training_datasets([df1])
        for feat in FINAL_FEATURES:
            assert merged[feat].isnull().sum() == 0, (
                f"NaN found in '{feat}' after merge"
            )

    def test_missing_feature_filled_with_50(self):
        """If a feature column is absent from all frames, it should be filled with 50.0."""
        df1 = self._make_frame(50).drop(columns=["Midterm_Score"])
        merged = pp.merge_training_datasets([df1])
        assert "Midterm_Score" in merged.columns
        assert (merged["Midterm_Score"] == 50.0).all(), (
            "Absent feature column should be filled with 50.0"
        )


# =============================================================================
# SECTION 5 - apply_smote()
# =============================================================================

class TestApplySmote:
    """Tests for SMOTE class balancing."""

    def _make_imbalanced_frame(self, n_majority=80, n_minority=20):
        np.random.seed(42)
        n = n_majority + n_minority
        df = pd.DataFrame({
            "Quizzes_Avg":         np.random.uniform(30, 90, n),
            "Assignments_Avg":     np.random.uniform(30, 90, n),
            "Midterm_Score":       np.random.uniform(30, 90, n),
            "Participation_Score": np.random.uniform(30, 90, n),
            "Projects_Score":      np.random.uniform(30, 90, n),
            TARGET_COL:           [0]*n_majority + [1]*n_minority,
            "source":             "DS1",
        })
        return df

    def test_balances_classes(self):
        """After SMOTE the minority class count should match the majority class."""
        df = self._make_imbalanced_frame(80, 20)
        balanced = pp.apply_smote(df)
        counts = balanced[TARGET_COL].value_counts()
        assert counts[0] == counts[1], (
            f"Classes not balanced after SMOTE: {counts.to_dict()}"
        )

    def test_output_has_final_features(self):
        df = self._make_imbalanced_frame()
        balanced = pp.apply_smote(df)
        for feat in FINAL_FEATURES:
            assert feat in balanced.columns

    def test_source_column_dropped_before_smote(self):
        """SMOTE must not see the 'source' string column - it must be removed."""
        df = self._make_imbalanced_frame()
        balanced = pp.apply_smote(df)
        assert "source" not in balanced.columns, (
            "'source' column must be dropped before SMOTE"
        )

    def test_no_nulls_after_smote(self):
        df = self._make_imbalanced_frame()
        balanced = pp.apply_smote(df)
        assert balanced[FINAL_FEATURES].isnull().sum().sum() == 0

    def test_output_larger_than_input(self):
        """SMOTE must produce more records than the original minority class."""
        df = self._make_imbalanced_frame(80, 20)
        balanced = pp.apply_smote(df)
        assert len(balanced) > len(df), "SMOTE output must be larger than imbalanced input"


# =============================================================================
# SECTION 6 - scale_features()
# =============================================================================

class TestScaleFeatures:
    """Tests for StandardScaler output shape, statistics, and scaler persistence."""

    def _make_clean_frame(self, n=100):
        np.random.seed(0)
        df = pd.DataFrame({
            "Quizzes_Avg":         np.random.uniform(0, 100, n),
            "Assignments_Avg":     np.random.uniform(0, 100, n),
            "Midterm_Score":       np.random.uniform(0, 100, n),
            "Participation_Score": np.random.uniform(0, 100, n),
            "Projects_Score":      np.random.uniform(0, 100, n),
            TARGET_COL:           np.random.randint(0, 2, n),
        })
        return df

    def test_output_has_same_row_count(self):
        df = self._make_clean_frame(100)
        scaled_df, _ = pp.scale_features(df)
        assert len(scaled_df) == 100

    def test_output_has_all_final_features(self):
        df = self._make_clean_frame(100)
        scaled_df, _ = pp.scale_features(df)
        for feat in FINAL_FEATURES:
            assert feat in scaled_df.columns

    def test_target_column_preserved(self):
        df = self._make_clean_frame(100)
        scaled_df, _ = pp.scale_features(df)
        assert TARGET_COL in scaled_df.columns, "At_Risk must be preserved after scaling"

    def test_scaled_features_have_near_zero_mean(self):
        """StandardScaler should produce zero-mean features (tolerance 0.1)."""
        df = self._make_clean_frame(200)
        scaled_df, _ = pp.scale_features(df)
        for feat in FINAL_FEATURES:
            mean_val = scaled_df[feat].mean()
            assert abs(mean_val) < 0.1, (
                f"Feature '{feat}' mean after scaling = {mean_val:.4f}, expected ~0"
            )

    def test_scaled_features_have_near_unit_std(self):
        """StandardScaler should produce unit-variance features (tolerance 0.1)."""
        df = self._make_clean_frame(200)
        scaled_df, _ = pp.scale_features(df)
        for feat in FINAL_FEATURES:
            std_val = scaled_df[feat].std()
            assert abs(std_val - 1.0) < 0.1, (
                f"Feature '{feat}' std after scaling = {std_val:.4f}, expected ~1"
            )

    def test_returns_fitted_scaler(self):
        """scale_features must return the fitted StandardScaler object."""
        from sklearn.preprocessing import StandardScaler
        df = self._make_clean_frame(100)
        _, scaler = pp.scale_features(df)
        assert isinstance(scaler, StandardScaler), (
            "Second return value must be a fitted StandardScaler"
        )

    def test_scaler_has_correct_feature_count(self):
        df = self._make_clean_frame(100)
        _, scaler = pp.scale_features(df)
        assert len(scaler.mean_) == len(FINAL_FEATURES), (
            f"Scaler should have {len(FINAL_FEATURES)} means, got {len(scaler.mean_)}"
        )

    def test_target_column_not_scaled(self):
        """At_Risk must remain binary (0/1) after scaling - not transformed."""
        df = self._make_clean_frame(100)
        scaled_df, _ = pp.scale_features(df)
        unique_vals = set(scaled_df[TARGET_COL].unique())
        assert unique_vals.issubset({0, 1}), (
            f"At_Risk was scaled when it should stay binary, got {unique_vals}"
        )


# =============================================================================
# SECTION 7 - End-to-End Schema Validation (synthetic pipeline)
# =============================================================================

class TestEndToEndSchema:
    """
    Run the full in-memory preprocessing pipeline on synthetic data
    and confirm the final output matches the ML-ready schema.
    This validates that merge -> SMOTE -> scale produces a valid training frame.
    """

    def _make_synthetic_dataset(self, n=100, source="DS_SYN"):
        np.random.seed(99)
        n_at_risk = int(n * 0.25)
        df = pd.DataFrame({
            "Quizzes_Avg":         np.random.uniform(10, 95, n),
            "Assignments_Avg":     np.random.uniform(10, 95, n),
            "Midterm_Score":       np.random.uniform(10, 95, n),
            "Participation_Score": np.random.uniform(10, 95, n),
            "Projects_Score":      np.random.uniform(10, 95, n),
            TARGET_COL:           [1]*n_at_risk + [0]*(n - n_at_risk),
            "source":             source,
        })
        return df

    def test_full_pipeline_produces_valid_frame(self):
        ds = self._make_synthetic_dataset(120)
        merged  = pp.merge_training_datasets([ds])
        smoted  = pp.apply_smote(merged)
        final, scaler = pp.scale_features(smoted)

        # Schema check
        for feat in FINAL_FEATURES:
            assert feat in final.columns, f"Missing feature '{feat}' in final output"
        assert TARGET_COL in final.columns

        # No nulls
        assert final[FINAL_FEATURES].isnull().sum().sum() == 0

        # Target still binary
        assert set(final[TARGET_COL].unique()).issubset({0, 1})

        # Row count positive
        assert len(final) > 0

    def test_smote_applied_after_merge_balances_classes(self):
        ds = self._make_synthetic_dataset(200)
        merged = pp.merge_training_datasets([ds])
        smoted = pp.apply_smote(merged)
        counts = smoted[TARGET_COL].value_counts()
        assert counts[0] == counts[1], (
            f"Classes not equal after SMOTE: {counts.to_dict()}"
        )

    def test_scaler_transform_matches_training_mean(self):
        """
        Apply the saved scaler to a known input and verify the transform
        produces a standardised output within expected bounds.
        """
        ds = self._make_synthetic_dataset(200)
        merged = pp.merge_training_datasets([ds])
        smoted = pp.apply_smote(merged)
        final, scaler = pp.scale_features(smoted)

        # Create a test row with all features = scaler mean (should scale to ~0)
        test_row = pd.DataFrame([dict(zip(FINAL_FEATURES, scaler.mean_))])
        scaled_row = scaler.transform(test_row)
        for val in scaled_row[0]:
            assert abs(val) < 1e-9, (
                f"Input equal to mean should scale to ~0, got {val}"
            )


# =============================================================================
# SECTION 8 - File-Absent Graceful Failure (Integration guard)
# =============================================================================

class TestFileAbsentGracefulFailure:
    """
    Confirms that load_dataset functions return None gracefully when
    dataset files are missing - the pipeline should not crash.
    These tests pass whether or not real datasets are present.
    """

    def test_load_dataset1_missing_returns_none(self, tmp_path, monkeypatch):
        """load_dataset1() must return None when file is not found."""
        monkeypatch.setattr(pp, "DS1_FILE", str(tmp_path / "nonexistent.csv"))
        result = pp.load_dataset1()
        assert result is None, "Missing dataset should return None, not crash"

    def test_load_dataset2_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pp, "DS2_FILE", str(tmp_path / "nonexistent.csv"))
        result = pp.load_dataset2()
        assert result is None

    def test_load_dataset3_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pp, "DS3_FILE", str(tmp_path / "nonexistent.csv"))
        result = pp.load_dataset3()
        assert result is None

    def test_load_dataset4_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pp, "DS4_FILE", str(tmp_path / "nonexistent.csv"))
        result = pp.load_dataset4()
        assert result is None


# =============================================================================
# SECTION 9 - FINAL_FEATURES Constant Integrity
# =============================================================================

class TestConstantIntegrity:
    """
    Verify that the FINAL_FEATURES list in preprocess.py matches the
    expected 5-feature schema used by the ML model.
    A mismatch here would silently break the entire training pipeline.
    """

    EXPECTED_FEATURES = [
        "Quizzes_Avg",
        "Assignments_Avg",
        "Midterm_Score",
        "Participation_Score",
        "Projects_Score",
    ]

    def test_final_features_count(self):
        assert len(pp.FINAL_FEATURES) == 5, (
            f"Expected 5 features, got {len(pp.FINAL_FEATURES)}: {pp.FINAL_FEATURES}"
        )

    def test_final_features_exact_names(self):
        assert list(pp.FINAL_FEATURES) == self.EXPECTED_FEATURES, (
            f"FINAL_FEATURES mismatch.\n"
            f"Expected : {self.EXPECTED_FEATURES}\n"
            f"Got      : {list(pp.FINAL_FEATURES)}"
        )

    def test_target_col_is_atrisk(self):
        assert pp.TARGET_COL == "At_Risk", (
            f"TARGET_COL should be 'At_Risk', got '{pp.TARGET_COL}'"
        )

    def test_no_target_col_in_final_features(self):
        """At_Risk must not appear inside FINAL_FEATURES - it is the label."""
        assert "At_Risk" not in pp.FINAL_FEATURES, (
            "At_Risk is a label and must not appear in FINAL_FEATURES"
        )
