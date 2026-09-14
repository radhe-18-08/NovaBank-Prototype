"""
test_profile_deviation.py — Unit tests for backend/profile_deviation.py.

Purpose: Test personalised profile-deviation scoring across matching, differing and insufficient keystroke samples.
Inputs: Synthetic current timing vectors and enrolled profile statistics.
Outputs: Pytest pass/fail assertions for score bounds, similarity behaviour, trimming, standard-deviation floor and sufficiency flags.


Tests verify:
    - Perfect typing match produces score ≈ 1.0
    - Close-to-profile typing produces a high score
    - Strongly different typing produces a low score
    - Score is always bounded to [0.0, 1.0]
    - STD_FLOOR_MS prevents division by zero
    - TRIM_FRACTION correctly excludes worst outliers
    - MIN_VALID_FEATURES flag is set correctly
    - Empty/degenerate inputs handled safely
    - All required keys are present in the return dict
"""
import pytest
from profile_deviation import (
    calculate_profile_deviation,
    DIVISOR,
    TRIM_FRACTION,
    MIN_VALID_FEATURES,
)
from features import STD_FLOOR_MS

# ── Shared profile: mean=100ms, std=20ms for 5 features ─────────────────────
MEANS_5 = [100.0] * 5
STDS_5  = [20.0]  * 5


class TestScoreValues:
    def test_perfect_match_gives_score_one(self):
        """Typing exactly the profile mean → z=0 → score=1.0."""
        result = calculate_profile_deviation(MEANS_5, MEANS_5, STDS_5)
        assert result["score"] == pytest.approx(1.0)

    def test_close_match_gives_high_score(self):
        """Small deviation from profile → score should be above 0.7."""
        slight = [102.0, 98.0, 101.0, 99.0, 100.0]
        result = calculate_profile_deviation(slight, MEANS_5, STDS_5)
        assert result["score"] > 0.7

    def test_large_deviation_gives_low_score(self):
        """Typing ≈10× the std away → score should be below 0.3."""
        very_different = [300.0] * 5   # z ≈ (300-100)/20 = 10
        result = calculate_profile_deviation(very_different, MEANS_5, STDS_5)
        assert result["score"] < 0.3

    def test_extreme_deviation_score_floored_at_zero(self):
        """Score cannot go below 0."""
        extreme = [10_000.0] * 5
        result = calculate_profile_deviation(extreme, MEANS_5, STDS_5)
        assert result["score"] >= 0.0

    def test_score_always_at_most_one(self):
        result = calculate_profile_deviation(MEANS_5, MEANS_5, STDS_5)
        assert result["score"] <= 1.0


class TestStdFloor:
    def test_zero_std_does_not_cause_division_by_zero(self):
        """STD_FLOOR_MS prevents division by zero when all samples were identical."""
        zero_stds = [0.0] * 5
        result = calculate_profile_deviation(MEANS_5, MEANS_5, zero_stds)
        # Should not raise; perfect match → score = 1.0
        assert result["score"] == pytest.approx(1.0)

    def test_std_below_floor_treated_as_floor(self):
        """std=1ms is below STD_FLOOR_MS; z-score must use the floor value."""
        tiny_stds = [1.0] * 5
        # typing 1ms away from mean; without floor z would be 1, with floor z is tiny
        current = [101.0] * 5
        result = calculate_profile_deviation(current, MEANS_5, tiny_stds)
        # Score should still be high — tiny absolute deviation
        assert result["score"] > 0.9


class TestTrimFraction:
    def test_trimmed_features_count_is_correct(self):
        """With 5 features and TRIM_FRACTION=0.20, expect floor(5*0.20)=1 trimmed."""
        result = calculate_profile_deviation(MEANS_5, MEANS_5, STDS_5)
        expected_trim = max(0, int(5 * TRIM_FRACTION))
        assert result["trimmed_features"] == expected_trim

    def test_outlier_is_excluded(self):
        """Trimming should absorb a single large outlier.

        With 5 features and TRIM_FRACTION=0.20: trim_count = floor(5*0.20) = 1.
        One feature deviates enormously (z≈45); after trimming it is excluded
        so the remaining z-scores are all 0.  Score = 1.0, which is ≥ the
        untrimmed score (comparison against perfect is therefore equal, not
        strictly less).  The correct assertion is that:
          (a) at least one feature is trimmed, AND
          (b) the score remains high (outlier absorbed).
        The earlier assertion `score_with < score_without` was too strict
        for this edge case — a test-design error, not a code bug.
        """
        current_with_outlier = [100.0, 100.0, 100.0, 100.0, 1000.0]
        result = calculate_profile_deviation(current_with_outlier, MEANS_5, STDS_5)
        # Verify trimming occurred
        assert result["trimmed_features"] >= 1, "Expected at least one feature to be trimmed"
        # After trimming the outlier, remaining z-scores are all 0 → score = 1.0
        assert result["score"] >= 0.9, (
            f"Trimming should keep score high; got {result['score']}"
        )


class TestInsufficientDataFlag:
    def test_fewer_than_min_valid_features_sets_flag(self):
        """< MIN_VALID_FEATURES features → insufficient_data=True."""
        short = [100.0, 100.0]   # only 2 features
        result = calculate_profile_deviation(short, short, [20.0, 20.0])
        assert result["insufficient_data"] is True

    def test_enough_features_clears_flag(self):
        result = calculate_profile_deviation(MEANS_5, MEANS_5, STDS_5)
        assert result["insufficient_data"] is False

    def test_empty_input_sets_insufficient_flag(self):
        result = calculate_profile_deviation([], [], [])
        assert result["insufficient_data"] is True

    def test_empty_input_does_not_raise(self):
        result = calculate_profile_deviation([], [], [])
        assert 0.0 <= result["score"] <= 1.0


class TestReturnStructure:
    def test_has_all_required_keys(self):
        result = calculate_profile_deviation(MEANS_5, MEANS_5, STDS_5)
        for key in ("score", "avg_z", "valid_features", "trimmed_features", "insufficient_data"):
            assert key in result, f"Missing key: {key}"

    def test_avg_z_is_zero_for_perfect_match(self):
        result = calculate_profile_deviation(MEANS_5, MEANS_5, STDS_5)
        assert result["avg_z"] == pytest.approx(0.0)

    def test_valid_features_count_equals_input_length(self):
        result = calculate_profile_deviation(MEANS_5, MEANS_5, STDS_5)
        assert result["valid_features"] == 5

    def test_score_is_float(self):
        result = calculate_profile_deviation(MEANS_5, MEANS_5, STDS_5)
        assert isinstance(result["score"], float)
