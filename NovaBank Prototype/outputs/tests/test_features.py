"""
test_features.py — Unit tests for backend/features.py (keystroke feature extraction).

Purpose: Test positional and aggregate keystroke feature extraction and profile-statistic calculations.
Inputs: Synthetic keydown/keyup timing samples with known expected timings.
Outputs: Pytest pass/fail assertions for hold/flight values, 13-feature vectors, edge cases and profile statistics.


Tests verify:
    - extract_positional_features produces a (2n-1)-length vector
    - Hold and flight times are computed correctly
    - Negative flight times (overlapping keys) are clamped to zero
    - extract_aggregate_features returns exactly 13 features
    - Each aggregate feature has the expected value for a known input
    - Edge cases: empty input, single keystroke
    - build_profile_stats computes means/stds and applies STD_FLOOR_MS
"""
import pytest
from features import (
    extract_positional_features,
    extract_aggregate_features,
    build_profile_stats,
    STD_FLOOR_MS,
    _raw_timings,
)

# ── Shared fixtures ──────────────────────────────────────────────────────────

# 4-keystroke sequence with known timing values:
#   key 0: hold=80ms
#   key 1: hold=80ms, flight from key0 = 100-80 = 20ms
#   key 2: hold=80ms, flight from key1 = 210-180 = 30ms
#   key 3: hold=80ms, flight from key2 = 330-290 = 40ms
SAMPLE_4 = [
    {"downTime":   0.0, "upTime":  80.0},
    {"downTime": 100.0, "upTime": 180.0},
    {"downTime": 210.0, "upTime": 290.0},
    {"downTime": 330.0, "upTime": 410.0},
]

SAMPLE_1 = [{"downTime": 0.0, "upTime": 60.0}]


# ══════════════════════════════════════════════════════════════════════════════
# extract_positional_features
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionalFeatures:
    def test_length_is_2n_minus_1(self):
        """For n keystrokes, output length must be 2n-1."""
        feats = extract_positional_features(SAMPLE_4)
        assert len(feats) == 2 * 4 - 1  # 7

    def test_hold_times_correct(self):
        feats = extract_positional_features(SAMPLE_4)
        # First n values are hold times
        assert feats[0] == pytest.approx(80.0)
        assert feats[1] == pytest.approx(80.0)
        assert feats[2] == pytest.approx(80.0)
        assert feats[3] == pytest.approx(80.0)

    def test_flight_times_correct(self):
        feats = extract_positional_features(SAMPLE_4)
        # Values at indices n..2n-2 are flight times
        assert feats[4] == pytest.approx(20.0)
        assert feats[5] == pytest.approx(30.0)
        assert feats[6] == pytest.approx(40.0)

    def test_overlapping_keys_flight_clamped_to_zero(self):
        """When key i+1 is pressed before key i is released, flight = 0."""
        overlapping = [
            {"downTime":  0.0, "upTime": 100.0},
            {"downTime": 50.0, "upTime": 150.0},   # flight = 50 - 100 = -50 → 0
        ]
        feats = extract_positional_features(overlapping)
        assert feats[2] == pytest.approx(0.0)

    def test_empty_input_returns_empty_list(self):
        assert extract_positional_features([]) == []

    def test_single_key_has_no_flights(self):
        feats = extract_positional_features(SAMPLE_1)
        # 1 key → 2(1)-1 = 1 feature (hold only, no flights)
        assert len(feats) == 1
        assert feats[0] == pytest.approx(60.0)


# ══════════════════════════════════════════════════════════════════════════════
# extract_aggregate_features
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregateFeatures:
    def test_returns_exactly_13_features(self):
        feats = extract_aggregate_features(SAMPLE_4)
        assert len(feats) == 13

    def test_empty_input_returns_13_zeros(self):
        feats = extract_aggregate_features([])
        assert feats == [0.0] * 13

    def test_mean_hold_correct(self):
        feats = extract_aggregate_features(SAMPLE_4)
        assert feats[0] == pytest.approx(80.0)  # all holds = 80ms

    def test_std_hold_correct(self):
        feats = extract_aggregate_features(SAMPLE_4)
        assert feats[1] == pytest.approx(0.0)   # identical holds → std=0

    def test_mean_flight_correct(self):
        feats = extract_aggregate_features(SAMPLE_4)
        # flights: 20, 30, 40 → mean = 30ms
        assert feats[2] == pytest.approx(30.0)

    def test_n_keys_correct(self):
        feats = extract_aggregate_features(SAMPLE_4)
        assert feats[5] == pytest.approx(4.0)

    def test_typing_speed_positive(self):
        feats = extract_aggregate_features(SAMPLE_4)
        assert feats[6] > 0.0

    def test_cv_hold_capped_at_10(self):
        feats = extract_aggregate_features(SAMPLE_4)
        assert feats[7] <= 10.0

    def test_cv_flight_capped_at_10(self):
        feats = extract_aggregate_features(SAMPLE_4)
        assert feats[8] <= 10.0

    def test_min_hold_correct(self):
        feats = extract_aggregate_features(SAMPLE_4)
        assert feats[9] == pytest.approx(80.0)

    def test_max_hold_correct(self):
        feats = extract_aggregate_features(SAMPLE_4)
        assert feats[10] == pytest.approx(80.0)

    def test_min_flight_correct(self):
        feats = extract_aggregate_features(SAMPLE_4)
        assert feats[11] == pytest.approx(20.0)  # minimum flight

    def test_max_flight_correct(self):
        feats = extract_aggregate_features(SAMPLE_4)
        assert feats[12] == pytest.approx(40.0)  # maximum flight

    def test_single_key_flights_are_zero(self):
        feats = extract_aggregate_features(SAMPLE_1)
        assert feats[2]  == pytest.approx(0.0)   # mean_flight
        assert feats[11] == pytest.approx(0.0)   # min_flight
        assert feats[12] == pytest.approx(0.0)   # max_flight

    def test_all_features_are_floats(self):
        feats = extract_aggregate_features(SAMPLE_4)
        for f in feats:
            assert isinstance(f, float)


# ══════════════════════════════════════════════════════════════════════════════
# build_profile_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildProfileStats:
    def test_returns_means_and_stds_keys(self):
        stats = build_profile_stats([[80.0, 20.0], [90.0, 25.0]])
        assert "means" in stats
        assert "stds" in stats

    def test_means_correct(self):
        stats = build_profile_stats([[80.0, 20.0], [100.0, 40.0]])
        assert stats["means"][0] == pytest.approx(90.0)
        assert stats["means"][1] == pytest.approx(30.0)

    def test_std_floor_applied_for_identical_samples(self):
        """When all samples are identical, std=0 → must be floored at STD_FLOOR_MS."""
        samples = [[100.0, 50.0], [100.0, 50.0], [100.0, 50.0]]
        stats = build_profile_stats(samples)
        for sd in stats["stds"]:
            assert sd >= STD_FLOOR_MS

    def test_empty_samples_returns_empty(self):
        stats = build_profile_stats([])
        assert stats == {"means": [], "stds": []}

    def test_output_lengths_match_feature_count(self):
        samples = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        stats = build_profile_stats(samples)
        assert len(stats["means"]) == 3
        assert len(stats["stds"])  == 3
