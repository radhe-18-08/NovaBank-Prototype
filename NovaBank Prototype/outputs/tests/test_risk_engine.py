"""
test_risk_engine.py — Unit tests for backend/risk_engine.py.

Purpose: Test the operational risk-engine thresholds, signal combination and safe fallback rules.
Inputs: Synthetic credential results, ML/profile scores and data-sufficiency combinations around decision boundaries.
Outputs: Pytest pass/fail assertions for combined scores and expected ALLOW/VERIFY/BLOCK outcomes.


Tests verify all boundary conditions around the three decision thresholds:
    ALLOW_THRESHOLD  = 0.55
    VERIFY_THRESHOLD = 0.28
    ML_WEIGHT        = 0.50
    PROFILE_WEIGHT   = 0.50

Key invariants:
    - Wrong credentials → always BLOCK, regardless of behavioural scores
    - combined_score >= 0.55           → ALLOW  (LOW risk)
    - 0.28 <= combined_score < 0.55    → VERIFY (MEDIUM risk)
    - combined_score < 0.28            → BLOCK  (HIGH risk)
    - Missing ML score: fall back to profile_score alone, no fabrication
    - Insufficient behavioural data → VERIFY (not BLOCK — no strong evidence)
    - No behavioural data at all      → VERIFY (not BLOCK)

IMPORTANT: These tests validate the operational thresholds as used in the
final submitted prototype. They must NOT be changed to make results look
better. If a test exposes a genuine bug it is reported here first.
"""
import pytest
from risk_engine import evaluate, ALLOW_THRESHOLD, VERIFY_THRESHOLD, ML_WEIGHT, PROFILE_WEIGHT


# ── Constant values ──────────────────────────────────────────────────────────

class TestConstants:
    def test_allow_threshold_is_0_55(self):
        assert ALLOW_THRESHOLD == pytest.approx(0.55)

    def test_verify_threshold_is_0_28(self):
        assert VERIFY_THRESHOLD == pytest.approx(0.28)

    def test_ml_weight_is_0_50(self):
        assert ML_WEIGHT == pytest.approx(0.50)

    def test_profile_weight_is_0_50(self):
        assert PROFILE_WEIGHT == pytest.approx(0.50)

    def test_weights_sum_to_one(self):
        assert ML_WEIGHT + PROFILE_WEIGHT == pytest.approx(1.0)

    def test_allow_threshold_above_verify_threshold(self):
        assert ALLOW_THRESHOLD > VERIFY_THRESHOLD


# ── Credential gate (Step 1) ─────────────────────────────────────────────────

class TestCredentialGate:
    def test_wrong_credentials_gives_block(self):
        result = evaluate({"credential_result": False, "ml_score": 0.99, "profile_score": 0.99})
        assert result["decision"] == "BLOCK"

    def test_wrong_credentials_gives_high_risk(self):
        result = evaluate({"credential_result": False, "ml_score": 0.99, "profile_score": 0.99})
        assert result["risk_level"] == "HIGH"

    def test_wrong_credentials_ignores_perfect_behavioural_scores(self):
        """No amount of perfect keystroke data overrides a failed password check."""
        result = evaluate({"credential_result": False, "ml_score": 1.0, "profile_score": 1.0})
        assert result["decision"] == "BLOCK"

    def test_wrong_credentials_with_no_scores_still_blocks(self):
        result = evaluate({"credential_result": False, "ml_score": None, "profile_score": None})
        assert result["decision"] == "BLOCK"


# ── ALLOW boundary ───────────────────────────────────────────────────────────

class TestAllowBoundary:
    def test_score_exactly_at_allow_threshold_gives_allow(self):
        s = ALLOW_THRESHOLD   # 0.55
        result = evaluate({"credential_result": True, "ml_score": s, "profile_score": s})
        assert result["decision"] == "ALLOW"

    def test_score_above_allow_threshold_gives_allow(self):
        result = evaluate({"credential_result": True, "ml_score": 0.80, "profile_score": 0.80})
        assert result["decision"] == "ALLOW"
        assert result["risk_level"] == "LOW"

    def test_perfect_scores_give_allow(self):
        result = evaluate({"credential_result": True, "ml_score": 1.0, "profile_score": 1.0})
        assert result["decision"] == "ALLOW"


# ── VERIFY boundary ──────────────────────────────────────────────────────────

class TestVerifyBoundary:
    def test_score_just_below_allow_gives_verify(self):
        """0.54 < 0.55 → VERIFY"""
        s = ALLOW_THRESHOLD - 0.01
        result = evaluate({"credential_result": True, "ml_score": s, "profile_score": s})
        assert result["decision"] == "VERIFY"
        assert result["risk_level"] == "MEDIUM"

    def test_score_exactly_at_verify_threshold_gives_verify(self):
        """0.28 is on the boundary → still VERIFY (not BLOCK)"""
        s = VERIFY_THRESHOLD   # 0.28
        result = evaluate({"credential_result": True, "ml_score": s, "profile_score": s})
        assert result["decision"] == "VERIFY"

    def test_midpoint_score_gives_verify(self):
        """Score halfway between thresholds → VERIFY"""
        s = (ALLOW_THRESHOLD + VERIFY_THRESHOLD) / 2   # 0.415
        result = evaluate({"credential_result": True, "ml_score": s, "profile_score": s})
        assert result["decision"] == "VERIFY"


# ── BLOCK boundary ───────────────────────────────────────────────────────────

class TestBlockBoundary:
    def test_score_just_below_verify_threshold_gives_block(self):
        """0.27 < 0.28 → BLOCK"""
        s = VERIFY_THRESHOLD - 0.01
        result = evaluate({"credential_result": True, "ml_score": s, "profile_score": s})
        assert result["decision"] == "BLOCK"
        assert result["risk_level"] == "HIGH"

    def test_very_low_score_gives_block(self):
        result = evaluate({"credential_result": True, "ml_score": 0.01, "profile_score": 0.01})
        assert result["decision"] == "BLOCK"

    def test_zero_score_gives_block(self):
        result = evaluate({"credential_result": True, "ml_score": 0.0, "profile_score": 0.0})
        assert result["decision"] == "BLOCK"


# ── Signal combination weights ───────────────────────────────────────────────

class TestCombinationWeights:
    def test_combined_score_is_weighted_average(self):
        ml_s, pr_s = 0.40, 0.60
        expected = ML_WEIGHT * ml_s + PROFILE_WEIGHT * pr_s
        result = evaluate({"credential_result": True, "ml_score": ml_s, "profile_score": pr_s})
        assert result["combined_score"] == pytest.approx(expected, abs=0.0001)

    def test_equal_scores_combined_equals_each(self):
        """50/50 weights on equal scores → combined = the score itself."""
        s = 0.70
        result = evaluate({"credential_result": True, "ml_score": s, "profile_score": s})
        assert result["combined_score"] == pytest.approx(s, abs=0.0001)


# ── Missing ML score ─────────────────────────────────────────────────────────

class TestMissingMlScore:
    def test_ml_none_uses_profile_alone(self):
        """ml_score=None → fall back to profile_score, no fabricated ML value."""
        result = evaluate({"credential_result": True, "ml_score": None, "profile_score": 0.80})
        assert result["decision"] == "ALLOW"         # 0.80 ≥ 0.55

    def test_ml_none_reports_ml_unavailable(self):
        result = evaluate({"credential_result": True, "ml_score": None, "profile_score": 0.80})
        assert result["ml_available"] is False

    def test_ml_none_combined_score_equals_profile_score(self):
        result = evaluate({"credential_result": True, "ml_score": None, "profile_score": 0.65})
        assert result["combined_score"] == pytest.approx(0.65)

    def test_ml_only_no_profile(self):
        """profile_score=None but ML available → use ML alone."""
        result = evaluate({"credential_result": True, "ml_score": 0.70, "profile_score": None})
        assert result["decision"] == "ALLOW"
        assert result["combined_score"] == pytest.approx(0.70)


# ── No behavioural evidence ───────────────────────────────────────────────────

class TestNoBehaviouralEvidence:
    def test_both_signals_none_gives_verify(self):
        """No evidence → step-up verification, never hard-block."""
        result = evaluate({
            "credential_result":    True,
            "ml_score":             None,
            "profile_score":        None,
        })
        assert result["decision"] == "VERIFY"

    def test_insufficient_profile_no_ml_gives_verify(self):
        """insufficient_data=True with no ML → VERIFY."""
        result = evaluate({
            "credential_result":    True,
            "ml_score":             None,
            "profile_score":        None,
            "profile_insufficient": True,
        })
        assert result["decision"] == "VERIFY"

    def test_no_evidence_risk_level_is_medium(self):
        result = evaluate({
            "credential_result": True,
            "ml_score":          None,
            "profile_score":     None,
        })
        assert result["risk_level"] == "MEDIUM"


# ── Return structure ──────────────────────────────────────────────────────────

class TestReturnStructure:
    def test_has_all_required_keys(self):
        result = evaluate({"credential_result": True, "ml_score": 0.70, "profile_score": 0.70})
        for key in ("decision", "risk_level", "combined_score",
                    "ml_available", "combination_method", "reason"):
            assert key in result, f"Missing key: {key}"

    def test_decision_values_are_valid(self):
        for score in (0.10, 0.35, 0.70):
            result = evaluate({"credential_result": True, "ml_score": score, "profile_score": score})
            assert result["decision"] in ("ALLOW", "VERIFY", "BLOCK")

    def test_risk_level_values_are_valid(self):
        for score in (0.10, 0.35, 0.70):
            result = evaluate({"credential_result": True, "ml_score": score, "profile_score": score})
            assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
