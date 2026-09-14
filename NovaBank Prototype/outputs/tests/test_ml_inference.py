"""
test_ml_inference.py — Unit tests for backend/ml_inference.py.

Purpose: Test Random Forest inference handling without loading or changing the real trained model.
Inputs: Mocked model probabilities, claimed-user labels, feature vectors and missing/error conditions.
Outputs: Pytest pass/fail assertions for claimed-user scoring, training-set flags and safe fallback behaviour.


Tests verify:
    - Missing model file returns ml_score=None (safe fallback)
    - is_loaded() reports False when model is absent
    - Claimed-user class probability is correctly selected from predict_proba output
    - Unknown users (not in training labels) receive max-probability proxy score
    - in_training_set flag is set correctly
    - All required keys are present in the return dict
    - Prediction errors are caught and return ml_score=None

Uses unittest.mock to avoid loading or modifying the real keystroke_rf.pkl.
The real trained model is NEVER touched by these tests.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from ml_inference import MLInference


# ── Shared constants ─────────────────────────────────────────────────────────
# 13 aggregate features (values are representative, not meaningful here)
DUMMY_FEATURES = [80.0, 15.0, 30.0, 10.0, 1200.0, 8.0, 6.5,
                  0.2, 0.3, 60.0, 120.0, 20.0, 50.0]

# Three-class RF: Radhika=0, Asha=1, Varun=2
LABEL_MAP  = {"label_to_idx": {"Radhika": 0, "Asha": 1, "Varun": 2}}
# predict_proba output: Radhika=0.15, Asha=0.10, Varun=0.75
MOCK_PROBA = np.array([[0.15, 0.10, 0.75]])


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def inference_no_model():
    """MLInference with no model file on disk."""
    with patch("ml_inference.os.path.exists", return_value=False):
        obj = MLInference()
    return obj


@pytest.fixture
def inference_with_mock_model():
    """MLInference with a fully mocked Random Forest (real pkl never loaded)."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = MOCK_PROBA

    with patch("ml_inference.os.path.exists", return_value=False):
        obj = MLInference()

    # Manually inject the mock so no file I/O occurs
    obj.model        = mock_model
    obj.label_to_idx = {"Radhika": 0, "Asha": 1, "Varun": 2}
    obj.idx_to_label = {0: "Radhika", 1: "Asha", 2: "Varun"}
    return obj


# ── No model available ────────────────────────────────────────────────────────

class TestNoModel:
    def test_is_loaded_false_when_no_model(self, inference_no_model):
        assert inference_no_model.is_loaded() is False

    def test_predict_returns_none_ml_score(self, inference_no_model):
        result = inference_no_model.predict(DUMMY_FEATURES, "Radhika")
        assert result["ml_score"] is None

    def test_model_version_says_not_trained(self, inference_no_model):
        result = inference_no_model.predict(DUMMY_FEATURES, "Radhika")
        assert result["model_version"] == "not_trained"

    def test_in_training_set_false_when_no_model(self, inference_no_model):
        result = inference_no_model.predict(DUMMY_FEATURES, "Radhika")
        assert result["in_training_set"] is False


# ── Model available ────────────────────────────────────────────────────────────

class TestWithMockModel:
    def test_is_loaded_true_with_model(self, inference_with_mock_model):
        assert inference_with_mock_model.is_loaded() is True

    def test_claimed_user_class_probability_selected(self, inference_with_mock_model):
        """P(Varun | features) = proba[2] = 0.75"""
        result = inference_with_mock_model.predict(DUMMY_FEATURES, "Varun")
        assert result["ml_score"] == pytest.approx(0.75)

    def test_different_user_gets_different_class_probability(self, inference_with_mock_model):
        """P(Radhika | features) = proba[0] = 0.15"""
        result = inference_with_mock_model.predict(DUMMY_FEATURES, "Radhika")
        assert result["ml_score"] == pytest.approx(0.15)

    def test_in_training_set_true_for_known_user(self, inference_with_mock_model):
        result = inference_with_mock_model.predict(DUMMY_FEATURES, "Asha")
        assert result["in_training_set"] is True

    def test_score_in_range(self, inference_with_mock_model):
        result = inference_with_mock_model.predict(DUMMY_FEATURES, "Radhika")
        assert 0.0 <= result["ml_score"] <= 1.0

    def test_result_has_required_keys(self, inference_with_mock_model):
        result = inference_with_mock_model.predict(DUMMY_FEATURES, "Radhika")
        for key in ("ml_score", "model_version", "prediction", "in_training_set"):
            assert key in result, f"Missing key: {key}"

    def test_top_predicted_class_is_varun(self, inference_with_mock_model):
        """argmax([0.15, 0.10, 0.75]) → index 2 → 'Varun'"""
        result = inference_with_mock_model.predict(DUMMY_FEATURES, "Radhika")
        assert result["prediction"] == "Varun"


# ── Unknown user (not in training set) ────────────────────────────────────────

class TestUnknownUser:
    def test_unknown_user_returns_max_probability(self, inference_with_mock_model):
        """User not in training set → use max(proba) as conservative proxy."""
        result = inference_with_mock_model.predict(DUMMY_FEATURES, "UnknownUser")
        # max([0.15, 0.10, 0.75]) = 0.75
        assert result["ml_score"] == pytest.approx(0.75)

    def test_unknown_user_in_training_set_false(self, inference_with_mock_model):
        result = inference_with_mock_model.predict(DUMMY_FEATURES, "UnknownUser")
        assert result["in_training_set"] is False


# ── Error handling ─────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_prediction_error_returns_none_score(self, inference_with_mock_model):
        """If predict_proba raises, ml_score must be None (not propagate exception)."""
        inference_with_mock_model.model.predict_proba.side_effect = RuntimeError("GPU error")
        result = inference_with_mock_model.predict(DUMMY_FEATURES, "Radhika")
        assert result["ml_score"] is None
        assert result["prediction"] == "error"
