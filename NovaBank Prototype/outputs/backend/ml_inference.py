"""
ml_inference.py — Signal 1: Random Forest ML behavioural scorer.

Purpose: Load the trained Random Forest and calculate the claimed-user behavioural machine-learning score for a login attempt.
Inputs: Aggregate keystroke features and the username claimed during authentication.
Outputs: ML score and supporting model-status metadata, or a safe unavailable result when inference cannot be performed.


This module loads the trained Random Forest model and produces a behavioural
confidence score for a given login attempt.

─────────────────────────────────────────────────────────────────────────────
MODEL FORMULATION
─────────────────────────────────────────────────────────────────────────────
The RF is trained as a MULTI-CLASS CLASSIFIER — one class per enrolled user.
Given a feature vector, the model outputs a probability distribution over all
known users (class probabilities via predict_proba).

At authentication, the probability assigned to the CLAIMED user's class
is used as the ML behavioural score:

    ml_score = P(class = claimed_username | aggregate_features)

Interpretation:
    HIGH ml_score → the current typing rhythm looks like the claimed user
    LOW  ml_score → the typing rhythm does not match the claimed user's
                    learned pattern (potential impostor)

IMPORTANT CAVEAT: This probability is the model's CLASSIFICATION confidence,
not a calibrated probability of genuineness. Raw RF probabilities tend to be
overconfident. Consider Platt scaling or isotonic regression for calibration
if dissertation experiments show poor discrimination. Report the chosen
calibration method (or lack thereof) in the limitations section.

─────────────────────────────────────────────────────────────────────────────
FALLBACK BEHAVIOUR
─────────────────────────────────────────────────────────────────────────────
If the model file does not exist (not yet trained), ml_score = None is returned.
The risk engine handles this gracefully by relying on the profile score alone.
This is the expected state before the first model training run.

─────────────────────────────────────────────────────────────────────────────
USER NOT IN TRAINING LABELS
─────────────────────────────────────────────────────────────────────────────
If a user was enrolled AFTER the model was last trained, they will not appear
in the model's class list. In this case, ml_score = max(P) across all classes
is used as a proxy — measuring whether the typing is "typical" for ANY known
user. This is a conservative approximation; retrain the model to include the
new user for accurate results.
"""

import os
import pickle
import json
from typing import Optional, Dict, Any, List


_MODEL_PATH     = os.path.join(os.path.dirname(__file__), '..', 'model', 'keystroke_rf.pkl')
_LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'label_map.json')
_MODEL_VERSION  = 'rf_v1'


class MLInference:
    """Loads and serves the trained Random Forest model."""

    def __init__(self):
        self.model        = None
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        """Attempt to load model and label map from disk."""
        try:
            if os.path.exists(_MODEL_PATH):
                with open(_MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                print(f'[MLInference] Model loaded from {_MODEL_PATH}')

                if os.path.exists(_LABEL_MAP_PATH):
                    with open(_LABEL_MAP_PATH, 'r') as f:
                        data = json.load(f)
                    self.label_to_idx = data.get('label_to_idx', {})
                    self.idx_to_label = {int(v): k for k, v in self.label_to_idx.items()}
                    print(f'[MLInference] Labels: {list(self.label_to_idx.keys())}')
            else:
                print('[MLInference] No model file found. Run model/train_model.py to train.')
        except Exception as e:
            print(f'[MLInference] Load error: {e}')
            self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, aggregate_features: List[float], claimed_username: str) -> Dict[str, Any]:
        """
        Compute the ML behavioural score for a claimed user.

        Args:
            aggregate_features: 13-element list from features.extract_aggregate_features()
            claimed_username:   username the requester claims to be

        Returns dict:
            ml_score       : float in [0,1] or None if model not available
            model_version  : str
            prediction     : str — top predicted class (for display/logging)
            in_training_set: bool — whether the claimed user was in training data
        """
        if self.model is None:
            return {
                'ml_score':        None,
                'model_version':   'not_trained',
                'prediction':      'unavailable',
                'in_training_set': False,
            }

        try:
            X     = [aggregate_features]
            proba = self.model.predict_proba(X)[0]   # shape: (n_classes,)

            in_training_set = claimed_username in self.label_to_idx

            if in_training_set:
                idx      = self.label_to_idx[claimed_username]
                ml_score = float(proba[idx])
            else:
                # New user not in training set — use max probability as proxy
                ml_score = float(max(proba))

            top_idx   = int(proba.argmax())
            predicted = self.idx_to_label.get(top_idx, 'unknown')

            return {
                'ml_score':        round(ml_score, 4),
                'model_version':   _MODEL_VERSION,
                'prediction':      predicted,
                'in_training_set': in_training_set,
            }

        except Exception as e:
            print(f'[MLInference] Prediction error: {e}')
            return {
                'ml_score':        None,
                'model_version':   _MODEL_VERSION,
                'prediction':      'error',
                'in_training_set': False,
            }
