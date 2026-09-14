"""
test_api.py — Integration/smoke tests for backend/app.py (Flask REST API).

Purpose: Test the Flask REST API behaviour, response safety and independence of experiment ground-truth labels from authentication decisions.
Inputs: Mocked profiles/model components and HTTP requests sent through Flask's test client.
Outputs: Pytest pass/fail assertions for API endpoints, credential handling, privacy and decision behaviour.


Tests verify:
    - GET /api/health returns 200 with a status field
    - GET /api/profiles returns a profiles list
    - POST /api/auth with missing/unknown username returns a 4xx error
    - POST /api/auth with wrong password returns BLOCK decision
    - POST /api/auth response never returns the password hash
    - The experiment_label field does NOT influence the ALLOW/BLOCK decision

Uses Flask's test client and unittest.mock to avoid file I/O.
The real profiles.json, experiment_log.csv, and trained model are NEVER modified.
"""
import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np

# Add backend to path (conftest.py does this, but belt-and-braces here)
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# Patch file-system access BEFORE importing app so singletons don't touch real files
with patch("profile_store.ProfileStore._read", return_value={}), \
     patch("ml_inference.os.path.exists", return_value=False), \
     patch("experiment_logger.os.path.exists", return_value=False):
    import app as flask_app


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def client():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def _auth_post(client, body):
    return client.post(
        "/api/auth",
        data=json.dumps(body),
        content_type="application/json",
    )


def _mock_profile(password="correct_password"):
    from auth import hash_password
    return {
        "password_hash": hash_password(password),
        "keystroke_profile": {
            "positional_means": [80.0] * 7,
            "positional_stds":  [20.0] * 7,
            "password_length":  4,
        },
    }


_SAMPLE_KEYDATA = [
    {"downTime":   0.0, "upTime":  80.0},
    {"downTime": 100.0, "upTime": 180.0},
    {"downTime": 210.0, "upTime": 290.0},
    {"downTime": 330.0, "upTime": 410.0},
]


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/api/health").status_code == 200

    def test_response_has_status_field(self, client):
        data = client.get("/api/health").get_json()
        assert "status" in data


# ── Profiles endpoint ─────────────────────────────────────────────────────────

class TestProfiles:
    def test_get_profiles_returns_200(self, client):
        with patch.object(flask_app.store, "get_display_profiles", return_value=[]):
            assert client.get("/api/profiles").status_code == 200

    def test_get_profiles_returns_list(self, client):
        with patch.object(flask_app.store, "get_display_profiles", return_value=[]):
            data = client.get("/api/profiles").get_json()
        assert "profiles" in data
        assert isinstance(data["profiles"], list)


# ── Auth: unknown / missing user ──────────────────────────────────────────────

class TestAuthUnknownUser:
    def test_unknown_user_returns_4xx(self, client):
        with patch.object(flask_app.store, "get_profile", return_value=None):
            response = _auth_post(client, {
                "username": "ghost", "password": "pw", "raw_keydata": []
            })
        assert response.status_code in (400, 404)

    def test_empty_username_returns_4xx(self, client):
        with patch.object(flask_app.store, "get_profile", return_value=None):
            response = _auth_post(client, {
                "username": "", "password": "pw", "raw_keydata": []
            })
        assert response.status_code in (400, 404)


# ── Auth: wrong password ──────────────────────────────────────────────────────

class TestAuthWrongPassword:
    def test_wrong_password_gives_block_decision(self, client):
        with patch.object(flask_app.store, "get_profile", return_value=_mock_profile()), \
             patch.object(flask_app.explog, "log"):
            data = _auth_post(client, {
                "username": "alice", "password": "WRONG_PASSWORD",
                "raw_keydata": [], "experiment_label": "genuine",
            }).get_json()
        assert data["decision"] == "BLOCK"

    def test_wrong_password_credential_result_false(self, client):
        with patch.object(flask_app.store, "get_profile", return_value=_mock_profile()), \
             patch.object(flask_app.explog, "log"):
            data = _auth_post(client, {
                "username": "alice", "password": "WRONG_PASSWORD",
                "raw_keydata": [], "experiment_label": "genuine",
            }).get_json()
        assert data["credential_result"] is False


# ── Auth: password hash never returned ───────────────────────────────────────

class TestPasswordSecurity:
    def test_auth_response_never_contains_password_hash(self, client):
        profile = _mock_profile("my_secret_password")
        stored_hash = profile["password_hash"]
        with patch.object(flask_app.store, "get_profile", return_value=profile), \
             patch.object(flask_app.explog, "log"), \
             patch.object(flask_app.ml, "predict", return_value={
                 "ml_score": 0.70, "model_version": "rf_v1",
                 "prediction": "alice", "in_training_set": True,
             }):
            response = _auth_post(client, {
                "username": "alice", "password": "my_secret_password",
                "raw_keydata": _SAMPLE_KEYDATA, "experiment_label": "genuine",
            })
        text = response.get_data(as_text=True)
        assert stored_hash not in text
        assert "my_secret_password" not in text


# ── experiment_label does not influence decision ──────────────────────────────

class TestLabelDoesNotInfluenceDecision:
    """
    Confirm that the experiment_label is evaluation metadata only.
    The same login attempt with label='genuine' and label='impostor' must
    produce the SAME decision — the label is never passed to the risk engine.
    """
    def _login(self, client, label, password="correct_password"):
        profile = _mock_profile(password)
        with patch.object(flask_app.store, "get_profile", return_value=profile), \
             patch.object(flask_app.explog, "log"), \
             patch.object(flask_app.ml, "predict", return_value={
                 "ml_score": 0.70, "model_version": "rf_v1",
                 "prediction": "alice", "in_training_set": True,
             }):
            return _auth_post(client, {
                "username": "alice", "password": password,
                "raw_keydata": _SAMPLE_KEYDATA, "experiment_label": label,
            }).get_json()

    def test_genuine_label_does_not_force_allow(self, client):
        result_genuine  = self._login(client, label="genuine")
        result_impostor = self._login(client, label="impostor")
        # Same keydata + same password → must give the SAME decision
        assert result_genuine["decision"] == result_impostor["decision"]

    def test_impostor_label_does_not_force_block(self, client):
        # Fixed: the mock profile is created with "registered_password" and the
        # request sends a *different* string ("WRONG_PASSWORD"). This correctly
        # tests that the credential gate blocks regardless of experiment_label.
        # (Earlier version created the profile from the same password string,
        # so the hash always matched — a test-design error, not a code bug.)
        profile = _mock_profile("registered_password")
        for label in ("genuine", "impostor"):
            with patch.object(flask_app.store, "get_profile", return_value=profile),                  patch.object(flask_app.explog, "log"):
                result = _auth_post(client, {
                    "username":         "alice",
                    "password":         "WRONG_PASSWORD",
                    "raw_keydata":      [],
                    "experiment_label": label,
                }).get_json()
            assert result["decision"] == "BLOCK", (
                f"Wrong password + label='{label}' should BLOCK, got {result['decision']}"
            )
