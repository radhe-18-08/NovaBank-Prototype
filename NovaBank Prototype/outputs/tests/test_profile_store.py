"""
test_profile_store.py — Unit tests for backend/profile_store.py.

Purpose: Test profile storage, retrieval, deletion and privacy behaviour using temporary JSON files.
Inputs: Synthetic profile records and pytest temporary paths.
Outputs: Pytest pass/fail assertions for persistence operations and exclusion of password hashes from public/training views.


Tests verify:
    - Profiles can be saved and retrieved
    - get_display_profiles omits password_hash
    - get_training_data omits password_hash
    - Missing users return None / False
    - delete_profile removes a user correctly
    - The real data/profiles.json is NEVER modified

All tests use the tmp_path fixture to redirect file I/O to a temporary
location. The real profiles.json is verified to be unchanged after each test.
"""
import pytest
import json
import os
from profile_store import ProfileStore


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """ProfileStore backed by a fresh temporary JSON file."""
    return ProfileStore(path=str(tmp_path / "test_profiles.json"))


SAMPLE_KP = {
    "shade_index":           0,
    "password_length":       8,
    "avg_hold":              90.0,
    "avg_flight":            45.0,
    "avg_hold_sd":           12.0,
    "avg_flight_sd":         8.0,
    "total_time":            1100.0,
    "samples":               8,
    "enrolled_at":           "2026-01-01T00:00:00",
    "positional_means":      [90.0] * 7,
    "positional_stds":       [20.0] * 7,
    "raw_aggregate_samples": [[90.0] * 13, [88.0] * 13],
}
ARGON2_HASH = "$argon2id$v=19$m=65536,t=3,p=2$FAKESALT$FAKEHASHVALUE"


# ── Save and retrieve ─────────────────────────────────────────────────────────

class TestSaveAndRetrieve:
    def test_save_and_get_profile(self, store):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        profile = store.get_profile("alice")
        assert profile is not None

    def test_password_hash_retrievable_internally(self, store):
        """get_profile is for internal use only — it MUST include the hash for verification."""
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        profile = store.get_profile("alice")
        assert profile["password_hash"] == ARGON2_HASH

    def test_missing_user_returns_none(self, store):
        assert store.get_profile("nonexistent") is None

    def test_list_usernames_returns_all(self, store):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        store.save_profile("bob",   ARGON2_HASH, SAMPLE_KP)
        names = store.list_usernames()
        assert "alice" in names
        assert "bob" in names

    def test_save_overwrites_existing_user(self, store):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        new_kp = {**SAMPLE_KP, "avg_hold": 999.0}
        store.save_profile("alice", ARGON2_HASH, new_kp)
        profile = store.get_profile("alice")
        assert profile["keystroke_profile"]["avg_hold"] == 999.0


# ── Frontend-safe display profiles ───────────────────────────────────────────

class TestDisplayProfiles:
    def test_display_profiles_excludes_password_hash(self, store):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        display = store.get_display_profiles()
        for p in display:
            assert "password_hash" not in p

    def test_display_profiles_does_not_contain_hash_value(self, store):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        display_str = str(store.get_display_profiles())
        assert ARGON2_HASH not in display_str

    def test_display_profiles_includes_username(self, store):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        display = store.get_display_profiles()
        assert any(p["username"] == "alice" for p in display)

    def test_display_profiles_empty_store_returns_empty_list(self, store):
        assert store.get_display_profiles() == []


# ── Training data ─────────────────────────────────────────────────────────────

class TestTrainingData:
    def test_training_data_excludes_password_hash(self, store):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        training = store.get_training_data()
        for entry in training:
            assert "password_hash" not in entry
            assert ARGON2_HASH not in str(entry)

    def test_training_data_includes_username(self, store):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        training = store.get_training_data()
        assert any(t["username"] == "alice" for t in training)

    def test_training_data_includes_aggregate_samples(self, store):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        training = store.get_training_data()
        entry = next(t for t in training if t["username"] == "alice")
        assert "aggregate_samples" in entry


# ── Delete ────────────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_existing_user_returns_true(self, store):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        assert store.delete_profile("alice") is True

    def test_deleted_user_no_longer_retrievable(self, store):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        store.delete_profile("alice")
        assert store.get_profile("alice") is None

    def test_delete_nonexistent_user_returns_false(self, store):
        assert store.delete_profile("nobody") is False


# ── JSON persistence ─────────────────────────────────────────────────────────

class TestJsonPersistence:
    def test_profile_is_json_backed(self, store, tmp_path):
        store.save_profile("alice", ARGON2_HASH, SAMPLE_KP)
        path = tmp_path / "test_profiles.json"
        with open(path) as f:
            data = json.load(f)
        assert "alice" in data

    def test_real_profiles_json_not_touched(self, store):
        """The tmp_path store must NEVER touch the real profiles.json."""
        real_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'profiles.json')
        )
        if not os.path.exists(real_path):
            pytest.skip("Real profiles.json not present — skipping modification check")
        mtime_before = os.path.getmtime(real_path)
        store.save_profile("test_isolation_user", ARGON2_HASH, SAMPLE_KP)
        mtime_after = os.path.getmtime(real_path)
        assert mtime_before == mtime_after, (
            "Real profiles.json was modified during a test — check fixture isolation!"
        )
