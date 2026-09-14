"""
test_auth.py — Unit tests for backend/auth.py (Argon2id password hashing).

Purpose: Test Argon2id password hashing and verification behaviour.
Inputs: Representative correct, incorrect and malformed password/hash values.
Outputs: Pytest pass/fail assertions covering secure hashing, random salting and safe verification failure.


Tests verify:
    - Hashing produces a valid Argon2id PHC string (not plaintext)
    - Correct password verifies successfully
    - Incorrect password is rejected
    - Malformed/corrupt hashes fail safely (no exception)
    - Identical passwords produce different hashes (random salt)
    - Hashed output never contains the plaintext password
"""
import pytest
from auth import hash_password, verify_password


class TestHashPassword:
    def test_returns_argon2id_phc_string(self):
        """Output must start with the Argon2id PHC identifier."""
        h = hash_password("TestPassword1!")
        assert h.startswith("$argon2id$"), (
            f"Expected Argon2id PHC format, got: {h[:30]}"
        )

    def test_hash_does_not_contain_plaintext(self):
        """The plaintext password must never appear in the hash string."""
        password = "SuperSecret99"
        h = hash_password(password)
        assert password not in h

    def test_same_password_produces_different_hashes(self):
        """Each call uses a fresh random salt — identical passwords hash differently."""
        h1 = hash_password("password123")
        h2 = hash_password("password123")
        assert h1 != h2, "Same password hashed twice should not produce identical output"

    def test_hash_is_string(self):
        h = hash_password("any_password")
        assert isinstance(h, str)


class TestVerifyPassword:
    def test_correct_password_verifies(self):
        h = hash_password("correct_password")
        assert verify_password("correct_password", h) is True

    def test_wrong_password_fails(self):
        h = hash_password("correct_password")
        assert verify_password("wrong_password", h) is False

    def test_empty_password_verifies_against_its_own_hash(self):
        h = hash_password("")
        assert verify_password("", h) is True

    def test_empty_password_wrong_on_nonempty_hash(self):
        h = hash_password("nonempty")
        assert verify_password("", h) is False

    def test_malformed_hash_fails_safely(self):
        """A corrupt or non-Argon2 hash string must return False, not raise."""
        assert verify_password("anything", "not_a_valid_hash") is False

    def test_empty_hash_fails_safely(self):
        assert verify_password("anything", "") is False

    def test_partial_hash_fails_safely(self):
        assert verify_password("pw", "$argon2id$v=19$m=65536,t=3,p=2$TRUNCATED") is False

    def test_case_sensitive(self):
        h = hash_password("Password")
        assert verify_password("password", h) is False  # lowercase p

    def test_returns_bool(self):
        h = hash_password("test")
        result = verify_password("test", h)
        assert isinstance(result, bool)
