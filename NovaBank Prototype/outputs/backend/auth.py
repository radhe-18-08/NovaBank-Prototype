"""
auth.py — Password hashing and verification.

Purpose: Hash passwords securely with Argon2id and verify supplied passwords against stored hashes.
Inputs: Plaintext passwords at enrolment or login, and stored Argon2id password hashes for verification.
Outputs: Argon2id password-hash strings or Boolean password-verification results.


ALGORITHM: Argon2id  (upgraded from PBKDF2-HMAC-SHA256)
─────────────────────────────────────────────────────────────────────────────
Argon2id is the winner of the 2015 Password Hashing Competition and is
the current OWASP recommendation for new systems. It is memory-hard
(unlike PBKDF2), making GPU/ASIC brute-force attacks significantly more
expensive even if the hash file is compromised.

Parameters (OWASP minimum recommendations for interactive logins, 2024):
    time_cost    = 3         — number of iterations
    memory_cost  = 65536     — 64 MB RAM required per hash
    parallelism  = 2         — threads used per hash
    hash_len     = 32        — 256-bit output
    salt_len     = 16        — 128-bit random salt (auto-generated per hash)

Storage format: The argon2-cffi library uses the standard PHC string format:
    $argon2id$v=19$m=65536,t=3,p=2$<salt_b64>$<hash_b64>

This format is self-describing — no separate salt storage is needed.
The hash is NOT reversible. It cannot be decrypted.

SECURITY GUARANTEES:
    - Plaintext password is NEVER stored, logged, or returned.
    - Each password gets a unique random salt — same password hashes differently.
    - Comparison uses argon2-cffi's constant-time verify to prevent timing attacks.
    - The raw password MUST be discarded immediately after hashing/verification.

ACADEMIC PROTOTYPE NOTE:
    This prototype runs on localhost with no TLS. A production banking system
    would additionally require: HTTPS, secure session management, CSRF tokens,
    rate limiting on login endpoints, and account lockout after failed attempts.

DEPENDENCY:
    pip install argon2-cffi>=21.3.0
    (listed in requirements.txt)
"""

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError


# ── Argon2id configuration ──────────────────────────────────────────────────
# These parameters meet OWASP's minimum recommendations for interactive logins.
# Do not reduce time_cost or memory_cost without good reason.
_PH = PasswordHasher(
    time_cost=3,       # 3 iterations
    memory_cost=65536, # 64 MB RAM per hash
    parallelism=2,     # 2 parallel threads
    hash_len=32,       # 256-bit output
    salt_len=16,       # 128-bit salt (auto-generated)
)


def hash_password(plaintext: str) -> str:
    """
    Hash a plaintext password using Argon2id.

    Returns a self-describing PHC format string that includes the salt,
    parameters, and hash — ready to be stored in profiles.json.

    The plaintext password is used only within this function and must be
    discarded by the caller immediately after. It is never returned.
    """
    return _PH.hash(plaintext)


def verify_password(plaintext: str, stored_hash: str) -> bool:
    """
    Verify a plaintext password against a stored Argon2id hash.

    Returns True if the password matches, False otherwise.
    Uses constant-time comparison internally (argon2-cffi guarantee).
    The plaintext password is not retained after this function returns.
    """
    try:
        return _PH.verify(stored_hash, plaintext)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        # Wrong password or malformed hash → deny access
        return False
    except Exception:
        # Any unexpected error → fail closed (deny access)
        return False
