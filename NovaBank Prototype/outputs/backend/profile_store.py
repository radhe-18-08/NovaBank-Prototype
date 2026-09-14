"""
profile_store.py — Thread-safe JSON-backed profile storage.

Purpose: Provide thread-safe JSON storage and retrieval for enrolled NOVA Bank user profiles.
Inputs: Usernames, Argon2id password hashes, keystroke profiles and profile-management requests.
Outputs: Saved/retrieved/deleted profile records and privacy-safe profile/training views that omit password hashes where appropriate.


Profiles are stored in data/profiles.json.

Each entry contains:
    - password_hash    : Argon2id hash (never the plaintext password).
                         Format: PHC string ($argon2id$v=19$...) produced by
                         backend/auth.py:hash_password().
    - keystroke_profile: behavioural statistics and display data.

The password_hash is NEVER included in data returned to the frontend.
All public methods that return profile data omit password_hash.
get_profile() (internal only) returns the full entry including the hash,
which is needed for password verification at login.
"""

import json
import os
import threading
from typing import Optional, Dict, List, Any


# Default storage location (relative to this file → ../data/profiles.json)
_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'profiles.json')


class ProfileStore:
    """Thread-safe profile store backed by a JSON file."""

    def __init__(self, path: str = _DEFAULT_PATH):
        self.path = os.path.abspath(path)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._write({})

    # ── Private I/O ──────────────────────────────────────────────────────────

    def _read(self) -> Dict:
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write(self, data: Dict) -> None:
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── Write operations ─────────────────────────────────────────────────────

    def save_profile(self, username: str, password_hash: str,
                     keystroke_profile: Dict[str, Any]) -> None:
        """
        Save or overwrite a user profile.
        password_hash must already be hashed — this function does NOT hash passwords.
        """
        with self._lock:
            data = self._read()
            data[username] = {
                'password_hash':     password_hash,
                'keystroke_profile': keystroke_profile,
            }
            self._write(data)

    def delete_profile(self, username: str) -> bool:
        with self._lock:
            data = self._read()
            if username in data:
                del data[username]
                self._write(data)
                return True
            return False

    # ── Read operations (include password_hash — internal use only) ──────────

    def get_profile(self, username: str) -> Optional[Dict]:
        """
        Return the FULL profile entry for internal use (password verification).
        Contains password_hash — must NEVER be forwarded to the frontend.
        Returns None if not found.
        """
        with self._lock:
            return self._read().get(username)

    # ── Read operations (safe for frontend) ──────────────────────────────────

    def list_usernames(self) -> List[str]:
        """Return list of enrolled usernames. No passwords or hashes included."""
        with self._lock:
            return list(self._read().keys())

    def get_display_profiles(self) -> List[Dict]:
        """
        Return profile data safe for the frontend — password_hash is EXCLUDED.
        Includes everything needed for chart rendering, profile cards, and sidebar.
        """
        with self._lock:
            data = self._read()
        result = []
        for username, entry in data.items():
            kp = entry.get('keystroke_profile', {})
            result.append({
                'username':          username,
                'shade_index':       kp.get('shade_index', 0),
                'password_length':   kp.get('password_length', 0),
                'avg_hold':          kp.get('avg_hold', 0),
                'avg_flight':        kp.get('avg_flight', 0),
                'avg_hold_sd':       kp.get('avg_hold_sd', 0),
                'avg_flight_sd':     kp.get('avg_flight_sd', 0),
                'total_time':        kp.get('total_time', 0),
                'samples':           kp.get('samples', 0),
                'enrolled_at':       kp.get('enrolled_at', ''),
                'positional_means':  kp.get('positional_means', []),
                'positional_stds':   kp.get('positional_stds', []),
                # Password deliberately omitted
            })
        return result

    def get_training_data(self) -> List[Dict]:
        """
        Return aggregate feature samples for ML model training.
        Includes: username, aggregate_samples (list of feature vectors per sample).
        Password_hash deliberately excluded.
        """
        with self._lock:
            data = self._read()
        result = []
        for username, entry in data.items():
            kp = entry.get('keystroke_profile', {})
            result.append({
                'username':          username,
                'password_length':   kp.get('password_length', 0),
                'aggregate_samples': kp.get('raw_aggregate_samples', []),
            })
        return result
