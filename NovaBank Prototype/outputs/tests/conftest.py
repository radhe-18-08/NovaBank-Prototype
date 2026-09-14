"""
conftest.py — pytest configuration for NOVA Bank unit tests.

Purpose: Configure pytest imports and shared test environment for the NOVA Bank test suite without modifying real research data.
Inputs: Pytest session startup and the project directory structure.
Outputs: Test import-path configuration used by the individual unit and integration tests.


Adds the backend/ directory to sys.path so all test modules can import
backend modules (auth, features, profile_deviation, risk_engine, etc.)
directly without installing the project as a package.

IMPORTANT: Tests must never modify the real data files:
    - data/profiles.json
    - data/experiment_log.csv
    - model/keystroke_rf.pkl

All file-based tests use pytest's tmp_path fixture (or monkeypatch) to
redirect I/O to temporary directories that are automatically cleaned up.
"""
import sys
import os

# Resolve paths relative to this file so tests can be run from any directory
_HERE    = os.path.dirname(__file__)
_BACKEND = os.path.abspath(os.path.join(_HERE, '..', 'backend'))

if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
