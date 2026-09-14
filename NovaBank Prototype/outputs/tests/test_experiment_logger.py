"""
test_experiment_logger.py — Unit tests for backend/experiment_logger.py.

Purpose: Test experiment CSV creation, logging, privacy and summary calculations using temporary files.
Inputs: Synthetic authentication/experiment records and pytest temporary paths.
Outputs: Pytest pass/fail assertions for CSV structure, stored fields, repeated writes and summary counts.


Tests verify:
    - Header is written on first use
    - A valid row is written and readable
    - experiment_label is stored as evaluation metadata only (not used in decision)
    - Passwords and plaintext secrets are never written (not in FIELDNAMES schema)
    - Repeated writes preserve CSV structure
    - get_summary returns correct counts

IMPORTANT: All tests use tmp_path — the real data/experiment_log.csv
is NEVER modified.
"""
import pytest
import csv
import os
from experiment_logger import ExperimentLogger, FIELDNAMES


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def logger(tmp_path):
    """ExperimentLogger backed by a temporary file (real log never touched)."""
    return ExperimentLogger(path=str(tmp_path / "test_log.csv"))


def _make_entry(**overrides):
    base = {
        "event_id":          "EVT-TEST001",
        "timestamp":         "2026-01-01T00:00:00",
        "username":          "alice",
        "experiment_label":  "genuine",
        "credential_result": True,
        "ml_score":          0.75,
        "profile_score":     0.80,
        "combined_score":    0.775,
        "ml_available":      True,
        "decision":          "ALLOW",
        "risk_level":        "LOW",
    }
    base.update(overrides)
    return base


# ── Header ────────────────────────────────────────────────────────────────────

class TestHeader:
    def test_header_written_on_creation(self, tmp_path):
        path = str(tmp_path / "log.csv")
        ExperimentLogger(path=path)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == FIELDNAMES

    def test_header_not_duplicated_on_second_instantiation(self, tmp_path):
        path = str(tmp_path / "log.csv")
        ExperimentLogger(path=path)
        ExperimentLogger(path=path)
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        # Only one header line
        assert sum(1 for l in lines if "event_id" in l) == 1


# ── Writing and reading ───────────────────────────────────────────────────────

class TestWriting:
    def test_valid_row_is_written(self, logger):
        logger.log(_make_entry())
        rows = logger.load_all()
        assert len(rows) == 1

    def test_username_is_correct(self, logger):
        logger.log(_make_entry(username="alice"))
        rows = logger.load_all()
        assert rows[0]["username"] == "alice"

    def test_decision_is_correct(self, logger):
        logger.log(_make_entry(decision="BLOCK"))
        rows = logger.load_all()
        assert rows[0]["decision"] == "BLOCK"

    def test_repeated_writes_accumulate(self, logger):
        for i in range(5):
            logger.log(_make_entry(event_id=f"EVT-{i:04d}"))
        rows = logger.load_all()
        assert len(rows) == 5

    def test_repeated_writes_preserve_schema(self, logger):
        for i in range(3):
            logger.log(_make_entry(event_id=f"EVT-{i:04d}"))
        rows = logger.load_all()
        for row in rows:
            assert set(row.keys()) == set(FIELDNAMES)


# ── experiment_label as metadata only ────────────────────────────────────────

class TestExperimentLabel:
    def test_genuine_label_is_stored(self, logger):
        logger.log(_make_entry(experiment_label="genuine"))
        rows = logger.load_all()
        assert rows[0]["experiment_label"] == "genuine"

    def test_impostor_label_is_stored(self, logger):
        logger.log(_make_entry(experiment_label="impostor", decision="BLOCK"))
        rows = logger.load_all()
        assert rows[0]["experiment_label"] == "impostor"

    def test_label_and_decision_are_independent_fields(self, logger):
        """A genuine label does not force ALLOW; an impostor label does not force BLOCK."""
        # Genuine label but BLOCK decision (e.g., behavioural mismatch)
        logger.log(_make_entry(experiment_label="genuine", decision="BLOCK"))
        rows = logger.load_all()
        assert rows[0]["experiment_label"] == "genuine"
        assert rows[0]["decision"] == "BLOCK"


# ── Password security ─────────────────────────────────────────────────────────

class TestPasswordSecurity:
    def test_password_not_in_fieldnames(self):
        """The CSV schema must never include a password-related column."""
        forbidden = ("password", "plaintext", "secret", "passwd")
        for field in FIELDNAMES:
            for bad in forbidden:
                assert bad not in field.lower(), (
                    f"Forbidden term '{bad}' found in FIELDNAMES column '{field}'"
                )

    def test_extra_password_key_in_entry_is_not_written(self, logger):
        """If a caller accidentally passes a 'password' key, it must be silently dropped."""
        logger.log({**_make_entry(), "password": "SECRET_MUST_NOT_APPEAR"})
        rows = logger.load_all()
        assert "password" not in rows[0]
        for val in rows[0].values():
            assert "SECRET_MUST_NOT_APPEAR" not in str(val)


# ── Summary ───────────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_counts_genuine(self, logger):
        logger.log(_make_entry(experiment_label="genuine", decision="ALLOW"))
        summary = logger.get_summary()
        assert summary["genuine"] == 1

    def test_summary_counts_impostor(self, logger):
        logger.log(_make_entry(experiment_label="impostor", decision="BLOCK"))
        summary = logger.get_summary()
        assert summary["impostor"] == 1

    def test_summary_genuine_allowed(self, logger):
        logger.log(_make_entry(experiment_label="genuine", decision="ALLOW"))
        summary = logger.get_summary()
        assert summary["genuine_allowed"] == 1

    def test_summary_impostor_blocked(self, logger):
        logger.log(_make_entry(experiment_label="impostor", decision="BLOCK"))
        summary = logger.get_summary()
        assert summary["impostor_blocked"] == 1

    def test_summary_total_attempts(self, logger):
        logger.log(_make_entry())
        logger.log(_make_entry(event_id="EVT-002", experiment_label="impostor"))
        summary = logger.get_summary()
        assert summary["total_attempts"] == 2


# ── Real log is never modified ────────────────────────────────────────────────

class TestRealLogProtection:
    def test_tmp_path_fixture_uses_separate_file(self, logger, tmp_path):
        """The logger fixture writes to a temp file, not the real experiment_log.csv."""
        real_log = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'experiment_log.csv')
        )
        assert os.path.abspath(logger.path) != real_log
