"""
experiment_logger.py — Log every authentication attempt for evaluation.

Purpose: Record authentication attempts needed to evaluate genuine and impostor performance of the prototype.
Inputs: Authentication metadata, ground-truth experiment label, behavioural scores, credential result and final decision.
Outputs: Rows appended to the experiment CSV plus summary/export information used for offline evaluation.


Records all fields needed to compute:
    - Accuracy, Precision, Recall, F1
    - False Acceptance Rate (FAR)   — impostors incorrectly allowed
    - False Rejection Rate (FRR)    — genuine users incorrectly rejected
    - Confusion matrix

─────────────────────────────────────────────────────────────────────────────
EXPERIMENT PROTOCOL
─────────────────────────────────────────────────────────────────────────────
For the genuine/impostor experiment:
1. Genuine user: logs in as themselves (experiment_label = 'genuine')
2. Impostor:     another person logs in using the same username and password
                 (experiment_label = 'impostor')

The CSV export is used by model/evaluate.py for offline metric computation.

─────────────────────────────────────────────────────────────────────────────
PRIVACY NOTE
─────────────────────────────────────────────────────────────────────────────
Passwords are NEVER logged. The log contains only:
    - username (identifier, needed for grouping by user in evaluation)
    - experiment_label ('genuine' or 'impostor')
    - scores and decisions
"""

import csv
import os
import threading
from datetime import datetime
from typing import Dict, Any, List


_LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'experiment_log.csv')

FIELDNAMES = [
    'event_id',
    'timestamp',
    'username',
    'experiment_label',      # 'genuine' | 'impostor' | 'unknown'
    'credential_result',     # True | False
    'ml_score',              # float or empty
    'profile_score',         # float or empty
    'combined_score',        # float or empty
    'ml_available',          # True | False
    'decision',              # ALLOW | VERIFY | BLOCK
    'risk_level',            # LOW | MEDIUM | HIGH
    # Passwords are NEVER included in this log
]


class ExperimentLogger:
    """Thread-safe CSV logger for authentication attempts."""

    def __init__(self, path: str = _LOG_PATH):
        self.path  = os.path.abspath(path)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Write header if file does not yet exist
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()

    def log(self, entry: Dict[str, Any]) -> None:
        """Append one authentication attempt to the log."""
        row = {}
        for key in FIELDNAMES:
            val = entry.get(key, '')
            # Format floats to 4 decimal places for readability
            if key in ('ml_score', 'profile_score', 'combined_score') and val is not None and val != '':
                try:
                    val = f'{float(val):.4f}'
                except (ValueError, TypeError):
                    val = ''
            row[key] = '' if val is None else val

        with self._lock:
            with open(self.path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(row)

    def load_all(self) -> List[Dict[str, str]]:
        """Load all logged entries as a list of dicts."""
        with self._lock:
            if not os.path.exists(self.path):
                return []
            with open(self.path, 'r', encoding='utf-8') as f:
                return list(csv.DictReader(f))

    def export_csv(self) -> str:
        """Return the raw CSV content as a string."""
        with self._lock:
            if not os.path.exists(self.path):
                return ','.join(FIELDNAMES) + '\n'
            with open(self.path, 'r', encoding='utf-8') as f:
                return f.read()

    def get_summary(self) -> Dict[str, Any]:
        """Return high-level counts for dashboard display."""
        rows = self.load_all()
        total     = len(rows)
        genuine   = [r for r in rows if r['experiment_label'] == 'genuine']
        impostors = [r for r in rows if r['experiment_label'] == 'impostor']
        return {
            'total_attempts': total,
            'genuine':        len(genuine),
            'impostor':       len(impostors),
            'genuine_allowed':  sum(1 for r in genuine   if r['decision'] == 'ALLOW'),
            'genuine_blocked':  sum(1 for r in genuine   if r['decision'] == 'BLOCK'),
            'impostor_allowed': sum(1 for r in impostors if r['decision'] == 'ALLOW'),
            'impostor_blocked': sum(1 for r in impostors if r['decision'] == 'BLOCK'),
        }
