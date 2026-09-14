"""
features.py — Central keystroke feature extractor.

Purpose: Convert raw browser keystroke events into positional and aggregate timing features used by enrolment, profiling and machine learning.
Inputs: Ordered keydown/keyup timing events captured while a password is typed.
Outputs: Hold/flight timing vectors, 13 aggregate features, and enrolled profile statistics.


This is the SINGLE SOURCE OF TRUTH for feature engineering.
Both the enrolment pipeline and the ML training pipeline call these functions.
The JS frontend mirrors extract_positional_features() exactly.

─────────────────────────────────────────────────────────────────────────────
SIGNAL 1 — AGGREGATE FEATURES (password-independent, used by Random Forest)
─────────────────────────────────────────────────────────────────────────────
These 13 features summarise the overall typing rhythm without depending on
which specific keys are pressed. They can be meaningfully compared across
users who type different passwords, making them suitable for a general-purpose
ML model trained on any keystroke dynamics dataset.

Index | Name           | Description
------+----------------+-----------------------------------------------------
  0   | mean_hold      | Mean key dwell/hold time (ms)
  1   | std_hold       | Standard deviation of hold times (ms)
  2   | mean_flight    | Mean inter-key flight time (ms)
  3   | std_flight     | Standard deviation of flight times (ms)
  4   | total_time     | Total typing duration: last_keyup - first_keydown (ms)
  5   | n_keys         | Number of keystrokes captured
  6   | typing_speed   | Typing speed in characters per second
  7   | cv_hold        | Coefficient of variation for hold times (capped at 10)
  8   | cv_flight      | Coefficient of variation for flight times (capped at 10)
  9   | min_hold       | Minimum hold time (ms)
 10   | max_hold       | Maximum hold time (ms)
 11   | min_flight     | Minimum flight time (ms); 0 for single-key passwords
 12   | max_flight     | Maximum flight time (ms); 0 for single-key passwords

─────────────────────────────────────────────────────────────────────────────
SIGNAL 2 — POSITIONAL FEATURES (password-specific, used by Profile Deviation)
─────────────────────────────────────────────────────────────────────────────
These per-key-position features are only meaningful when compared against the
SAME USER typing the SAME PASSWORD. They form the enrolled profile baseline.

Order: [hold_0, hold_1, ..., hold_{n-1}, flight_0, flight_1, ..., flight_{n-2}]
Length: 2n-1 where n = number of keystrokes

This EXACTLY mirrors the JavaScript extractFeatures() function in
enrolment_studio.html and nova_bank_portal.html.

─────────────────────────────────────────────────────────────────────────────
RAW KEYDATA FORMAT
─────────────────────────────────────────────────────────────────────────────
Both functions accept raw_keydata: list of dicts with keys:
    downTime (float): timestamp when key was pressed (ms, from performance.now())
    upTime   (float): timestamp when key was released (ms)

Times are relative to page load — only the DIFFERENCES matter, not absolute values.
"""

import math
from typing import List, Dict, Any


# ── Type alias ─────────────────────────────────────────────────────────────
KeyEvent = Dict[str, float]   # {"downTime": float, "upTime": float}


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _raw_timings(kd: List[KeyEvent]):
    """
    Compute hold times and flight times from raw keystroke events.

    hold[i]   = kd[i].upTime   - kd[i].downTime        (how long key i was held)
    flight[i] = kd[i+1].downTime - kd[i].upTime         (gap between releasing i and pressing i+1)
                clamped to ≥ 0  (negative values arise when keys overlap; treat as 0)

    Returns: (holds: list[float], flights: list[float])
    """
    if not kd:
        return [], []
    holds   = [k['upTime'] - k['downTime'] for k in kd]
    flights = [max(0.0, kd[i + 1]['downTime'] - kd[i]['upTime'])
               for i in range(len(kd) - 1)]
    return holds, flights


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals, mean=None):
    if len(vals) < 2:
        return 0.0
    m = mean if mean is not None else _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_positional_features(kd: List[KeyEvent]) -> List[float]:
    """
    Extract per-position features from raw keystroke data.

    Mirrors the JS function exactly:
        function extractFeatures(kd) {
          const holds   = kd.map(k => k.upTime - k.downTime);
          const flights = [];
          for (let i = 0; i < kd.length - 1; i++)
            flights.push(Math.max(0, kd[i+1].downTime - kd[i].upTime));
          return [...holds, ...flights];
        }

    Returns a list of length 2n-1 (n = number of keystrokes).
    """
    holds, flights = _raw_timings(kd)
    return holds + flights


def extract_aggregate_features(kd: List[KeyEvent]) -> List[float]:
    """
    Extract 13 password-independent aggregate features.
    See module docstring for the complete feature table.

    Returns a list of exactly 13 floats.
    Returns all zeros if kd is empty.
    """
    if not kd:
        return [0.0] * 13

    holds, flights = _raw_timings(kd)
    n = len(holds)

    # ── Hold stats ──────────────────────────────────────────────
    mean_hold = _mean(holds)
    std_hold  = _std(holds, mean_hold)

    # ── Flight stats ────────────────────────────────────────────
    if flights:
        mean_flight = _mean(flights)
        std_flight  = _std(flights, mean_flight)
        min_flight  = min(flights)
        max_flight  = max(flights)
    else:
        mean_flight = std_flight = min_flight = max_flight = 0.0

    # ── Total typing time ────────────────────────────────────────
    total_time = max(k['upTime'] for k in kd) - min(k['downTime'] for k in kd)

    # ── Typing speed (chars per second) ─────────────────────────
    typing_speed = (n / (total_time / 1000.0)) if total_time > 0 else 0.0

    # ── Coefficient of variation (capped to prevent extreme outliers) ──
    CV_CAP = 10.0
    cv_hold   = min(std_hold   / mean_hold   if mean_hold   > 0 else 0.0, CV_CAP)
    cv_flight = min(std_flight / mean_flight if mean_flight > 0 else 0.0, CV_CAP)

    return [
        mean_hold,          # 0
        std_hold,           # 1
        mean_flight,        # 2
        std_flight,         # 3
        total_time,         # 4
        float(n),           # 5
        typing_speed,       # 6
        cv_hold,            # 7
        cv_flight,          # 8
        min(holds),         # 9
        max(holds),         # 10
        min_flight,         # 11
        max_flight,         # 12
    ]


# ══════════════════════════════════════════════════════════════════════════════
# PROFILE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

STD_FLOOR_MS = 20.0  # Minimum std (ms) to prevent division by zero in z-scores.
# 20ms reflects realistic human timing resolution: studies show ~20ms is roughly
# the threshold at which timing differences become perceptible. Values below this
# produce unrealistically tight z-scores from noisy 3-sample enrolment data.


def build_profile_stats(samples: List[List[float]]) -> Dict[str, List[float]]:
    """
    Compute per-feature mean and standard deviation from multiple enrolment samples.

    Args:
        samples: list of feature vectors (all must have the same length)

    Returns:
        dict with keys:
            'means': list[float]
            'stds':  list[float]  (floored at STD_FLOOR_MS to prevent zero-division)
    """
    if not samples:
        return {'means': [], 'stds': []}

    n_features = len(samples[0])
    means, stds = [], []

    for f in range(n_features):
        vals = [s[f] for s in samples if f < len(s)]
        if not vals:
            means.append(0.0)
            stds.append(STD_FLOOR_MS)
            continue
        m   = _mean(vals)
        sd  = _std(vals, m)
        means.append(m)
        stds.append(max(sd, STD_FLOOR_MS))

    return {'means': means, 'stds': stds}
