"""
profile_deviation.py — Signal 2: User-specific profile deviation scorer.

Purpose: Compare a current typing sample with an enrolled user-specific timing profile to produce an independent behavioural similarity score.
Inputs: Current positional keystroke features and the claimed user's enrolled profile means and standard deviations.
Outputs: Profile similarity/deviation score with evidence-quality information used by the risk engine.


This is an INDEPENDENT signal from the Random Forest ML model.

The enrolled user's typing samples produce a personal behavioural baseline
(per-position mean and standard deviation). At login, we compare the current
typing against THIS USER'S OWN baseline.

─────────────────────────────────────────────────────────────────────────────
HOW THE Z-SCORE SIMILARITY WORKS
─────────────────────────────────────────────────────────────────────────────
For each feature position i:

    z_i = |current_feature[i] - profile_mean[i]| / max(profile_std[i], STD_FLOOR_MS)

The TRIMMED MEAN z-score (worst TRIM_FRACTION of keystrokes excluded) is then
mapped to a [0, 1] similarity score:

    profile_score = max(0, min(1, 1 - trimmed_avg_z / DIVISOR))

─────────────────────────────────────────────────────────────────────────────
CONFIGURABLE HYPERPARAMETERS — MUST BE JUSTIFIED EXPERIMENTALLY
─────────────────────────────────────────────────────────────────────────────
DIVISOR = 8.0
    avg_z at which similarity → 0. Increased from the original 4.0 because:
    - With STD_FLOOR_MS=20ms, z=8 means ~160ms deviation per keystroke on average,
      which is a large, consistent difference (likely an impostor).
    - z=2 → score=0.75  (slight natural variation → mostly accepted)
    - z=4 → score=0.50  (noticeable difference → borderline)
    - z=8 → score=0.0   (very different typist → blocked)
    DISSERTATION: Validate by plotting genuine vs impostor score distributions.
    Choose the value that maximises separation between the two populations.

STD_FLOOR_MS = 20.0 (from features.py)
    Minimum standard deviation to prevent division by zero and to reflect
    realistic hardware+neuromuscular timing resolution limits (~20ms).
    Increased from 3ms because 3ms produced z-scores that were over-sensitive
    to normal typing jitter.

TRIM_FRACTION = 0.20
    Top 20% of z-scores (worst-matching keystrokes) are excluded before
    computing the mean. This protects against one distracted/mis-timed
    keystroke torpedoing an otherwise good login attempt.
    With e.g. 8 features: 1 outlier is excluded.
    DISSERTATION: Compare FAR/FRR with TRIM_FRACTION=0 vs 0.20 vs 0.30.

MIN_VALID_FEATURES = 3
    If fewer than this many features can be compared, the result is flagged
    as 'insufficient_data'. The risk engine then returns VERIFY rather than
    BLOCK — there is not enough evidence to make a strong decision either way.

─────────────────────────────────────────────────────────────────────────────
RETURN VALUE
─────────────────────────────────────────────────────────────────────────────
Returns a dict (not a bare float) so the risk engine can see metadata:
    {
        'score':              float in [0.0, 1.0],  # higher = more similar
        'avg_z':              float,                # trimmed mean z-score
        'valid_features':     int,                  # features actually compared
        'trimmed_features':   int,                  # outlier features excluded
        'insufficient_data':  bool,                 # True → too few features
    }

Direction: HIGHER score = MORE similar = LOWER risk of impostor.
"""

from features import STD_FLOOR_MS
from typing import List, Dict, Any


# ── Configurable parameters ──────────────────────────────────────────────────
DIVISOR        = 8.0    # avg_z at which similarity → 0; see module docstring
TRIM_FRACTION  = 0.20   # fraction of worst z-scores to exclude (outlier protection)
MIN_VALID_FEATURES = 3  # below this, flag as insufficient data


def calculate_profile_deviation(
    current_features: List[float],
    profile_means:    List[float],
    profile_stds:     List[float],
) -> Dict[str, Any]:
    """
    Compute how similar the current typing is to the enrolled profile.

    Args:
        current_features : positional feature vector from the current login attempt
        profile_means    : per-position mean from enrolment samples
        profile_stds     : per-position std from enrolment samples (floored at STD_FLOOR_MS)

    Returns:
        dict with keys:
            score              : float in [0.0, 1.0]   — 1.0=perfect match, 0.0=no match
            avg_z              : float                  — trimmed mean z-score
            valid_features     : int                    — number of features compared
            trimmed_features   : int                    — number of outlier features excluded
            insufficient_data  : bool                   — True if too few features to decide
    """
    n = min(len(current_features), len(profile_means), len(profile_stds))

    # ── No overlap at all ────────────────────────────────────────────────────
    if n == 0:
        return _result(score=0.5, avg_z=0.0, valid=0, trimmed=0, insufficient=True)

    # ── Compute per-feature z-scores ─────────────────────────────────────────
    z_scores = []
    for i in range(n):
        std = max(profile_stds[i], STD_FLOOR_MS)   # defensive floor (also set at build time)
        z   = abs(current_features[i] - profile_means[i]) / std
        z_scores.append(z)

    valid_count = len(z_scores)

    # ── Trim worst TRIM_FRACTION of z-scores (outlier keystroke protection) ──
    trim_count  = max(0, int(valid_count * TRIM_FRACTION))
    sorted_z    = sorted(z_scores)                          # ascending
    trimmed_z   = sorted_z[:valid_count - trim_count]       # drop the largest

    if not trimmed_z:
        # Edge case: trim removed all features (shouldn't happen with <5 features)
        trimmed_z = sorted_z

    avg_z = sum(trimmed_z) / len(trimmed_z)
    score = max(0.0, min(1.0, 1.0 - avg_z / DIVISOR))

    insufficient = valid_count < MIN_VALID_FEATURES

    return _result(
        score=round(score, 4),
        avg_z=round(avg_z, 4),
        valid=valid_count,
        trimmed=trim_count,
        insufficient=insufficient,
    )


def _result(score, avg_z, valid, trimmed, insufficient) -> Dict[str, Any]:
    return {
        'score':             score,
        'avg_z':             avg_z,
        'valid_features':    valid,
        'trimmed_features':  trimmed,
        'insufficient_data': insufficient,
    }
