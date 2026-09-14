"""
risk_engine.py — Combines credential result, ML score, and profile score into a decision.

Purpose: Combine credential verification, Random Forest score and personalised profile score into the prototype authentication decision.
Inputs: Credential result, ML score, profile score and behavioural-data sufficiency flags.
Outputs: Combined behavioural score, risk level and ALLOW/VERIFY/BLOCK decision with supporting decision metadata.


─────────────────────────────────────────────────────────────────────────────
DECISION LOGIC
─────────────────────────────────────────────────────────────────────────────

Step 1 — Credential gate:
    Incorrect credentials → BLOCK (always, regardless of keystroke data)

Step 2 — Insufficient behavioural data:
    If the profile score was flagged as insufficient_data=True (too few features)
    AND ML is also unavailable → VERIFY (not BLOCK).
    There is not enough evidence to strongly reject; require step-up verification.

Step 3 — Combine behavioural signals:
    Both ML and profile available:
        combined = ML_WEIGHT * ml_score + PROFILE_WEIGHT * profile_score
    Only profile available (ML not trained yet):
        combined = profile_score
    Only ML available:
        combined = ml_score
    Neither available:
        combined = None → VERIFY (insufficient evidence to BLOCK)

Step 4 — Risk decision:
    combined >= ALLOW_THRESHOLD   → ALLOW (LOW risk)
    combined >= VERIFY_THRESHOLD  → VERIFY (MEDIUM risk)
    combined <  VERIFY_THRESHOLD  → BLOCK (HIGH risk)

─────────────────────────────────────────────────────────────────────────────
OPERATIONAL THRESHOLDS (submitted prototype values)
─────────────────────────────────────────────────────────────────────────────

ALLOW_THRESHOLD = 0.55
    With DIVISOR=8 and STD_FLOOR=20ms, a score of 0.55 corresponds to
    avg_z ≈ 3.6, meaning the typist is on average ~72ms per keystroke from
    the enrolled profile. This is the operational ALLOW boundary used during
    the live experiment sessions reported in the dissertation.

VERIFY_THRESHOLD = 0.28
    Score of 0.28 → avg_z ≈ 5.8 → ~116ms average deviation per keystroke.
    Below this represents a strongly inconsistent typing pattern. The VERIFY
    zone (0.28–0.55) routes borderline attempts to OTP rather than blocking.

NOTE: These are the operational thresholds used in the submitted prototype
    and reported in the dissertation results. The offline binary evaluation
    script (model/evaluate.py) uses a separate default threshold of 0.65,
    which is the binary accept/reject boundary used for FAR/FRR reporting
    and is intentionally distinct from the three-level operational thresholds.
    The raw combined_score is always preserved in experiment_log.csv so
    decisions can be re-evaluated at any threshold post-hoc.

─────────────────────────────────────────────────────────────────────────────
COMBINATION WEIGHTS
─────────────────────────────────────────────────────────────────────────────
The equal 50/50 weighting was selected for the submitted prototype.
Both signals (ML behavioural score and per-user profile deviation score)
are on a 0–1 scale and contribute equally to the combined_score.
"""

from typing import Optional, Dict, Any


# ══════════════════════════════════════════════════════════════════════════════
# !! SINGLE CONFIGURATION AREA — ALL THRESHOLDS LIVE HERE ONLY !!
# Do NOT hard-code thresholds in any other file.
# ══════════════════════════════════════════════════════════════════════════════
#
# These are the operational values used in the submitted prototype.
# They are the fixed operational values used in the submitted prototype.
# They were not retrospectively tuned to maximise the final reported metrics.
# The final experiment is used to evaluate their behaviour, not to claim that
# they are universally optimal.
#
#   ALLOW_THRESHOLD = 0.55
#     Corresponds to avg_z ≈ 3.6, meaning ~72ms average per-keystroke deviation
#     from the enrolled profile. Defines the prototype ALLOW boundary for a strong behavioural match.
#
#   VERIFY_THRESHOLD = 0.28
#     Corresponds to avg_z ≈ 5.8 (~116ms average deviation).
#     Defines the prototype BLOCK boundary. Scores in the VERIFY zone
#     (0.28–0.55) are routed to step-up verification rather than hard-blocking.
#
# The offline binary evaluation script (model/evaluate.py) uses a separate
# default threshold of 0.65 for binary FAR/FRR reporting; that value is
# intentionally distinct from these three-level operational thresholds.
# See the dissertation Methods section for the rationale.

ALLOW_THRESHOLD  = 0.55   # operational threshold — see dissertation Methods
VERIFY_THRESHOLD = 0.28   # operational threshold — see dissertation Methods

# ── Signal combination weights ────────────────────────────────────────────────
# Equal 50/50 weighting used in the submitted prototype.
# Both signals are normalised to [0, 1] before combination.
# Weights must sum to 1.0.
ML_WEIGHT      = 0.50     # Signal 1: Random Forest (behavioural population model)
PROFILE_WEIGHT = 0.50     # Signal 2: User-specific profile deviation


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the risk engine.

    Args:
        inputs: dict with keys:
            credential_result    : bool         — True = password verified
            ml_score             : float | None — 0..1, None if model not trained
            profile_score        : float | None — 0..1, None if no profile
            profile_insufficient : bool         — True if too few features for Signal 2

    Returns:
        dict with keys:
            decision        : 'ALLOW' | 'VERIFY' | 'BLOCK'
            risk_level      : 'LOW'   | 'MEDIUM' | 'HIGH'
            combined_score  : float   | None
            ml_available    : bool
            combination_method : str
            reason          : str
    """
    credential_result    = inputs.get('credential_result', False)
    ml_score             = inputs.get('ml_score')
    profile_score        = inputs.get('profile_score')
    profile_insufficient = inputs.get('profile_insufficient', False)

    ml_available = ml_score is not None

    # ── Step 1: Credential gate ──────────────────────────────────────────────
    if not credential_result:
        return _result(
            decision='BLOCK',
            risk_level='HIGH',
            combined_score=None,
            ml_available=ml_available,
            combination_method='N/A — credential check failed',
            reason='Incorrect credentials. Access denied.',
        )

    # ── Step 2: Insufficient behavioural data → VERIFY, not BLOCK ────────────
    if profile_insufficient and not ml_available:
        return _result(
            decision='VERIFY',
            risk_level='MEDIUM',
            combined_score=None,
            ml_available=False,
            combination_method='Insufficient behavioural data — too few keystrokes captured',
            reason='Insufficient keystroke data to complete behavioural verification. '
                   'Step-up verification required.',
        )

    # ── Step 3: Combine signals ──────────────────────────────────────────────
    if ml_available and profile_score is not None:
        combined = ML_WEIGHT * ml_score + PROFILE_WEIGHT * profile_score
        method   = (f'Weighted average: {ML_WEIGHT}×ML({ml_score:.3f}) '
                    f'+ {PROFILE_WEIGHT}×Profile({profile_score:.3f})')
    elif profile_score is not None:
        combined = profile_score
        method   = f'Profile deviation only (ML model not yet trained): {profile_score:.3f}'
    elif ml_available:
        combined = ml_score
        method   = f'ML score only (no personal profile): {ml_score:.3f}'
    else:
        # No behavioural signal at all — step up rather than hard-block
        return _result(
            decision='VERIFY',
            risk_level='MEDIUM',
            combined_score=None,
            ml_available=False,
            combination_method='No behavioural signal available',
            reason='No behavioural data was available to evaluate this login. '
                   'Step-up verification required.',
        )

    combined = round(combined, 4)

    # ── Step 4: Threshold decision ───────────────────────────────────────────
    if combined >= ALLOW_THRESHOLD:
        decision   = 'ALLOW'
        risk_level = 'LOW'
        reason     = (f'Credentials correct. Behavioural match strong '
                      f'({combined:.0%}). {method}.')
    elif combined >= VERIFY_THRESHOLD:
        decision   = 'VERIFY'
        risk_level = 'MEDIUM'
        reason     = (f'Credentials correct but behavioural confidence '
                      f'borderline ({combined:.0%}). Step-up verification required. {method}.')
    else:
        decision   = 'BLOCK'
        risk_level = 'HIGH'
        reason     = (f'Credentials correct, but behavioural profile strongly '
                      f'mismatched ({combined:.0%}). Possible account takeover. {method}.')

    return _result(
        decision=decision,
        risk_level=risk_level,
        combined_score=combined,
        ml_available=ml_available,
        combination_method=method,
        reason=reason,
    )


def _result(**kwargs) -> Dict[str, Any]:
    return kwargs
