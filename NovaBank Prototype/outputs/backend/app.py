"""
app.py — NOVA Bank Keystroke Dynamics — Flask REST API

Purpose: Run the Flask REST API that connects enrolment, credential checking, behavioural scoring, risk decisions, profile management, and experiment logging.
Inputs: HTTP requests containing enrolment or login data, including username, password and captured keystroke timings.
Outputs: JSON API responses such as health information, profile summaries, enrolment results, behavioural scores and ALLOW/VERIFY/BLOCK decisions.


Endpoints:
    GET  /api/health              Backend health + model status
    GET  /api/profiles            All enrolled profiles (safe for frontend — no hashes)
    POST /api/enrol               Register a new participant
    POST /api/auth                Authenticate and score a login attempt
    DELETE /api/profiles/<name>   Remove a participant
    GET  /api/experiment/summary  High-level experiment counts
    GET  /api/experiment/export   Download experiment_log.csv

Run:  python app.py
Port: 5001

SECURITY NOTE:
    Passwords received in POST /api/enrol and POST /api/auth are:
    - Hashed immediately (enrol) or compared against the hash (auth)
    - NEVER stored, logged, or returned in any response
    - Discarded as soon as hashing/verification is complete
"""

import sys
import os
import uuid
from datetime import datetime

# Make backend/ importable when running from this file's directory
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from auth             import hash_password, verify_password
from profile_store    import ProfileStore
from features         import extract_positional_features, extract_aggregate_features, build_profile_stats
from profile_deviation import calculate_profile_deviation
from ml_inference     import MLInference
from risk_engine      import evaluate as risk_evaluate
from experiment_logger import ExperimentLogger


# ── App setup ────────────────────────────────────────────────────────────────
app   = Flask(__name__)
CORS(app, origins='*')   # Allow localhost and file:// during development

# ── Singletons ───────────────────────────────────────────────────────────────
store   = ProfileStore()
ml      = MLInference()
explog  = ExperimentLogger()


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':        'ok',
        'model_loaded':  ml.is_loaded(),
        'enrolled_count': len(store.list_usernames()),
    })


# ══════════════════════════════════════════════════════════════════════════════
# PROFILES (read — no passwords returned)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/profiles', methods=['GET'])
def get_profiles():
    """
    Return all enrolled profiles for the frontend dashboard.
    Passwords and hashes are NEVER included.
    """
    return jsonify({'profiles': store.get_display_profiles()})


# ══════════════════════════════════════════════════════════════════════════════
# ENROLMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/enrol', methods=['POST'])
def enrol():
    """
    Register a new participant.

    Expected JSON body:
    {
        "username":       string,
        "password":       string,    ← plaintext, hashed immediately, never stored
        "password_length": int,
        "shade_index":    int,
        "raw_samples":    [          ← list of 8 raw keystroke arrays (one per valid enrolment sample)
            [{"downTime": ms, "upTime": ms}, ...],
            [...],
            ...                      ← 8 arrays total
        ]
    }
    """
    data = request.get_json(force=True)

    username        = (data.get('username') or '').strip()
    password        = data.get('password') or ''          # plaintext — hash immediately
    password_length = int(data.get('password_length', 0))
    shade_index     = int(data.get('shade_index', 0))
    raw_samples     = data.get('raw_samples', [])

    # ── Validation ────────────────────────────────────────────────────────────
    if not username:
        return jsonify({'success': False, 'error': 'Username is required'}), 400
    if not password:
        return jsonify({'success': False, 'error': 'Password is required'}), 400
    if len(raw_samples) < 8:
        return jsonify({'success': False, 'error': f'8 valid keystroke samples required; received {len(raw_samples)}'}), 400
    if store.get_profile(username):
        return jsonify({'success': False, 'error': f'"{username}" is already enrolled'}), 409

    # ── Hash password — plaintext is discarded after this line ───────────────
    pw_hash  = hash_password(password)
    password = None   # explicitly clear from local scope

    # ── Extract features from each sample ────────────────────────────────────
    positional_samples = []
    aggregate_samples  = []
    for kd in raw_samples:
        positional_samples.append(extract_positional_features(kd))
        aggregate_samples.append(extract_aggregate_features(kd))

    # ── Build profile statistics ──────────────────────────────────────────────
    pos_stats = build_profile_stats(positional_samples)
    agg_stats = build_profile_stats(aggregate_samples)

    # ── Aggregate display stats (from aggregate feature means) ────────────────
    # agg_feat index: 0=mean_hold, 1=std_hold, 2=mean_flight, 3=std_flight, 4=total_time
    def _avg(samples, idx):
        vals = [s[idx] for s in aggregate_samples if idx < len(s)]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    avg_hold      = round(_avg(aggregate_samples, 0))
    avg_hold_sd   = round(_avg(aggregate_samples, 1))
    avg_flight    = round(_avg(aggregate_samples, 2))
    avg_flight_sd = round(_avg(aggregate_samples, 3))
    total_time    = round(_avg(aggregate_samples, 4))

    # ── Save profile ──────────────────────────────────────────────────────────
    keystroke_profile = {
        'shade_index':            shade_index,
        'password_length':        password_length,
        'positional_means':       pos_stats['means'],
        'positional_stds':        pos_stats['stds'],
        'aggregate_means':        agg_stats['means'],
        'aggregate_stds':         agg_stats['stds'],
        'raw_aggregate_samples':  aggregate_samples,   # kept for ML training export
        # Passwords are NOT stored in any form here
        'avg_hold':               avg_hold,
        'avg_hold_sd':            avg_hold_sd,
        'avg_flight':             avg_flight,
        'avg_flight_sd':          avg_flight_sd,
        'total_time':             total_time,
        'samples':                len(raw_samples),
        'enrolled_at':            datetime.utcnow().isoformat(),
    }
    store.save_profile(username, pw_hash, keystroke_profile)

    return jsonify({
        'success':  True,
        'username': username,
        'message':  f'Profile for "{username}" enrolled successfully.',
    })


# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth', methods=['POST'])
def authenticate():
    """
    Authenticate a login attempt.

    Expected JSON body:
    {
        "username":          string,
        "password":          string,    ← plaintext, compared against hash, never stored
        "raw_keydata":       [{"downTime": ms, "upTime": ms}, ...],
        "experiment_label":  "genuine" | "impostor" | "unknown"
    }

    Returns scores for both behavioural signals:
        ml_score       : Signal 1 — Random Forest model
        profile_score  : Signal 2 — User-specific profile deviation
        combined_score : Weighted combination
        decision       : ALLOW | VERIFY | BLOCK
        risk_level     : LOW   | MEDIUM | HIGH
    """
    data = request.get_json(force=True)

    username          = (data.get('username') or '').strip()
    password          = data.get('password') or ''
    raw_keydata       = data.get('raw_keydata', [])
    experiment_label  = data.get('experiment_label', 'unknown')

    event_id  = 'EVT-' + uuid.uuid4().hex[:8].upper()
    timestamp = datetime.utcnow().isoformat()

    # ── Look up profile ───────────────────────────────────────────────────────
    profile_entry = store.get_profile(username)
    if not profile_entry:
        return jsonify({'success': False, 'error': f'User "{username}" not found'}), 404

    # ── Verify password — plaintext discarded after comparison ───────────────
    credential_ok = verify_password(password, profile_entry['password_hash'])
    password = None   # explicitly clear

    # ── Always log the attempt (without the password) ─────────────────────────
    log_base = {
        'event_id':         event_id,
        'timestamp':        timestamp,
        'username':         username,
        'experiment_label': experiment_label,
        'credential_result': credential_ok,
    }

    if not credential_ok:
        # Wrong password — block immediately, no behavioural analysis needed
        risk = risk_evaluate({
            'credential_result': False,
            'ml_score':          None,
            'profile_score':     None,
        })
        explog.log({**log_base, 'ml_score': None, 'profile_score': None,
                    'combined_score': None, 'ml_available': False,
                    'decision': risk['decision'], 'risk_level': risk['risk_level']})
        return jsonify({
            'success':           True,
            'event_id':          event_id,
            'credential_result': False,
            **risk,
        })

    # ── Correct password → behavioural analysis ───────────────────────────────
    kp = profile_entry['keystroke_profile']

    # Signal 2: Profile deviation (positional features vs enrolled profile)
    positional_features = extract_positional_features(raw_keydata)
    pd_result     = calculate_profile_deviation(
        positional_features,
        kp['positional_means'],
        kp['positional_stds'],
    )
    profile_score        = pd_result['score']
    profile_insufficient = pd_result['insufficient_data']

    # Signal 1: ML behavioural score (aggregate features vs RF model)
    aggregate_features = extract_aggregate_features(raw_keydata)
    ml_result          = ml.predict(aggregate_features, username)
    ml_score           = ml_result['ml_score']

    # Risk engine: combine both signals → decision
    risk = risk_evaluate({
        'credential_result':    True,
        'ml_score':             ml_score,
        'profile_score':        profile_score,
        'profile_insufficient': profile_insufficient,
    })

    # Log for experiment evaluation
    explog.log({
        **log_base,
        'ml_score':       ml_score,
        'profile_score':  profile_score,
        'combined_score': risk['combined_score'],
        'ml_available':   ml_result['ml_score'] is not None,
        'decision':       risk['decision'],
        'risk_level':     risk['risk_level'],
    })

    # Build security event for BLOCK on correct credentials
    security_event = None
    if risk['decision'] == 'BLOCK' and credential_ok:
        security_event = {
            'event_type':     'SUSPICIOUS_LOGIN',
            'event_id':       event_id,
            'username':       username,
            'timestamp':      timestamp,
            'decision':       risk['decision'],
            'risk_level':     risk['risk_level'],
            'ml_score':       ml_score,
            'profile_score':  profile_score,
            'combined_score': risk['combined_score'],
            'message': (
                'Security Alert: A login attempt using your correct credentials was detected, '
                'but the typing behaviour did not match your normal behavioural profile. '
                'If this was not you, please contact NOVA Bank Security immediately.'
            ),
        }

    return jsonify({
        'success':           True,
        'event_id':          event_id,
        'credential_result': True,
        'ml_score':          ml_score,
        'profile_score':     profile_score,
        'model_version':     ml_result['model_version'],
        'in_training_set':   ml_result['in_training_set'],
        'security_event':    security_event,
        **risk,
    })


# ══════════════════════════════════════════════════════════════════════════════
# PROFILE DELETION
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/profiles/<username>', methods=['DELETE'])
def delete_profile(username):
    if store.delete_profile(username):
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Not found'}), 404


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/experiment/summary', methods=['GET'])
def experiment_summary():
    return jsonify(explog.get_summary())


@app.route('/api/experiment/log', methods=['GET'])
def experiment_log():
    """
    Return experiment log as JSON for the research dashboard.
    Supports optional query params:
        ?limit=N     — return last N entries (default: all)
        ?label=genuine|impostor|unknown
    """
    rows  = explog.load_all()
    label = request.args.get('label')
    if label:
        rows = [r for r in rows if r.get('experiment_label') == label]
    try:
        limit = int(request.args.get('limit', 0))
        if limit > 0:
            rows = rows[-limit:]
    except ValueError:
        pass
    # Return newest first for dashboard display
    return jsonify({'entries': list(reversed(rows)), 'total': len(rows)})


@app.route('/api/experiment/export', methods=['GET'])
def experiment_export():
    """Download the experiment log as CSV."""
    csv_content = explog.export_csv()
    return Response(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=experiment_log.csv'},
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 60)
    print('NOVA Bank Keystroke Dynamics — Backend')
    print('Running on http://localhost:5001')
    print('=' * 60)
    app.run(debug=True, port=5001, host='0.0.0.0')
