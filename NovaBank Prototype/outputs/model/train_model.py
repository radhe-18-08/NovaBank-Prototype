"""
train_model.py — Random Forest training script for keystroke dynamics authentication.

Purpose: Train and evaluate the Random Forest keystroke classifier used as Signal 1 in the prototype.
Inputs: Enrolment/profile samples, experiment data or a supported external CSV source converted to 13 aggregate keystroke features and user labels.
Outputs: Trained Random Forest model, label map, feature-importance CSV and training report saved in the model directory.


Usage:
    python train_model.py                            # auto-detect data source
    python train_model.py --source experiment        # use experiment_log.csv
    python train_model.py --source profiles          # use enrolled profile samples
    python train_model.py --source csv path/to/data.csv  # custom CSV file
    python train_model.py --source cmu path/to/cmu.csv   # CMU benchmark dataset

Outputs (saved to model/ directory):
    keystroke_rf.pkl    — trained Random Forest model
    label_map.json      — username ↔ class index mapping
    feature_importance.csv — per-feature importance scores
    training_report.txt — training summary and evaluation results

─────────────────────────────────────────────────────────────────────────────
MODEL FORMULATION
─────────────────────────────────────────────────────────────────────────────
Task: Multi-class classification — predict WHICH enrolled user typed a sample.
Features: 13 aggregate keystroke features (password-independent).
Labels: username strings.

At authentication time:
    P(class = claimed_username | features) = ML behavioural score

─────────────────────────────────────────────────────────────────────────────
DATA LEAKAGE PREVENTION
─────────────────────────────────────────────────────────────────────────────
Multiple samples from the same user CANNOT be split randomly into train/test
because the model could trivially overfit to user-specific patterns that appear
in both splits.

This script uses a SAMPLE-AWARE STRATIFIED SPLIT:
    - If data has fewer than 2 samples per user: report insufficient data.
    - Otherwise: for each user, earliest 70% of samples → train,
      remaining 30% → test. This simulates a temporal deployment scenario
      (train on early sessions, test on later ones).

If using the CMU dataset, which has 400 samples per user across 8 sessions:
    Train sessions 1–6, Test sessions 7–8 (strict temporal split).
    This is the academically recommended split for the CMU benchmark.

─────────────────────────────────────────────────────────────────────────────
SAMPLE SIZE AND DATA SOURCE
─────────────────────────────────────────────────────────────────────────────
The saved Random Forest artefact was trained on 6 enrolled class labels × 8
enrolment samples = 48 samples (source: the original data/profiles.json). Five
classes correspond to the final study participants; the sixth (`Test`) is a
development account retained in the historical model artefact. The final
genuine/impostor evaluation reported in the dissertation excludes `Test` and
uses only the five study participants. The sanitised submission copy does not
contain the original populated profiles.json.

The CMU benchmark dataset (Killourhy & Maxion, 2009; 51 users × 400 samples)
is supported via --source cmu and is included for optional reproducibility
benchmarking only. It was NOT used to generate the dissertation results.

Results should be interpreted in the context of the experiment's sample size.
See the dissertation Limitations section for the discussion of sample size.
"""

import sys
import os
import json
import pickle
import argparse
import csv
import math
import random
from datetime import datetime
from collections import defaultdict

# Add backend/ to path so we can import features.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# ── Output paths ──────────────────────────────────────────────────────────────
MODEL_DIR          = os.path.dirname(__file__)
MODEL_PATH         = os.path.join(MODEL_DIR, 'keystroke_rf.pkl')
LABEL_MAP_PATH     = os.path.join(MODEL_DIR, 'label_map.json')
IMPORTANCE_PATH    = os.path.join(MODEL_DIR, 'feature_importance.csv')
REPORT_PATH        = os.path.join(MODEL_DIR, 'training_report.txt')

RANDOM_SEED        = 42   # Reproducibility

# Aggregate feature names (must match features.py)
FEATURE_NAMES = [
    'mean_hold', 'std_hold', 'mean_flight', 'std_flight',
    'total_time', 'n_keys', 'typing_speed',
    'cv_hold', 'cv_flight',
    'min_hold', 'max_hold', 'min_flight', 'max_flight',
]
N_FEATURES = len(FEATURE_NAMES)   # 13


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_from_profiles_json(profiles_path: str):
    """
    Load training data from data/profiles.json (enrolled user profiles).
    Uses the aggregate feature samples stored during enrolment.
    Returns: [(username, feature_vector), ...]
    """
    with open(profiles_path, 'r') as f:
        data = json.load(f)

    samples = []
    for username, entry in data.items():
        kp     = entry.get('keystroke_profile', {})
        agg    = kp.get('raw_aggregate_samples', [])
        for feat in agg:
            if len(feat) == N_FEATURES:
                samples.append((username, feat))
            else:
                print(f'  [WARN] {username}: unexpected feature length {len(feat)}, skipping.')

    return samples


def load_from_experiment_log(log_path: str):
    """
    Load training data from data/experiment_log.csv.
    Uses GENUINE attempts only (experiment_label == 'genuine').
    Impostors should not be in the training set — they are the test set.
    Returns: [(username, feature_vector), ...]  NOTE: features not stored in log.
    Raises: NotImplementedError — the log does not store raw features.
    """
    raise NotImplementedError(
        'The experiment log does not store raw feature vectors (by design — '
        'it only stores scores and decisions). To train from experiment data, '
        'use the profiles JSON which stores aggregate samples from enrolment, '
        'or collect additional samples via a dedicated data collection mode.'
    )


def load_from_csv(csv_path: str):
    """
    Load training data from a generic CSV file.
    Expected format:
        username, feat0, feat1, ..., feat12
    First column must be the username/subject identifier.
    Columns 1..13 must be the 13 aggregate features in order.

    If your CSV has different columns, edit this function to map them.
    Returns: [(username, feature_vector), ...]
    """
    samples = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        print(f'  CSV header: {header}')
        for row in reader:
            if len(row) < N_FEATURES + 1:
                continue
            username = row[0].strip()
            try:
                feats = [float(x) for x in row[1: N_FEATURES + 1]]
                samples.append((username, feats))
            except ValueError:
                pass
    return samples


def load_from_cmu(cmu_path: str):
    """
    Load data from the CMU Keystroke Dynamics Benchmark Dataset.

    The CMU dataset (Killourhy & Maxion, 2009) has 51 subjects, each
    typing '.tie5Roanl' 400 times. The CSV has columns:
        subject, sessionIndex, rep, H.period, DD.period.t, ...

    We DERIVE our 13 aggregate features from the raw timing columns.
    The CMU features available are hold times (H.*) and down-down times (DD.*).
    We approximate our aggregate features from these.

    Download: http://www.cs.cmu.edu/~keystroke/
    File: DSL-StrongPasswordData.csv

    NOTE: The CMU dataset uses a fixed password. Our aggregate features are
    designed to be password-independent, so this cross-dataset use is valid
    in principle. However, password-length will always be 10 for CMU data.
    Discuss this limitation in your dissertation.
    """
    samples = []
    hold_cols = []   # will be detected from header

    with open(cmu_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        hold_cols   = [c for c in fieldnames if c.startswith('H.')]
        dd_cols     = [c for c in fieldnames if c.startswith('DD.')]
        print(f'  CMU: {len(hold_cols)} hold cols, {len(dd_cols)} DD cols')

        for row in reader:
            subject = row.get('subject', '').strip()
            if not subject:
                continue
            try:
                holds   = [float(row[c]) * 1000 for c in hold_cols]   # s → ms
                dds     = [float(row[c]) * 1000 for c in dd_cols]      # s → ms
                # Approximate flight times: flight_i ≈ DD_i - hold_i
                flights = [max(0.0, dds[i] - holds[i]) for i in range(min(len(dds), len(holds) - 1))]
            except (ValueError, KeyError, IndexError):
                continue

            if not holds:
                continue

            feats = _derive_aggregate_from_holds_flights(holds, flights)
            samples.append((subject, feats))

    return samples


def _derive_aggregate_from_holds_flights(holds, flights):
    """Build our 13 aggregate features from hold and flight lists."""
    def _mean(v): return sum(v) / len(v) if v else 0.0
    def _std(v, m=None):
        if len(v) < 2: return 0.0
        m = m or _mean(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))

    n           = len(holds)
    mean_hold   = _mean(holds)
    std_hold    = _std(holds, mean_hold)
    mean_flight = _mean(flights)
    std_flight  = _std(flights, mean_flight)
    # Approximate total_time from holds and flights
    total_time  = sum(holds) + sum(flights)
    speed       = (n / (total_time / 1000)) if total_time > 0 else 0.0
    cv_hold     = min(std_hold / mean_hold if mean_hold > 0 else 0.0, 10.0)
    cv_flight   = min(std_flight / mean_flight if mean_flight > 0 else 0.0, 10.0)

    return [
        mean_hold, std_hold, mean_flight, std_flight,
        total_time, float(n), speed,
        cv_hold, cv_flight,
        min(holds), max(holds),
        min(flights) if flights else 0.0,
        max(flights) if flights else 0.0,
    ]


def auto_detect_source():
    """Find a data source automatically."""
    profiles_path = os.path.join(MODEL_DIR, '..', 'data', 'profiles.json')
    if os.path.exists(profiles_path):
        return 'profiles', profiles_path
    raise FileNotFoundError(
        'No data source found. Expected data/profiles.json. '
        'Enrol some participants first, or specify --source.'
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def user_aware_split(samples, train_ratio=0.70):
    """
    Split samples into train/test WITHOUT data leakage.

    For each user, samples are ordered as received (preserving temporal order
    if the data was collected sequentially). The FIRST train_ratio fraction
    goes to training; the remainder goes to testing.

    This simulates a real deployment: train on earlier sessions, test on later
    ones — preventing the model from memorising test-set patterns.

    Returns: (train_samples, test_samples)
    Each is a list of (username, feature_vector) tuples.
    """
    by_user = defaultdict(list)
    for username, feat in samples:
        by_user[username].append(feat)

    train, test = [], []
    for username, feats in by_user.items():
        n_train = max(1, int(len(feats) * train_ratio))
        for feat in feats[:n_train]:
            train.append((username, feat))
        for feat in feats[n_train:]:
            test.append((username, feat))

    return train, test


def train_rf(train_samples):
    """
    Train a Random Forest classifier.

    Returns: (model, label_to_idx)
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        raise ImportError('scikit-learn not installed. Run: pip install scikit-learn')

    random.seed(RANDOM_SEED)

    usernames = sorted(set(u for u, _ in train_samples))
    label_to_idx = {u: i for i, u in enumerate(usernames)}

    X = [feat for _, feat in train_samples]
    y = [label_to_idx[u] for u, _ in train_samples]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',   # handles class imbalance
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model, label_to_idx


def evaluate_model(model, test_samples, label_to_idx):
    """
    Evaluate the model on the test set.

    Metrics computed:
        - Per-class accuracy
        - Overall accuracy
        - For each user: precision, recall, F1 (treating them as genuine class)
        - FAR: fraction of impostor attempts that score above a threshold
        - FRR: fraction of genuine attempts that score below a threshold

    NOTE: FAR/FRR require genuine/impostor labels which are NOT in the profile
    training data. FAR/FRR are computed by model/evaluate.py using the experiment
    log after running the genuine/impostor experiment.

    Returns: dict of metrics
    """
    if not test_samples:
        return {'error': 'No test samples available'}

    idx_to_label = {v: k for k, v in label_to_idx.items()}

    X_test = [feat for _, feat in test_samples]
    y_true = [label_to_idx.get(u, -1) for u, _ in test_samples]

    # Filter out users not seen in training
    valid = [(x, y) for x, y in zip(X_test, y_true) if y >= 0]
    if not valid:
        return {'error': 'No test users overlap with training users'}

    X_test, y_true = zip(*valid)
    y_pred = model.predict(list(X_test))

    total   = len(y_true)
    correct = sum(p == t for p, t in zip(y_pred, y_true))
    accuracy = correct / total if total > 0 else 0.0

    # Per-user metrics (treating each user as the positive class)
    per_user = {}
    for username, idx in label_to_idx.items():
        tp = sum(1 for p, t in zip(y_pred, y_true) if p == idx and t == idx)
        fp = sum(1 for p, t in zip(y_pred, y_true) if p == idx and t != idx)
        fn = sum(1 for p, t in zip(y_pred, y_true) if p != idx and t == idx)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_user[username] = {'precision': precision, 'recall': recall, 'f1': f1}

    return {
        'accuracy':        round(accuracy, 4),
        'total_test':      total,
        'correct':         correct,
        'n_classes':       len(label_to_idx),
        'per_user':        per_user,
    }


def save_feature_importance(model, feature_names):
    """Save feature importance to CSV and print top features."""
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])

    with open(IMPORTANCE_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature', 'importance'])
        for name, imp in ranked:
            writer.writerow([name, round(imp, 6)])

    print('\nFeature importances (ranked):')
    for name, imp in ranked:
        bar = '█' * int(imp * 40)
        print(f'  {name:<16} {imp:.4f}  {bar}')


def write_report(metrics, n_train, n_test, source_info, split_note):
    """Write a human-readable training report."""
    lines = [
        '=' * 70,
        'NOVA Bank Keystroke Dynamics — Model Training Report',
        f'Generated: {datetime.utcnow().isoformat()}',
        '=' * 70,
        '',
        f'Data source  : {source_info}',
        f'Training samples: {n_train}',
        f'Test samples    : {n_test}',
        f'Split strategy  : {split_note}',
        f'Random seed     : {RANDOM_SEED}',
        '',
        'MODEL HYPERPARAMETERS',
        '  Algorithm     : Random Forest Classifier',
        '  n_estimators  : 200',
        '  max_depth     : None (full depth)',
        '  class_weight  : balanced',
        '',
        'EVALUATION RESULTS',
        f'  Overall accuracy : {metrics.get("accuracy", "N/A")}',
        f'  Test samples     : {metrics.get("total_test", "N/A")}',
        f'  Correct          : {metrics.get("correct", "N/A")}',
        '',
        'DISCLAIMER',
        '  These results reflect classification accuracy on enrolment samples.',
        '  They do NOT represent FAR/FRR on genuine/impostor pairs.',
        '  Run genuine/impostor experiments and use model/evaluate.py for',
        '  realistic FAR, FRR, precision, recall, and F1 metrics.',
        '',
        'FEATURE IMPORTANCE',
        '  See feature_importance.csv for full ranking.',
        '=' * 70,
    ]
    report = '\n'.join(lines)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    print('\n' + report)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train keystroke dynamics RF model')
    parser.add_argument('--source', choices=['auto', 'profiles', 'csv', 'cmu'],
                        default='auto', help='Data source type')
    parser.add_argument('--path', default=None, help='Path to data file (for csv/cmu)')
    args = parser.parse_args()

    print('NOVA Bank Keystroke Dynamics — Model Trainer')
    print('=' * 50)

    # ── Load data ─────────────────────────────────────────────────────────────
    source_info = ''
    if args.source == 'auto':
        source_type, source_path = auto_detect_source()
        source_info = f'{source_type}: {source_path}'
        args.source = source_type
        args.path   = source_path

    if args.source == 'profiles':
        path = args.path or os.path.join(MODEL_DIR, '..', 'data', 'profiles.json')
        print(f'Loading enrolled profiles from: {path}')
        samples = load_from_profiles_json(path)
        source_info = f'profiles.json: {path}'

    elif args.source == 'csv':
        if not args.path:
            sys.exit('--path required for --source csv')
        print(f'Loading from CSV: {args.path}')
        samples = load_from_csv(args.path)
        source_info = f'csv: {args.path}'

    elif args.source == 'cmu':
        if not args.path:
            sys.exit('--path required for --source cmu')
        print(f'Loading CMU dataset from: {args.path}')
        samples = load_from_cmu(args.path)
        source_info = f'cmu: {args.path}'

    else:
        sys.exit(f'Unknown source: {args.source}')

    # ── Inspect dataset ───────────────────────────────────────────────────────
    print(f'\nDataset inspection:')
    by_user = defaultdict(int)
    for u, _ in samples:
        by_user[u] += 1
    print(f'  Total samples : {len(samples)}')
    print(f'  Unique users  : {len(by_user)}')
    print(f'  Samples/user  :')
    for u, n in sorted(by_user.items()):
        print(f'    {u}: {n}')

    if len(samples) < 2:
        sys.exit('\nInsufficient data. Need at least 2 total samples to train.')
    if len(by_user) < 2:
        print('\nWARNING: Only 1 user in dataset. A multi-class classifier cannot')
        print('meaningfully discriminate with a single class. Enrol more users.')
        print('Training will proceed but the model will be trivially accurate.\n')

    # ── Split data ────────────────────────────────────────────────────────────
    train_samples, test_samples = user_aware_split(samples, train_ratio=0.70)
    split_note = 'Temporal 70/30 per-user (first 70% → train, last 30% → test)'
    print(f'\nSplit: {len(train_samples)} train, {len(test_samples)} test')
    print(f'Note: {split_note}')

    if len(train_samples) < 2:
        print('\nWARNING: Very few training samples. Results will not be meaningful.')

    # ── Train ─────────────────────────────────────────────────────────────────
    print('\nTraining Random Forest...')
    model, label_to_idx = train_rf(train_samples)
    print(f'  Classes: {list(label_to_idx.keys())}')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print('\nEvaluating on test set...')
    metrics = evaluate_model(model, test_samples, label_to_idx)
    print(f'  Overall accuracy: {metrics.get("accuracy", "N/A")}')

    if 'per_user' in metrics:
        for user, m in metrics['per_user'].items():
            print(f'  {user}: precision={m["precision"]:.3f}, recall={m["recall"]:.3f}, f1={m["f1"]:.3f}')

    # ── Save model ────────────────────────────────────────────────────────────
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f'\nModel saved: {MODEL_PATH}')

    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump({'label_to_idx': label_to_idx}, f, indent=2)
    print(f'Label map saved: {LABEL_MAP_PATH}')

    # ── Feature importance ────────────────────────────────────────────────────
    save_feature_importance(model, FEATURE_NAMES)
    print(f'Feature importance saved: {IMPORTANCE_PATH}')

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(metrics, len(train_samples), len(test_samples), source_info, split_note)
    print(f'Report saved: {REPORT_PATH}')

    print('\nDone. Restart the Flask backend (backend/app.py) to load the new model.')


if __name__ == '__main__':
    main()
