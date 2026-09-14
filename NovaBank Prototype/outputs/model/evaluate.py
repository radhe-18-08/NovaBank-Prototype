"""
evaluate.py — Compute FAR, FRR, and all classification metrics from experiment log.

Purpose: Evaluate labelled authentication attempts and calculate security/performance metrics for the dissertation experiments.
Inputs: Experiment-log CSV rows containing genuine/impostor labels and behavioural scores, plus an optional decision threshold.
Outputs: FAR, FRR, accuracy, precision/recall/F1, confusion counts, EER/threshold-sweep data, console output and evaluation report files.


Usage:
    python evaluate.py                              # uses data/experiment_log.csv
    python evaluate.py --log path/to/log.csv
    python evaluate.py --threshold 0.55            # test a different ALLOW threshold
    python evaluate.py --sweep                     # sweep thresholds, find EER

Outputs:
    Prints metrics to stdout.
    Saves evaluation_report.txt and roc_data.csv in model/ directory.

─────────────────────────────────────────────────────────────────────────────
METRIC DEFINITIONS
─────────────────────────────────────────────────────────────────────────────

FAR (False Acceptance Rate):
    Proportion of IMPOSTOR attempts that were ALLOWED.
    FAR = impostor_allowed / total_impostor_attempts
    Lower FAR = better security.

FRR (False Rejection Rate):
    Proportion of GENUINE attempts that were BLOCKED.
    FRR = genuine_blocked / total_genuine_attempts
    Lower FRR = better usability.

EER (Equal Error Rate):
    The threshold at which FAR ≈ FRR.
    Commonly used as a single-number comparison metric for biometric systems.

Accuracy:
    Computed treating genuine as positive class, impostor as negative.
    TP = genuine ALLOWED,  TN = impostor BLOCKED,
    FP = impostor ALLOWED, FN = genuine BLOCKED.

─────────────────────────────────────────────────────────────────────────────
NOTE ON MISSING ML SCORES
─────────────────────────────────────────────────────────────────────────────
If the ML model was not trained when an attempt was logged, ml_score will be
empty. This evaluator handles three evaluation modes:
    1. combined_score only (primary metric — uses whatever the system decided)
    2. profile_score only (evaluates Signal 2 in isolation)
    3. ml_score only (evaluates Signal 1 in isolation — requires ML to be trained)

This lets the dissertation compare: ML alone vs Profile alone vs Combined.
"""

import csv
import os
import sys
import json
import argparse
from collections import defaultdict


REPORT_PATH   = os.path.join(os.path.dirname(__file__), 'evaluation_report.txt')
ROC_PATH      = os.path.join(os.path.dirname(__file__), 'roc_data.csv')
LOG_DEFAULT   = os.path.join(os.path.dirname(__file__), '..', 'data', 'experiment_log.csv')


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(val, default=None):
    try:
        return float(val) if val and val.strip() else default
    except (ValueError, AttributeError):
        return default


def _compute_metrics(rows, score_field, threshold):
    """
    Compute TP, FP, TN, FN, FAR, FRR given a score field and threshold.
    Rows with missing score are excluded.
    """
    tp = fp = tn = fn = 0
    for r in rows:
        label = r.get('experiment_label', '').lower()
        score = _safe_float(r.get(score_field))
        if score is None:
            continue
        predicted_allow = score >= threshold
        if label == 'genuine':
            if predicted_allow: tp += 1
            else:               fn += 1
        elif label == 'impostor':
            if predicted_allow: fp += 1
            else:               tn += 1

    total_gen = tp + fn
    total_imp = fp + tn
    far = fp / total_imp if total_imp > 0 else None
    frr = fn / total_gen if total_gen > 0 else None
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0
          else None)
    return {
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'genuine_total':  total_gen,
        'impostor_total': total_imp,
        'FAR':       round(far, 4) if far is not None else 'N/A',
        'FRR':       round(frr, 4) if frr is not None else 'N/A',
        'accuracy':  round(acc, 4) if acc is not None else 'N/A',
        'precision': round(precision, 4) if precision is not None else 'N/A',
        'recall':    round(recall, 4) if recall is not None else 'N/A',
        'f1':        round(f1, 4) if f1 is not None else 'N/A',
    }


def _find_eer(rows, score_field):
    """
    Sweep thresholds to find the Equal Error Rate point.
    Returns (eer_threshold, eer_value) or None.
    """
    thresholds = [t / 100 for t in range(0, 101)]
    results = []
    for t in thresholds:
        m = _compute_metrics(rows, score_field, t)
        far = m['FAR'] if isinstance(m['FAR'], float) else None
        frr = m['FRR'] if isinstance(m['FRR'], float) else None
        if far is not None and frr is not None:
            results.append((t, far, frr, abs(far - frr)))

    if not results:
        return None, None

    # Find threshold where |FAR - FRR| is minimised
    best = min(results, key=lambda x: x[3])
    eer  = (best[1] + best[2]) / 2
    return round(best[0], 2), round(eer, 4)


def _confusion_matrix_str(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    return (
        f'\n  Confusion Matrix (Genuine=Positive, Impostor=Negative):\n'
        f'                    Predicted ALLOW   Predicted BLOCK\n'
        f'  Genuine  (pos)         {tp:>5}  (TP)        {fn:>5}  (FN)\n'
        f'  Impostor (neg)         {fp:>5}  (FP)        {tn:>5}  (TN)\n'
        f'  Total: {total}'
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Evaluate keystroke dynamics experiment results')
    parser.add_argument('--log',       default=LOG_DEFAULT, help='Path to experiment_log.csv')
    parser.add_argument('--threshold', type=float, default=0.65, help='ALLOW threshold')
    parser.add_argument('--sweep',     action='store_true', help='Sweep thresholds to find EER')
    args = parser.parse_args()

    if not os.path.exists(args.log):
        sys.exit(f'Log file not found: {args.log}\nRun the genuine/impostor experiment first.')

    with open(args.log, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    genuine   = [r for r in rows if r.get('experiment_label') == 'genuine']
    impostors = [r for r in rows if r.get('experiment_label') == 'impostor']
    unknown   = [r for r in rows if r.get('experiment_label') not in ('genuine', 'impostor')]

    print('=' * 70)
    print('NOVA Bank Keystroke Dynamics — Experiment Evaluation')
    print('=' * 70)
    print(f'Log file       : {args.log}')
    print(f'Total entries  : {len(rows)}')
    print(f'  Genuine      : {len(genuine)}')
    print(f'  Impostor     : {len(impostors)}')
    print(f'  Unlabelled   : {len(unknown)}')

    if not genuine and not impostors:
        print('\nNo labelled entries found. Run the genuine/impostor experiment.')
        print('In nova_bank_portal.html, set the experiment label before each attempt.')
        return

    if not genuine:
        print('\nWARNING: No genuine attempts logged.')
    if not impostors:
        print('\nWARNING: No impostor attempts logged. FAR cannot be computed.')

    threshold = args.threshold

    report_lines = [
        '=' * 70,
        'NOVA Bank Keystroke Dynamics — Evaluation Report',
        f'Threshold: {threshold}',
        f'Total labelled: {len(genuine)} genuine, {len(impostors)} impostor',
    ]

    # ── Evaluate three signal configurations ─────────────────────────────────
    configs = [
        ('combined_score',  'Combined (ML + Profile) — primary system decision'),
        ('profile_score',   'Profile Deviation Only — Signal 2 in isolation'),
        ('ml_score',        'ML (Random Forest) Only — Signal 1 in isolation'),
    ]

    roc_records = []

    for field, label in configs:
        print(f'\n── {label} ──')
        m = _compute_metrics(rows, field, threshold)
        print(f'  Threshold   : {threshold}')
        print(f'  Genuine     : {m["genuine_total"]} attempts')
        print(f'  Impostor    : {m["impostor_total"]} attempts')
        print(f'  FAR         : {m["FAR"]}   (impostors incorrectly ALLOWED)')
        print(f'  FRR         : {m["FRR"]}   (genuine users incorrectly BLOCKED)')
        print(f'  Accuracy    : {m["accuracy"]}')
        print(f'  Precision   : {m["precision"]}')
        print(f'  Recall      : {m["recall"]}')
        print(f'  F1          : {m["f1"]}')
        print(_confusion_matrix_str(m['TP'], m['FP'], m['TN'], m['FN']))

        report_lines += [
            '', f'── {label} (threshold={threshold}) ──',
            f'FAR={m["FAR"]}, FRR={m["FRR"]}, Acc={m["accuracy"]}, F1={m["f1"]}',
            f'TP={m["TP"]}, FP={m["FP"]}, TN={m["TN"]}, FN={m["FN"]}',
        ]

        if args.sweep:
            eer_t, eer_v = _find_eer(rows, field)
            if eer_v is not None:
                print(f'  EER         : {eer_v:.4f} at threshold {eer_t}')
                report_lines.append(f'EER={eer_v} at threshold {eer_t}')

            # Collect ROC points
            for t in [x / 100 for x in range(0, 101)]:
                rm = _compute_metrics(rows, field, t)
                far = rm['FAR'] if isinstance(rm['FAR'], float) else ''
                frr = rm['FRR'] if isinstance(rm['FRR'], float) else ''
                roc_records.append({'signal': field, 'threshold': t, 'FAR': far, 'FRR': frr})

    # ── Per-user breakdown ────────────────────────────────────────────────────
    users = sorted(set(r.get('username', '') for r in rows))
    if len(users) > 1:
        print('\n── Per-User Breakdown ──')
        report_lines.append('')
        report_lines.append('── Per-User Breakdown ──')
        for user in users:
            user_rows = [r for r in rows if r.get('username') == user]
            gen_rows  = [r for r in user_rows if r.get('experiment_label') == 'genuine']
            imp_rows  = [r for r in user_rows if r.get('experiment_label') == 'impostor']
            m         = _compute_metrics(user_rows, 'combined_score', threshold)
            line = (f'  {user:<20} genuine={len(gen_rows)} impostor={len(imp_rows)} '
                    f'FAR={m["FAR"]} FRR={m["FRR"]}')
            print(line)
            report_lines.append(line)

    # ── Save ROC data ─────────────────────────────────────────────────────────
    if roc_records:
        with open(ROC_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['signal', 'threshold', 'FAR', 'FRR'])
            writer.writeheader()
            writer.writerows(roc_records)
        print(f'\nROC data saved: {ROC_PATH}')

    # ── Save report ───────────────────────────────────────────────────────────
    report_text = '\n'.join(report_lines)
    with open(REPORT_PATH, 'w') as f:
        f.write(report_text)
    print(f'\nEvaluation report saved: {REPORT_PATH}')


if __name__ == '__main__':
    main()
