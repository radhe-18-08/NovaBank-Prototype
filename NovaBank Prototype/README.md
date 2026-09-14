# NOVA Bank — Keystroke Dynamics Authentication Prototype

**MSc FinTech Dissertation Project**  
Academic prototype demonstrating keystroke dynamics as a second-factor authentication signal in an online banking context.

---

## Overview

NOVA Bank is a full-stack proof-of-concept that authenticates users using the rhythm of their typing — specifically, the hold (dwell) and flight (inter-key) times captured while entering a password. No additional hardware is required; the system runs entirely in a web browser with a Python/Flask backend.

The system combines two independent behavioural signals:

- **Signal 1 — ML Behavioural Score**: A Random Forest multi-class classifier trained on 13 aggregate, password-independent keystroke features. At login, the classifier returns `P(class = claimed_user | features)` as a probability.
- **Signal 2 — Profile Deviation Score**: A per-user, per-position z-score comparison against the enrolled baseline (hold and flight times at each keystroke position). Score = `max(0, 1 − avg_z / DIVISOR)`.

The two scores are combined as a weighted average (`ML_WEIGHT = 0.50`, `PROFILE_WEIGHT = 0.50`) to produce a `combined_score`, which is mapped to one of three decisions:

| Decision | combined_score | Meaning |
|----------|---------------|---------|
| **ALLOW** | ≥ 0.55 | Low risk — access granted |
| **VERIFY** | ≥ 0.28 | Medium risk — OTP step-up required |
| **BLOCK** | < 0.28 | High risk — access denied |

The offline evaluation script (`model/evaluate.py`) uses a separate binary threshold of **0.65** for FAR/FRR reporting. This is intentionally distinct from the three-level operational thresholds above, as explained in the dissertation Methods section.

---

## System Architecture

```
Nova-bank-keystroke-authentication--main/
├── README.md
├── .gitignore
└── outputs/
    ├── backend/
    │   ├── app.py               Flask REST API (port 5001)
    │   ├── auth.py              Argon2id password hashing/verification
    │   ├── features.py          Feature extraction (single source of truth)
    │   ├── profile_deviation.py Signal 2 — per-user profile deviation scorer
    │   ├── profile_store.py     JSON-backed user profile CRUD
    │   ├── ml_inference.py      Signal 1 — Random Forest model serving
    │   ├── risk_engine.py       Decision logic — combines signals → ALLOW/VERIFY/BLOCK
    │   └── experiment_logger.py CSV experiment logger
    ├── data/
    │   ├── profiles.json        Enrolled user profiles and Argon2id hashes
    │   ├── experiment_log.csv   Live experiment log (append-only)
    │   └── experiment_*.csv     Archived experiment runs
    ├── model/
    │   ├── train_model.py       RF training script
    │   ├── evaluate.py          FAR/FRR/EER evaluation (binary threshold = 0.65)
    │   ├── keystroke_rf.pkl     Trained Random Forest model
    │   ├── label_map.json       Class index ↔ username mapping (6 users)
    │   ├── feature_config.json  Feature/threshold documentation (not imported by code)
    │   ├── feature_importance.csv
    │   ├── roc_data.csv
    │   ├── evaluation_report.txt
    │   └── training_report.txt
    ├── enrolment_studio.html    Enrolment UI (capture 8 samples per user)
    ├── nova_bank_portal.html    Login portal (live authentication)
    ├── experiment_console.html  Experimenter UI (label genuine/impostor attempts)
    └── tests/
        ├── conftest.py
        ├── test_auth.py
        ├── test_features.py
        ├── test_profile_deviation.py
        ├── test_risk_engine.py
        ├── test_experiment_logger.py
        ├── test_profile_store.py
        ├── test_ml_inference.py
        └── test_api.py
```

---

## Prerequisites

- Python 3.9 or later
- pip

Tested on macOS 14 (Sonoma) and Ubuntu 22.04 with Python 3.11.

---

## Installation

```bash
# 1. Clone / unzip the project
cd Nova-bank-keystroke-authentication--main

# 2. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r outputs/requirements.txt
```

### Dependencies (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| Flask | REST API server |
| flask-cors | Cross-origin requests from browser UIs |
| argon2-cffi | Argon2id password hashing |
| scikit-learn | Random Forest classifier |
| numpy | Numerical feature computation |
| pytest ≥ 7.0 | Unit test runner |

---

## Running the Backend

```bash
cd outputs/backend
python app.py
```

The API starts on **http://127.0.0.1:5001**. Keep this terminal open.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness check |
| POST | `/api/enrol` | Register/enrol a user from 8 keystroke samples |
| POST | `/api/auth` | Authenticate — returns decision + scores |
| GET | `/api/profiles` | List enrolled profiles (no hashes returned) |
| DELETE | `/api/profiles/<username>` | Delete a user profile |
| GET | `/api/experiment/summary` | Experiment counts/summary |
| GET | `/api/experiment/log` | Read logged experiment rows |
| GET | `/api/experiment/export` | Export the experiment CSV |

---

## Using the Web Interfaces

Open each HTML file directly in a browser (no web server required — they call the Flask API on port 5001).

**Step 1 — Enrolment** (`enrolment_studio.html`)  
Register a username and password, then type the password 8 times to build the keystroke baseline. 8 samples are required before the profile is saved.

**Step 2 — Login** (`nova_bank_portal.html`)  
Enter username and password. The system runs the full two-signal authentication pipeline and returns a decision.

**Step 3 — Experimentation** (`experiment_console.html`)  
Run labelled genuine and impostor sessions. Each attempt is logged to `data/experiment_log.csv` with its `experiment_label`, `combined_score`, and `decision`. The label is metadata only — it is stored in the log for post-hoc evaluation and **never passed to the risk engine**.

---

## Training the Model

The submitted model (`model/keystroke_rf.pkl`) contains **6 class labels**: the five final study participants plus one development account named `Test`. The saved training report records 48 enrolment observations in total (30 training, 18 held-out test). The final controlled genuine/impostor evaluation reported in the dissertation uses only the five study participants; the `Test` development account is not part of that final participant evaluation.

The final submission copy intentionally contains a sanitised `data/profiles.json`, so the exact historical model cannot be retrained from this archive alone. To train a new model after creating fresh profiles:

```bash
cd outputs
python model/train_model.py --source profiles
```

To train from the experiment log (if additional samples have been collected):

```bash
python model/train_model.py --source experiment
```

The CMU Keystroke Dynamics Benchmark Dataset (Killourhy & Maxion, 2009) is also supported via `--source cmu --path /path/to/cmu.csv` for optional benchmarking. It was **not** used to produce the dissertation results.

---

## Running the Evaluation

```bash
cd outputs
python model/evaluate.py
```

This computes FAR, FRR, and EER using a binary decision threshold of **0.65** (the offline evaluation threshold, distinct from the three-level operational thresholds). Use `--threshold` to override:

```bash
python model/evaluate.py --threshold 0.55
python model/evaluate.py --sweep           # sweep all thresholds and plot ROC
```

Results are written to `model/evaluation_report.txt` and `model/roc_data.csv`.

---

## Running the Tests

```bash
cd outputs
python -m pytest tests/ -v
```

Expected result: **147 passed** (verified against the submitted codebase).

The test suite covers: password hashing (`test_auth.py`), feature extraction (`test_features.py`), profile deviation scoring (`test_profile_deviation.py`), risk engine constants and decision boundaries (`test_risk_engine.py`), experiment logging (`test_experiment_logger.py`), profile storage (`test_profile_store.py`), ML inference (`test_ml_inference.py`), and Flask API endpoints (`test_api.py`).

Tests use `tmp_path` and `unittest.mock` throughout — they **never modify** `data/profiles.json`, `data/experiment_log.csv`, or the trained model.

---

## Key Design Decisions

**Why Argon2id?**  
Argon2id (time_cost=3, memory_cost=65536 KiB, parallelism=2, hash_len=32, salt_len=16) is resistant to GPU and side-channel attacks. The password hash is stored only in `profiles.json` and is never returned to the browser, never logged, and never passed to the ML pipeline.

**Why 8 enrolment samples?**  
8 samples provide enough variance for a per-position z-score baseline (Signal 2) while remaining feasible for a small-scale experiment. The `N_SAMP = 8` constant is defined in `backend/features.py`.

**Why a separate offline evaluation threshold (0.65)?**  
The three-level operational thresholds (0.55 / 0.28) were selected for the prototype to create ALLOW, VERIFY and BLOCK regions; a borderline session is routed to step-up verification rather than a hard block. The offline binary threshold (0.65) collapses the VERIFY zone to produce a single accept/reject boundary suitable for FAR/FRR computation on labelled experiment data.

**Positional vs. aggregate features**  
Signal 2 uses `2n-1` positional features (password-specific hold and flight times at each keystroke position). Signal 1 uses 13 aggregate, password-independent summary statistics that generalise across users with different passwords.

---

## Security Notes

The following invariants are enforced in code and tested in the unit test suite:

- Plaintext passwords are never stored, logged, returned to the browser, or passed to the ML model.
- `profiles.json` stores only Argon2id hashes (PHC string format).
- No API endpoint returns the `password_hash` field.
- `experiment_label` (genuine/impostor annotation) is stored in the experiment log as metadata only and does not influence the authentication decision.

---

## Dissertation Result Artefacts and Reproduction

The archive includes the trained model and the saved result artefacts used in the dissertation, including `model/training_report.txt`, `model/evaluation_report.txt`, `model/feature_importance.csv`, and `model/roc_data.csv`. The submitted `data/profiles.json` and `data/experiment_log.csv` are sanitised/empty to avoid distributing live participant credentials/behavioural records. Therefore, the exact historical training and final evaluation cannot be regenerated from the sanitised archive alone.

After enrolling new users and collecting a new labelled experiment log, the same pipeline can be run with:

```bash
cd outputs
python model/train_model.py --source profiles
python model/evaluate.py
python model/evaluate.py --sweep
```

These commands will create new model/evaluation artefacts from the newly collected data and may overwrite the corresponding saved files. Use a working copy if the original dissertation artefacts must be preserved.

---

## Running Without Changing the Submitted Artefacts

- `python -m pytest tests/ -v` uses temporary files/mocks and does **not** modify the real profile, experiment-log or trained-model files.
- Starting `python app.py` does not normally modify an existing data file by itself.
- Enrolling a user writes to `outputs/data/profiles.json`.
- Authentication/Experiment Console use can append to `outputs/data/experiment_log.csv`.
- `python model/evaluate.py` rewrites `model/evaluation_report.txt`; `--sweep` also rewrites `model/roc_data.csv`.
- `python model/train_model.py` rewrites the trained model and associated training artefacts.

For demonstrations, work from a duplicate folder so the submission copy remains unchanged.

## Dissertation Context

This prototype was developed as part of an MSc FinTech dissertation investigating the feasibility of keystroke dynamics as a transparent, hardware-free second-factor authentication mechanism for online banking. The system is an academic proof-of-concept and is not intended for production deployment.
