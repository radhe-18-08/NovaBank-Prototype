# NOVA Bank Submission Notes

This archive is a runnable research-prototype submission. The scientific algorithms, thresholds, model weights, trained model binary and saved dissertation result values have not been changed during final clean-up.

Important distinctions:
- The saved Random Forest model contains six class labels: five final study participants plus one development account named `Test`.
- The final genuine/impostor evaluation reported in the dissertation contains only the five study participants.
- `outputs/data/profiles.json` and `outputs/data/experiment_log.csv` are sanitised/empty in this submission copy. Historical result artefacts are retained in `outputs/model/`. Exact historical retraining/re-evaluation therefore requires the original research data and should not be claimed from this sanitised archive alone.
- Run demonstrations from a duplicate folder because enrolment, experiment logging, evaluation and model training can overwrite data/result artefacts.
