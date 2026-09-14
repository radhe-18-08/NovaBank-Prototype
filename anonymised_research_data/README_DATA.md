# NOVA Bank Anonymised Research Data

This folder contains an anonymised copy of the data needed to inspect the final controlled evaluation without exposing participant names or password credentials.

- Five participant identifiers are represented only as P01-P05.
- The mapping from participant names to these IDs is deliberately not included.
- Password hashes, plaintext passwords, enrolment timestamps, experiment event IDs and experiment timestamps are excluded.
- Keystroke profile timing/features, experimental ground-truth labels, behavioural scores and decisions are preserved.
- Numerical timing and score values were not altered during anonymisation.

`participant_profiles_anonymised.json` contains the five enrolled behavioural profiles with credential fields removed.
`final_experiment_anonymised.csv` contains the 46 final controlled experiment attempts. One impostor attempt failed credential verification, leaving 45 behaviourally scored attempts (25 genuine and 20 credential-valid impostor attempts).

The dissertation's binary comparison uses threshold 0.65. The operational ALLOW/VERIFY/BLOCK thresholds in the application are separate (ALLOW >= 0.55, VERIFY >= 0.28 and < 0.55, BLOCK < 0.28).

This anonymised research dataset is for academic verification of the submitted prototype results; it is not production banking data.
