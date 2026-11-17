# MBDS — More than a Hack 2027 (Phase 1)

Who has the ball, frame by frame, from automatic tracking: `H`, `A` or `D` for every frame of the
provided match. Code and trained model weights only (trained on public Metrica data): no match data
and nothing derived from it is in this repository.

**Technical document:** [MBDS.md](MBDS.md)

## What it does

Tracking → collective per-frame features → H/A/D probabilities from two models (gradient-boosted
trees, and a temporal convolutional network that reads up to ±34 s of play), averaged → Viterbi
decoding with switching costs → one label per frame. Trained **only on Metrica's three public sample
games** (other matches, event-derived labels), using degraded copies that reproduce the challenge
tracking's published figures (ball missing on 41% of graded frames, ~20.5 tracked players,
fragmented tracks); also tested on 7 public Bundesliga matches never used for training.

Leave-one-game-out on degraded Metrica, organisers' scorer (details in [MBDS.md §5](MBDS.md)):

| system | spellF1 clean | spellF1 degraded | macroF1 degraded |
|---|---|---|---|
| organisers' baseline (smooth50) | 0.429 | 0.242 | 0.517 |
| trees + decoder (our previous system) | 0.762 | 0.680 | 0.866 |
| **ours (trees + sequence models + decoder)** | **0.765** | **0.700** | **0.882** |

On 7 public Bundesliga matches never used for training: 0.672 → **0.700** (7 of 7 matches improved).
These are scores on other matches, not on the provided match. For reference, the baseline scores
0.2433 on the full provided match.

## Layout

    <workspace>/
      MBDS/                         this repository
        make_submission.py          regenerates the submission
        evaluate_metrica.py         reproduces the Metrica results (public data only)
        models/sequence_metrica.pt  the trained sequence models (weights only, no data)
        ext_metrica.py              Metrica loader + event -> H/A/D labels
        possession/
          data.py                   loads Metrica and the provided match into one shape
          docs_degrade.py           degraded training copies (DATA.md figures only)
          features.py               per-frame features
          model.py                  tree classifier, Viterbi decoder, organisers' scorer
          sequence.py               sequence model (temporal convolutional network)
          pipeline.py               end-to-end training and prediction
          paths.py                  where the data lives (overridable)
      more-than-a-hack-2027/        organisers' repository: games.py, scorer, validator
      sample-data/data/             Metrica public sample data (github.com/metrica-sports/sample-data)
      data/tactical/                provided match data (LOCAL ONLY, never commit)
      cache/, submissions/          created automatically, outside the repository

Folder names are detected automatically; to use other locations set `MTAH_REPO`, `METRICA_DIR`
and `MTAH_DATA_DIR` (folder containing `tactical/`).

## Reproduce the submission

    pip install -r requirements.txt                 # Python 3.13
    python make_submission.py --team MBDS               # ~8 min -> ../submissions/MBDS.csv
    python make_submission.py --team MBDS --retrain     # ~30 min, also retrains the networks (CPU)

Every random choice is seeded. The default command loads the shipped network weights
(`models/sequence_metrica.pt`) and reproduces the submitted file exactly; `--retrain` trains them
again from the Metrica games (CPU floating point can differ in the last bits across machines). The
script ends by running the organisers' `validate_submission.py` on the file and prints PASS / FAIL.

## Reproduce the Metrica results

    python evaluate_metrica.py                      # ~1.5 h on a CPU, reads only the public Metrica games

## Data and licence

- Training data: **Metrica Sports sample data**, https://github.com/metrica-sports/sample-data
  (three anonymised matches with tracking and event data). Metrica asks users to use it responsibly
  and to acknowledge the source in public work, which we do here and in MBDS.md.
- Test data (not needed to run anything here): 7 Bundesliga matches, open data by Sportec Solutions /
  DFL, CC BY 4.0 — Bassek, Rein, Weber & Memmert (2025), *Scientific Data* 12, 195,
  doi:10.1038/s41597-025-04505-y.
- The provided match data is not redistributable and is not included; `.gitignore` excludes data,
  caches and submissions.
