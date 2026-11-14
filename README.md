# MBDS — More than a Hack 2027 (Phase 1)

Who has the ball, frame by frame, from automatic tracking: `H`, `A` or `D` for every frame of the
provided match. Code only: no match data and no data-derived files are in this repository.

**Technical document:** [MBDS.md](MBDS.md)

## What it does

Tracking → collective per-frame features → gradient-boosted H/A/D probabilities → Viterbi decoding
with switching costs → one label per frame. Trained and tuned **only on Metrica's three public
sample games** (other matches, event-derived labels), using degraded copies that reproduce the
challenge tracking's published figures (ball missing on 41% of graded frames, ~20.5 tracked
players, fragmented tracks).

Leave-one-game-out on degraded Metrica, organisers' scorer (details in [MBDS.md §5](MBDS.md)):

| system | spellF1 clean | spellF1 degraded | macroF1 degraded |
|---|---|---|---|
| organisers' baseline (smooth50) | 0.429 | 0.242 | 0.517 |
| our frame classifier alone | 0.488 | 0.258 | 0.863 |
| **ours (classifier + decoder)** | **0.762** | **0.680** | **0.866** |

These are Metrica scores, not the score on the provided match. For reference, the baseline scores
0.2433 on the full provided match.

## Layout

    <workspace>/
      MBDS/                         this repository
        make_submission.py          regenerates the submission
        evaluate_metrica.py         reproduces the Metrica results (public data only)
        ext_metrica.py              Metrica loader + event -> H/A/D labels
        possession/
          data.py                   loads Metrica and the provided match into one shape
          docs_degrade.py           degraded training copies (DATA.md figures only)
          features.py               per-frame features
          model.py                  classifier, Viterbi decoder, organisers' scorer
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
    python make_submission.py --team MBDS           # ~6 min -> ../submissions/MBDS.csv

Deterministic (every random choice is seeded): two runs give byte-identical files. The script ends
by running the organisers' `validate_submission.py` on the file and prints PASS / FAIL.

## Reproduce the Metrica results

    python evaluate_metrica.py                      # ~10 min, reads only the public Metrica games

## Data and licence

- Training data: **Metrica Sports sample data**, https://github.com/metrica-sports/sample-data
  (three anonymised matches with tracking and event data). Metrica asks users to use it responsibly
  and to acknowledge the source in public work, which we do here and in MBDS.md.
- The provided match data is not redistributable and is not included; `.gitignore` excludes data,
  caches and submissions.
