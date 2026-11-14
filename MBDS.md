# MBDS — Possession from automatic tracking

*More than a Hack 2027, Phase 1. Everything below was trained and tuned on public data from other
matches (Metrica Sports sample games 1–3). The provided match is read only to produce the submission.*

## 1. Approach

Three stages, run per half:

1. **Collective per-frame features** from the tracking (124 features). Nothing assumes a track slot
    is one person, so fragmented and identity-switching tracklets are harmless; player speeds above
    12 m/s are treated as identity switches and ignored. Groups: ball state (seen, position,
    distance outside the pitch, speed, time since / until seen, gap length); contact (nearest home
    and away player, counts within 3 m / 10 m, which team was nearest at the last and next frame the
    ball was seen); team shape (share of each team goal-side of the ball, centroids, spreads, players
    tracked); motion (team speed and forward velocity); and centred, past-only and future-only
    rolling means of the key signals. Looking ahead is legitimate: labels are produced offline.

    Features are expressed with the home team attacking towards +x. The attack direction is found
    per match from goalkeeper-like positions — each team's rearmost player stays near the goal it
    defends, whoever has the ball — pooled over both halves, since teams change ends. On every
    Metrica half and on both halves of the provided match it is decided by a 55–67 m margin (a rule
    based on the home team's mean position was right by only ~3 m on some halves).

2. **Frame classifier**: a histogram gradient-boosting model (scikit-learn) gives p(H), p(A), p(D)
    per frame. Every match is featurised twice, as recorded and with home/away exchanged; training
    uses both and prediction averages both, so the model has no home bias.

3. **Possession decoder**: Viterbi over {H, A, D}. Each frame costs −log p(label); a change of team
    costs λ_team = 15 and entering or leaving dead ball costs λ_dead = 50. This is the step that
    matters for spellF1: the classifier alone invents several times too many possessions; decoding
    keeps its per-frame accuracy and turns it into possessions (§5).

**Why this design.** spellF1 rewards getting each possession as one interval. A per-frame model
plus a sequence decoder with an explicit cost per switch targets that directly, has two
interpretable parameters, and its score surface is flat around the chosen values (§5), so it is not
fragile.

**The part that matters most: training for where the ball goes missing.** The ball is missing on
41% of the graded frames of the provided match (DATA.md), but not *where*. In Metrica's clean
tracking the ball is missing almost only when play is dead, so a model trained on it learns "no
ball ⇒ dead ball". We measured that this collapses when the ball stays visible through stoppages,
and fixed it by training on degraded copies that span the possibilities (§3).

## 2. Assets used

| asset | used? | how |
|---|---|---|
| tactical ATD (players + ball) | **yes** | the only input to the features |
| ATD metadata | **yes** | team of each track slot |
| Metrica public sample games 1–3 | **yes** | training and validation data (§3) |
| Smart Tagging | tried, **dropped** | set-piece tags as extra features (time to / since a restart tag, its team). Metrica has no Smart Tagging, so we simulated tags from Metrica's set-piece events with DATA.md's per-type counts. Gain +0.007 spellF1 with realistic tags (noise level), −0.006 with sloppier tags: not robust, not used |
| homography, videos, broadcast feed | no | — |

Other ideas tested on Metrica and dropped for lack of a robust gain (all within ±0.01 spellF1):
extra team-shape features (defensive lines, depth, width, pressure), a decoder whose switching cost
is lowered where a transition detector fires, and simulated slot duplication / reshuffling in
training (the features are already insensitive to slot identity).

## 3. Training data

- **Source:** Metrica Sports public sample data (github.com/metrica-sports/sample-data), three
    anonymised full matches with tracking and synchronised event data. Other matches, not BAR–ATH.
- **Who labelled it:** Metrica produced the event data. We converted events to per-frame H/A/D
    labels with fixed rules (`ext_metrica.py`, §4); no person labelled any frame for us.
- **How much:** ~430,000 graded frames, ~1,050 possessions.
- **Degraded copies** (`possession/docs_degrade.py`, seeded). Every target figure comes from the
    organisers' DATA.md: the ball is missing on **41%** of graded frames; **~20.5** objects are tracked
    per frame out of 22, occasionally more when the tracker duplicates someone; slots appear, vanish
    and swap identity. Since DATA.md does not say *where* the ball goes missing, each game is used in
    three degraded copies, all at 41% missing ball:
    1. dead ball invisible (as in Metrica), the extra gaps only in live play;
    2. ball held visible through dead ball, gaps anywhere (visibility says nothing about the state);
    3. ball held visible through dead ball, all gaps in live play.

    All three also get player dropout (Markov runs, mean 2 s, down to ~20.5 tracked), up to two ghost
    slots per team duplicating a real player, slot order reshuffled within each team (identity
    switches) and 25 cm position noise. Gap lengths (lognormal, median 1.5 s, 0.2–20 s) and the other
    shape choices are generic and were fixed before testing.
- The classifier is trained on each game clean plus its three degraded copies, in both team
    orientations.
- **No labels of any kind from the provided match** were used to train, tune or select anything.

## 4. Possession rules implemented

Training labels (from Metrica events, `ext_metrica.py`), following the challenge's definition that
possession changes only when the other team gains *controlled* possession:

- PASS, SHOT, CARRY, RECOVERY or SET PIECE by team T → T in possession from that event.
- BALL LOST, CHALLENGE (won or lost) → **no change**: a loose or contested ball stays with the team
  that last had it until the other team controls it (a RECOVERY or its own action).
- BALL OUT, goal, offside, hand ball → dead from the end of the event; foul → dead from its start;
  dead ends at the next restart, which also gives possession to the restarting team.

At inference, "controlled" possession is enforced by the decoder: a change of team has to be
supported by enough frames of evidence to pay λ_team, so a single touch, a deflection or a lost
aerial duel does not flip possession.

## 5. Results

**Leave-one-game-out on Metrica** (`python evaluate_metrica.py`; organisers' `possession_metrics.py`,
unchanged; spellF1, mean of the three held-out games):

| system | clean | degraded: dead ball invisible | degraded: missing anywhere | degraded: dead ball visible | **mean degraded** | macroF1 (degraded) |
|---|---|---|---|---|---|---|
| organisers' baseline (smooth50) | 0.429 | 0.325 | 0.226 | 0.175 | 0.242 | 0.517 |
| frame classifier alone (argmax) | 0.488 | 0.266 | 0.247 | 0.260 | 0.258 | 0.863 |
| **classifier + decoder (submitted)** | **0.762** | **0.690** | **0.674** | **0.676** | **0.680** | **0.866** |

The classifier alone is accurate per frame (macroF1 0.86) but fragments possessions; the decoder
keeps that accuracy and multiplies spellF1 by 2.6. The baseline's 0.242 on degraded Metrica is close
to its published full-match score on the provided game (0.2433), a sign that the degraded copies are
about as hard as the real tracking. The submitted system predicts 2,565 possessions against 3,153 on
the degraded copies.

Decoder costs (spellF1, mean of degraded copies and held-out games). The chosen point (15 / 50) is
the best of the grid and the surface is flat (0.671–0.680):

| λ_team \ λ_dead | 18 | 26 | 36 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|
| 8 | 0.673 | 0.673 | 0.676 | 0.676 | 0.676 | 0.676 |
| 11 | 0.675 | 0.675 | 0.678 | 0.678 | 0.677 | 0.678 |
| **15** | 0.677 | 0.678 | 0.680 | **0.680** | 0.679 | 0.679 |
| 20 | 0.671 | 0.671 | 0.673 | 0.674 | 0.672 | 0.672 |

**Robustness check across simulators.** A model scores best on test data degraded by the same
simulator it was trained on. So every candidate was also tested on copies made by a second,
independent degradation simulator (same 41% missing ball and ~20.5 tracked players, different gap,
dropout and duplication mechanics), in the same leave-one-game-out protocol:

| training | test simulator A | test simulator B |
|---|---|---|
| clean + one Metrica-like degraded copy (ball missing mostly in dead ball) | 0.584 | 0.566 |
| **clean + three degraded copies (submitted)** | **0.654** | **0.672** |

The gain is consistent on both simulators (+0.07 and +0.11). The single-copy model is the best in
the Metrica-like scenario (ball invisible during stoppages: 0.71–0.72 against 0.68–0.70) but drops to
0.46–0.56 when the ball stays visible through stoppages; the submitted model stays at 0.64–0.66
there. We chose the model that holds up across scenarios, since DATA.md reports that the ball goes
missing "disproportionately on the frames that matter" (crowds, aerial balls), i.e. in live play.

**Public 5-minute window** (used once, only to check that the pipeline is correct — teams not
swapped, frames aligned): spellF1 0.674, macroF1 0.768, 18 predicted possessions against 17, against
0.326 / 0.621 / 26 for `baseline_smooth50`. With 17 possessions this is a sanity check, not an
estimate of the full-match score.

## 6. Limitations

- Training labels come from Metrica's event data; the challenge truth comes from corrected tracking.
  Systematic differences in where possessions start and end transfer as a bias we cannot measure.
- The degradation approximates the challenge noise from the figures published in DATA.md only.
  Where the ball goes missing is unknown, so we train on a range of cases; if the real pattern lies
  outside that range, the score will be lower.
- The model predicts ~19% fewer possessions than there are on degraded Metrica: very short
  possessions (under ~2 s, often one touch) are frequently merged into their neighbours.
- Three training games.

## 7. What we would do with more time

- **More training matches.** Three games is the main limit on the classifier; any public tracking
  with event data would help, especially with automatic (uncorrected) tracking.
- **Short possessions.** Most of the remaining error is in possessions under ~5 s. A touch detector
  (ball direction / speed changes next to a player) feeding the decoder is the most promising lever.
- **Smart Tagging** with real alignment data: the restart tags carry information the model lacks,
  but without tagged training matches we could not show a robust gain.
- **Ball re-detection** from the video to reduce the 41% of frames without a ball.

## 8. Reproduction

    pip install -r requirements.txt
    python make_submission.py --team MBDS      # ~6 min; writes ../submissions/MBDS.csv and validates it
    python evaluate_metrica.py                 # ~10 min; the Metrica results of §5 (public data only)

Expected layout (any location can be overridden with environment variables — see
`possession/paths.py`):

    <workspace>/
      MBDS/                      this repository
      more-than-a-hack-2027/     organisers' repository (MTAH_REPO)
      sample-data/data/          Metrica public sample data (METRICA_DIR)
      data/tactical/             provided match data, local only (MTAH_DATA_DIR)

Deterministic: every random choice is seeded and two runs give byte-identical files (checked).
The script ends by running the organisers' `validate_submission.py` on the file.

## 9. Use of AI assistance

AI coding assistants were used during development: to write and review code, design and run the
robustness experiments on Metrica, and draft this document. All modelling decisions were checked by
the team. The model was trained and tuned only on Metrica's public games.
