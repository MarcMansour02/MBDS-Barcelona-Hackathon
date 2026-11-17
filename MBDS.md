# MBDS — Possession from automatic tracking

*More than a Hack 2027, Phase 1. Everything below was trained and tuned on public data from other
matches (Metrica Sports sample games 1–3, plus 7 public Bundesliga matches used only as a test set).
The provided match is read only to produce the submission.*

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

2. **Two frame models, averaged.**
    - *Tree classifier*: a histogram gradient-boosting model (scikit-learn) gives p(H), p(A), p(D)
      per frame from its features, whose rolling windows reach up to ±7.5 s.
    - *Sequence model*: a temporal convolutional network (TCN, PyTorch, ~0.24 M weights) reads the
      same 124 features as a sequence at 5 frames per second — 9 residual blocks of dilated 1-D
      convolutions (kernel 5, dilations 1–32), so every prediction sees up to about ±34 s of play. Three
      networks with different seeds are averaged (`possession/sequence.py`).

    The frame probabilities are the mean of the two models. Every match is featurised twice, as
    recorded and with home/away exchanged; both models train on both and predict both, so neither
    has a home bias.

3. **Possession decoder**: Viterbi over {H, A, D}. Each frame costs −log p(label); a change of team
    costs λ_team = 15 and entering or leaving dead ball costs λ_dead = 50. This is the step that
    matters for spellF1: the classifier alone invents several times too many possessions; decoding
    keeps its per-frame accuracy and turns it into possessions (§5).

**Why this design.** spellF1 rewards getting each possession as one interval. A per-frame model
plus a sequence decoder with an explicit cost per switch targets that directly, has two
interpretable parameters, and its score surface is flat around the chosen values (§5), so it is not
fragile. The two frame models are complementary: on held-out matches the trees are more accurate
when the ball is visible and right at a change of possession, the sequence model during long ball
blackouts (> 5 s, ~29% of frames) and stoppages, where seconds of earlier and later play tell who
has the ball. Their errors differ, so the average scores higher than either alone (0.700 against
0.680 and 0.683, §5).

**The part that matters most: training for where the ball goes missing.** The ball is missing on
41% of the graded frames of the provided match (DATA.md), but not *where*. In Metrica's clean
tracking the ball is missing almost only when play is dead, so a model trained on it learns "no
ball ⇒ dead ball". We measured that this collapses when the ball stays visible through stoppages,
and fixed it by training on degraded copies that span the possibilities (§3).

**How missing and noisy tracking is handled, in one place:**

- *Training data*: both models learn only from degraded copies plus the clean game — 41% of the
  ball removed in three different placements, player dropout to ~20.5 tracked, ghost duplicates,
  identity switches and position noise (§3). All headline results are measured on degraded copies.
- *Features*: collective per team, so no feature relies on a slot being one person; speeds above
  12 m/s are ignored as identity switches; the attack direction uses medians, robust to ghosts.
- *Ball gaps are described, not guessed*: whether the ball is seen, time since / until it is seen,
  gap length, and which team was nearest at the last and next sighting.
- *Sequence model*: a missing-value indicator for every feature, and up to ±34 s of context to
  bridge long blackouts — where it is most accurate relative to the trees.
- *Decoder*: a change of team must accumulate evidence worth λ_team, so short noise bursts, ghost
  tracks or a brief moment with the other team nearest the ball cannot flip possession.

## 2. Assets used

| asset | used? | how |
|---|---|---|
| tactical ATD (players + ball) | **yes** | the only input to the features |
| ATD metadata | **yes** | team of each track slot |
| Metrica public sample games 1–3 | **yes** | training and validation data (§3) |
| 7 public Bundesliga matches (IDSSE / Sportec Solutions open data, CC BY 4.0) | **yes, validation only** | a second, independent test set: different league, tracking provider and labelling (§5). Never used to train the submitted model |
| Smart Tagging | tried, **dropped** | set-piece tags as extra features (time to / since a restart tag, its team). Metrica has no Smart Tagging, so we simulated tags from Metrica's set-piece events with DATA.md's per-type counts. Gain +0.007 spellF1 with realistic tags (noise level), −0.006 with sloppier tags: not robust, not used |
| homography, videos, broadcast feed | no | — |

Other ideas tested and dropped because none gave a robust gain (the later ones were judged with the
adoption rule of §5: at least +0.01 on both test sets, no match worse than −0.02):
extra team-shape features (defensive lines, depth, width, pressure), physics features (time for
each team to reach the projected ball, ball-carrier likelihood), a decoder whose switching cost is
lowered where a transition detector fires, a minimum stoppage duration in the decoder (2 s; no
change, λ_dead already rules out very short stoppages), a learned per-situation combination of the
two frame models instead of the plain average (worse: −0.013 Metrica, −0.005 Bundesliga — three
games are too few to fit it), a recurrent sequence model (bidirectional LSTM) instead of or next to
the TCN (best on Metrica alone, 0.708, but weaker on the Bundesliga matches, 0.676; added to the
ensemble +0.005 Metrica / +0.001 Bundesliga), simulated slot duplication / reshuffling and
wrong-team tracks in training, and adding the Bundesliga matches to the training data (better on Bundesliga-style labels,
−0.035 on Metrica; we kept the training labels closest to the challenge's convention).

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

- Both frame models are trained on each game clean plus its three degraded copies, in both team
    orientations.

- **Validation only — 7 Bundesliga matches** (open data by Sportec Solutions / DFL, CC BY 4.0,
    obtained through kloppy's open-data loader; Bassek, Rein, Weber & Memmert (2025), "An integrated
    dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12, 195,
    doi:10.1038/s41597-025-04505-y): optical tracking at 25 fps with the provider's own
    per-frame ball-status and ball-possession flags, produced by the provider's operators — not by
    us. ~670 graded minutes, ~2,000 possessions after conversion. The flags change on every touch, so for the
    challenge's *controlled* possession we merge brief opposing touches: a change of team counts only
    if the new team keeps the ball for at least 2 s (threshold chosen so that the number of
    possessions per minute of live play matches Metrica's, 5.3 vs 5.5). They were degraded exactly
    like the Metrica test copies and used **only to test** candidate models trained on Metrica (§5);
    no submitted model was trained on them.

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

Match timeline (published by the organisers, not learned): frames outside the two graded halves
are `X`; the second half is graded from frame 110,808 but kicks off at 113,742 (FAQ.md), so those
2,934 frames are `D` and the decoder starts the half at the kickoff. Without this, players walking
out before the restart produced ~11 spurious possessions.

## 5. Results

**Leave-one-game-out on Metrica** (`python evaluate_metrica.py`; organisers' `possession_metrics.py`,
unchanged; spellF1, mean of the three held-out games; both frame models are retrained in every fold):

| system | clean | degraded: dead ball invisible | degraded: missing anywhere | degraded: dead ball visible | **mean degraded** | macroF1 (degraded) |
|---|---|---|---|---|---|---|
| organisers' baseline (smooth50) | 0.429 | 0.325 | 0.226 | 0.175 | 0.242 | 0.517 |
| tree classifier alone (argmax) | 0.488 | 0.266 | 0.247 | 0.260 | 0.258 | 0.863 |
| trees + decoder (our previous system) | 0.762 | 0.690 | 0.674 | 0.676 | 0.680 | 0.866 |
| sequence models + decoder | 0.754 | 0.710 | 0.659 | 0.678 | 0.683 | 0.867 |
| **trees + sequence models + decoder (submitted)** | **0.765** | **0.724** | **0.677** | **0.698** | **0.700** | **0.882** |

The tree classifier alone is accurate per frame (macroF1 0.86) but fragments possessions; the
decoder keeps that accuracy and multiplies spellF1 by 2.6. Averaging in the sequence model adds
+0.020 spellF1 and +0.016 macroF1 over trees + decoder. The baseline's 0.242 on degraded Metrica is
close to its published full-match score on the provided game (0.2433), a sign that the degraded
copies are about as hard as the real tracking. The submitted system predicts 2,356 possessions
against 3,153 on the degraded copies.

**Second test set: 7 public Bundesliga matches** (§3; never used for training). Models trained on
all three Metrica games, each match degraded in the same three ways, same scorer and decoder:

| system | Bundesliga, mean spellF1 | macroF1 | matches improved vs previous system |
|---|---|---|---|
| trees + decoder (previous system) | 0.672 | 0.897 | — |
| **trees + sequence models + decoder (submitted)** | **0.700** | **0.906** | **7 of 7** |

The gain holds on a different league, tracking provider and labelling convention, which is why we
adopted it. Our rule, fixed before testing any candidate: adopt a change only if it improves the
mean by at least 0.01 on **both** Metrica and the Bundesliga matches and no single match drops by
more than 0.02. (These two rows come from our experiment scripts with the same protocol; on Metrica
those scripts give 0.668 → 0.699, in line with the table above.)

Decoder costs for the submitted system (spellF1, mean of degraded copies and held-out games). The
costs 15 / 50 were chosen for the tree-only system and kept; the surface is flat (0.688–0.709) and
we did not re-tune them on the same games we report:

| λ_team \ λ_dead | 18 | 26 | 36 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|
| 8 | 0.708 | 0.709 | 0.709 | 0.707 | 0.707 | 0.704 |
| 11 | 0.702 | 0.702 | 0.702 | 0.701 | 0.700 | 0.697 |
| **15** | 0.701 | 0.701 | 0.702 | **0.700** | 0.699 | 0.696 |
| 20 | 0.693 | 0.693 | 0.693 | 0.692 | 0.691 | 0.688 |

**Robustness check across simulators** (done for the tree-only system, which chose the training
copies both models now use). A model scores best on test data degraded by the same
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

**Public 5-minute window** (used only to check that the pipeline is correct — teams not swapped,
frames aligned): spellF1 0.638, macroF1 0.773, 15 predicted possessions against 17 (our previous
system: 0.674 / 0.768 / 18; `baseline_smooth50`: 0.326 / 0.621 / 26). With 17 possessions this is a
sanity check, not an estimate: retraining the same tree model with a different subsample of
training frames moves its public spellF1 between 0.61 and 0.67. Choices were made on the
held-out matches above only.

## 6. Limitations

- Training labels come from Metrica's event data; the challenge truth comes from corrected tracking.
  Systematic differences in where possessions start and end transfer as a bias we cannot measure.
- The degradation approximates the challenge noise from the figures published in DATA.md only.
  Where the ball goes missing is unknown, so we train on a range of cases; if the real pattern lies
  outside that range, the score will be lower.
- The model predicts ~25% fewer possessions than there are on degraded Metrica: very short
  possessions (under ~2 s, often one touch) are frequently merged into their neighbours.
- Three training games. The sequence model in particular would benefit from more: with three games
  it fits its training data almost perfectly (final training loss ~0.07), a sign that it partly
  memorises them.

## 7. What we would do with more time

- **More training matches.** Three games is the main limit, for the sequence model even more than
  for the trees. Larger public datasets exist (e.g. the 7 Bundesliga matches we used for testing,
  or full-tournament tracking releases), but each provider defines possession differently: adding
  the Bundesliga matches as they are lowered the Metrica score by 0.035. Aligning every source's
  labels to the challenge's *controlled* possession, then weighting the sources, is the next step.
- **Short possessions.** Most of the remaining error is in possessions under ~5 s. A touch detector
  (ball direction / speed changes next to a player) feeding the decoder is the most promising lever.
- **Smart Tagging** with real alignment data: the restart tags carry information the model lacks,
  but without tagged training matches we could not show a robust gain.
- **Ball re-detection** from the video to reduce the 41% of frames without a ball.

## 8. Reproduction

    pip install -r requirements.txt
    python make_submission.py --team MBDS
    python make_submission.py --team MBDS --retrain
    python make_submission.py --team MBDS --trees-only
    python evaluate_metrica.py

- `make_submission.py` (~8 min): trees + the shipped networks (`models/sequence_metrica.pt`);
  writes `../submissions/MBDS.csv` and runs the organisers' validator on it.
- `--retrain` (~30 min on a CPU): retrains the three networks from the Metrica games.
- `--trees-only`: our previous, tree-only system.
- `evaluate_metrica.py` (~1.5 h): the Metrica results of §5, public data only.

Expected layout (any location can be overridden with environment variables — see
`possession/paths.py`):

    <workspace>/
      MBDS/                      this repository
      more-than-a-hack-2027/     organisers' repository (MTAH_REPO)
      sample-data/data/          Metrica public sample data (METRICA_DIR)
      data/tactical/             provided match data, local only (MTAH_DATA_DIR)

Deterministic: every random choice is seeded (degradation, trees, network initialisation and
batches). The networks train on the CPU; floating-point results can differ in the last bits across
hardware and thread counts, so the trained networks are shipped in `models/sequence_metrica.pt` and
the default command reproduces the submitted file exactly.
The script ends by running the organisers' `validate_submission.py` on the file.

## 9. Use of AI assistance

AI coding assistants were used during development: to write and review code, design and run the
robustness experiments on the public matches, and draft this document. All modelling decisions were
checked by the team. The submitted model was trained only on Metrica's public games; candidates were
compared on held-out Metrica games and the public Bundesliga matches.
