"""End-to-end: Metrica training data -> model -> challenge submission.

Every random choice is seeded, so the same inputs produce the same file byte for byte.
"""
import itertools

import numpy as np

from . import D, data, docs_degrade, model

METRICA_GAMES = (1, 2, 3)
# Switching costs of the decoder, chosen by leave-one-game-out on degraded Metrica
# (evaluate_metrica.py prints the grid). The score surface is flat around them.
LAM_TEAM = 15.0
LAM_DEAD = 50.0

# Degraded copies of every Metrica game (possession/docs_degrade.py, DATA.md figures only).
# DATA.md says the ball is missing on 41% of graded frames but not where, so three copies span it:
# dead ball invisible / missing anywhere / dead ball visible.
DEGRADED_KINDS = {"degraded": "dead_invisible", "degraded_all": "all", "degraded_live": "live"}
KINDS = ("clean",) + tuple(DEGRADED_KINDS)


def prepare(match):
    """Features in both team orientations plus labels, ready for model.fit / predict."""
    F, F_swap = model.feature_pair(match)
    out = {"name": match["name"], "F": F, "F_swap": F_swap, "period": match["period"]}
    if match["labels"] is not None:
        out["labels"] = match["labels"]
        out["labels_swap"] = data.swap_teams(match)["labels"]
        out["gt"] = data.gt_codes(match)
    return out


def metrica_sets(games=METRICA_GAMES, verbose=True):
    """{(game, kind): prepared set} for kind in KINDS. Seeds are fixed per game and kind."""
    sets = {}
    for g in games:
        m = data.load_metrica(g)
        sets[(g, "clean")] = prepare(m)
        for kind, where in DEGRADED_KINDS.items():
            seed = 100 * g + docs_degrade.WHERES.index(where)
            sets[(g, kind)] = prepare(docs_degrade.docs_copy(m, seed=seed, where=where))
        if verbose:
            print("metrica game %d ready" % g, flush=True)
    return sets


def cross_validate(sets, games=METRICA_GAMES, verbose=True):
    """Leave-one-game-out probabilities: {(game, kind): [n, 3]}."""
    oof = {}
    for g in games:
        train = [sets[(k, kind)] for k in games if k != g for kind in KINDS]
        clf = model.fit(train)
        for kind in KINDS:
            s = sets[(g, kind)]
            oof[(g, kind)] = model.predict_proba(clf, s["F"], s["F_swap"])
        if verbose:
            print("fold: held out game %d" % g, flush=True)
    return oof


def tune_decoder(sets, oof, games=METRICA_GAMES, lam_team=(8, 11, 15, 20),
                 lam_dead=(18, 26, 36, 50, 70, 100), kinds=tuple(DEGRADED_KINDS)):
    """spellF1 for every (lam_team, lam_dead) on every held-out game, averaged over the degraded
    kinds (pass kinds=('degraded',) for the old single-copy behaviour)."""
    if isinstance(kinds, str):
        kinds = (kinds,)
    rows = []
    for g in games:
        for lt, ld in itertools.product(lam_team, lam_dead):
            sc = [model.score(sets[(g, k)]["gt"], model.decode(oof[(g, k)], sets[(g, k)]["period"], lt, ld))
                  ["spell_f1_mean"] for k in kinds]
            rows.append({"game": g, "lam_team": lt, "lam_dead": ld, "spell_f1": float(np.mean(sc))})
    return rows


def train_final(sets):
    """The submission model: all three Metrica games, clean and every degraded copy."""
    return model.fit(list(sets.values()))


# Organisers' FAQ.md ("Which frames are actually graded?"): the second half is graded from video
# frame 110,808 but kicks off at 113,742; "those 2,934 frames are graded and all of them are D".
# The first half is graded from its kickoff. Published fact about the match timeline, not learned.
PRE_KICKOFF_DEAD = {"BAR-ATH": [(110808, 113741)]}


def predict_challenge(clf, lam_team=LAM_TEAM, lam_dead=LAM_DEAD):
    """Labels for all 199,319 frames of the challenge match (X outside the two halves, D on graded
    frames before a kickoff, as published by the organisers)."""
    match = data.load_challenge()
    s = prepare(match)
    proba = model.predict_proba(clf, s["F"], s["F_swap"])
    period = match["period"].copy()
    dead = np.zeros(len(period), bool)
    for first, last in PRE_KICKOFF_DEAD.get(match["name"], []):
        dead[first:last + 1] = True
    period[dead] = 0                                  # decode each half from its kickoff
    labels = model.decode(proba, period, lam_team, lam_dead)
    labels[dead] = D
    return labels, proba, match
