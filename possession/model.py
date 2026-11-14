"""Frame classifier + possession decoder + scoring helpers."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import H, A, D, X, LETTERS, paths
from . import features as feats
from .data import swap_teams, gt_codes

SEED = 0


# --------------------------------------------------------------------------- features, both ways

def feature_pair(match):
    """Features as recorded and with home/away exchanged. The swapped copy doubles the training
    data and, at prediction time, lets us average both views so the model has no home bias."""
    return feats.compute(match), feats.compute(swap_teams(match))


# ------------------------------------------------------------------------------- frame classifier

def training_rows(items, step=4):
    """items: list of (features DataFrame, labels). Every `step`-th labelled graded frame."""
    Xs, ys = [], []
    for k, (F, y, period) in enumerate(items):
        keep = np.flatnonzero((period > 0) & (y != X))
        keep = keep[(keep + k) % step == 0]
        Xs.append(F.iloc[keep])
        ys.append(y[keep])
    import pandas as pd
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def make_classifier():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=SEED)


def fit(train_sets, step=4):
    """train_sets: list of dicts with F, F_swap, labels, labels_swap, period."""
    items = []
    for s in train_sets:
        items.append((s["F"], s["labels"], s["period"]))
        items.append((s["F_swap"], s["labels_swap"], s["period"]))
    Xtr, ytr = training_rows(items, step)
    clf = make_classifier()
    clf.fit(Xtr, ytr)
    return clf


def predict_proba(clf, F, F_swap):
    """[n, 3] probabilities for H, A, D, averaged over the two team orientations."""
    p = clf.predict_proba(F)
    q = clf.predict_proba(F_swap)[:, [A, H, D]]
    return (p + q) / 2


# ------------------------------------------------------------------------------------ decoder

def viterbi(logp, lam_team, lam_dead):
    """Most likely H/A/D sequence given per-frame log-probabilities and switching costs.

    Changing team costs lam_team, entering or leaving dead ball costs lam_dead. This is what turns
    a flickering frame classifier into possessions: a switch only happens when the evidence for it
    accumulated over many frames outweighs the cost.
    """
    n = len(logp)
    C = [[0.0, lam_team, lam_dead], [lam_team, 0.0, lam_dead], [lam_dead, lam_dead, 0.0]]
    back = np.zeros((n, 3), np.int8)
    lp = logp.tolist()
    d0, d1, d2 = lp[0]
    for t in range(1, n):
        a0, a1, a2 = lp[t]
        # state 0 (H)
        c0 = (d0, d1 - C[1][0], d2 - C[2][0])
        b0 = 0 if c0[0] >= c0[1] and c0[0] >= c0[2] else (1 if c0[1] >= c0[2] else 2)
        c1 = (d0 - C[0][1], d1, d2 - C[2][1])
        b1 = 1 if c1[1] >= c1[0] and c1[1] >= c1[2] else (0 if c1[0] >= c1[2] else 2)
        c2 = (d0 - C[0][2], d1 - C[1][2], d2)
        b2 = 2 if c2[2] >= c2[0] and c2[2] >= c2[1] else (0 if c2[0] >= c2[1] else 1)
        back[t, 0], back[t, 1], back[t, 2] = b0, b1, b2
        d0, d1, d2 = c0[b0] + a0, c1[b1] + a1, c2[b2] + a2
    path = np.empty(n, np.int8)
    last = (d0, d1, d2)
    path[-1] = int(np.argmax(last))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def period_runs(period):
    """[(start, stop)] contiguous stretches of one non-zero period."""
    out, n, i = [], len(period), 0
    while i < n:
        if period[i] == 0:
            i += 1
            continue
        j = i
        while j < n and period[j] == period[i]:
            j += 1
        out.append((i, j))
        i = j
    return out


def decode(proba, period, lam_team=8.0, lam_dead=8.0, floor=1e-4):
    """Labels for every frame: Viterbi path inside each period, X outside (never scored)."""
    labels = np.full(len(proba), X, np.int8)
    logp = np.log(np.clip(proba, floor, 1.0))
    for s, e in period_runs(period):
        labels[s:e] = viterbi(logp[s:e], lam_team, lam_dead)
    return labels


def argmax_labels(proba, period):
    return np.where(period > 0, proba.argmax(1), X).astype(np.int8)


# ----------------------------------------------------------------------------------- baseline

def baseline_labels(F, period, window=50):
    """The organisers' baseline (nearest player to the ball + majority vote), reimplemented on
    our features so it can also be run on Metrica."""
    paths.organisers_on_path()
    from starter.baseline import smooth
    near = F["near"].to_numpy()
    codes = np.where(near == 1, 0, np.where(near == -1, 1, -1)).astype(np.int8)
    letters = smooth(codes, window)
    out = np.array([LETTERS.index(c) for c in "".join(letters)], np.int8)
    return np.where(period > 0, out, X).astype(np.int8)


# ------------------------------------------------------------------------------------ scoring

KEYS = ["spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "mean_spell_duration_s",
        "gt_mean_spell_duration_s", "alive_dead_f1", "team_acc_gt_alive"]


def score(gt, pred):
    """Organisers' scorer, unchanged. gt and pred are code arrays (H=0 A=1 D=2 X=3)."""
    paths.organisers_on_path()
    import possession_metrics as pm
    r = pm.evaluate(np.asarray(gt, np.int8), np.asarray(pred, np.int8))
    return {k: r[k] for k in KEYS}


def write_submission(path, labels):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, LETTERS[c]) for i, c in enumerate(labels))
    return path
