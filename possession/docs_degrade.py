"""Make a clean Metrica game look like the challenge tracking, using ONLY figures stated in the
organisers' DATA.md (nothing measured on the provided match):
  * ball missing on 41% of graded frames (DATA.md: "41% of the graded ones");
  * ~20.5 tracked objects per graded frame out of 22, occasionally more than 22 (duplicates);
  * slots appear, vanish and swap identity.
DATA.md does not say WHERE the ball is missing, so three copies span the plausible range:
  'dead_invisible'  dead ball invisible (as in Metrica), extra gaps only in live play
  'all'             ball held visible through dead ball, gaps anywhere (visibility says nothing)
  'live'            ball held visible through dead ball, all gaps in live play
Generic choices (made without looking at any data): ball gaps lognormal, median 1.5 s, 0.2-20 s;
player dropout in Markov runs of mean 2 s; 0.25 m position noise; up to 2 ghost slots per team;
slot order reshuffled on average every 30 s.
Works on the package's match dict (possession/data.py: pos [n,S,2], team [S], ball [n,2], labels, period). Seeded.
"""
import numpy as np
import pandas as pd

FPS = 25
D = 2                         # code for dead ball (H=0, A=1, D=2, X=3)
BALL_MISSING = 0.41           # DATA.md
TRACKED_PER_FRAME = 20.5      # DATA.md
WHERES = ("dead_invisible", "all", "live")


def _markov(rng, n, frac, mean_len):
    """Boolean runs with stationary share `frac` and mean run length `mean_len` frames."""
    b = 1.0 / mean_len
    a = frac * b / (1 - frac)
    u = rng.random(n)
    out = np.zeros(n, bool)
    s = False
    for t in range(n):
        s = (u[t] > b) if s else (u[t] < a)
        out[t] = s
    return out


def drop_players(pos, rng, frac=1 - TRACKED_PER_FRAME / 22, mean_len_s=2.0, noise_m=0.25):
    pos = pos.copy()
    for j in range(pos.shape[1]):
        pos[_markov(rng, len(pos), frac, mean_len_s * FPS), j] = np.nan
    return pos + rng.normal(0, noise_m, pos.shape).astype(np.float32)


def ball_gaps(match, rng, where, frac=BALL_MISSING, noise_m=0.25):
    ball = match["ball"].copy()
    graded = match["period"] > 0
    dead = match["labels"] == D
    if where in ("all", "live"):
        held = pd.DataFrame(ball).ffill().bfill().to_numpy(np.float32)
        ball[dead] = held[dead]
    live_only = where in ("dead_invisible", "live")
    pool = np.flatnonzero(graded & ~dead) if live_only else np.flatnonzero(graded)
    miss = np.isnan(ball[:, 0])
    while miss[graded].mean() < frac:
        s = int(rng.choice(pool))
        L = int(np.clip(rng.lognormal(np.log(1.5 * FPS), 1.0), 0.2 * FPS, 20 * FPS))
        seg = np.arange(s, min(len(ball), s + L))
        miss[seg[~dead[seg]] if live_only else seg] = True
    ball[miss] = np.nan
    return ball + rng.normal(0, noise_m, ball.shape).astype(np.float32)


def slot_noise(pos, team, rng, n_ghost=2, ghost_frac=0.04, ghost_len_s=2.0, reid_s=30):
    """Ghost slots duplicating a real player, then slot order reshuffled within each team."""
    n = len(pos)
    new_pos, new_team = [], []
    for t in (0, 1):
        T = pos[:, team == t]
        k = T.shape[1]
        G = np.full((n, n_ghost, 2), np.nan, np.float32)
        for g in range(n_ghost):
            on = _markov(rng, n, ghost_frac, ghost_len_s * FPS)
            pad = np.concatenate(([0], on.astype(np.int8), [0]))
            e = np.flatnonzero(np.diff(pad))
            for st, en in zip(e[0::2], e[1::2]):
                src = int(rng.integers(k))
                off = rng.normal(0, 0.8, 2) + np.cumsum(rng.normal(0, 0.05, (en - st, 2)), axis=0)
                G[st:en, g] = T[st:en, src] + off
        S = np.concatenate([T, G], axis=1)
        ev = np.flatnonzero(rng.random(n) < 1.0 / (reid_s * FPS))
        bounds = np.r_[0, ev, n]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b > a:
                S[a:b] = S[a:b, rng.permutation(S.shape[1])]
        new_pos.append(S)
        new_team.append(np.full(S.shape[1], t, np.int8))
    return np.concatenate(new_pos, 1).astype(np.float32), np.concatenate(new_team)


def docs_copy(match, seed, where):
    """One degraded copy of a labelled Metrica match."""
    assert where in WHERES
    rng = np.random.default_rng(seed)
    pos = drop_players(match["pos"], rng)
    ball = ball_gaps(match, rng, where)
    pos, team = slot_noise(pos, match["team"], rng)
    return {**match, "pos": pos, "team": team, "ball": ball.astype(np.float32),
            "name": "%s_docs_%s" % (match["name"], where)}
