"""Per-frame features. Collective only: nothing assumes a slot is one person.

Groups
  ball      seen?, position in home-attack coordinates, distance outside the pitch, speed,
            time since / until the ball is seen, length of the current gap
  contact   nearest home / away player distance, second nearest, players within 3 m and 10 m,
            which team is nearest -- now, at the last frame the ball was seen, and at the next one
  shape     players goal-side of the ball per team (the defending team sits behind it),
            team centroids and spreads, number of players tracked
  motion    team mean speed and mean forward velocity (the team in possession moves forward)
  context   centred, past-only and future-only rolling means of the key signals. The labels are
            produced offline for the whole match, so looking ahead is legitimate.
"""
import numpy as np
import pandas as pd

FPS = 25
MAX_SPEED = 12.0  # m/s; anything faster is an identity switch, not running


def _nanmean_axis1(a, mask):
    cnt = mask.sum(1)
    s = np.where(mask, a, 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / cnt, np.nan)


def _nanstd_axis1(a, mask):
    m = _nanmean_axis1(a, mask)
    cnt = mask.sum(1)
    v = np.where(mask, (a - m[:, None]) ** 2, 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 1, np.sqrt(v / np.maximum(cnt, 1)), np.nan)


def roll_mean(x, lo, hi):
    """Mean of non-NaN values of x over [t+lo, t+hi] (inclusive), NaN if none."""
    x = np.asarray(x, np.float64)
    ok = ~np.isnan(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cc = np.concatenate(([0], np.cumsum(ok)))
    n = len(x)
    i = np.arange(n)
    a = np.clip(i + lo, 0, n)
    b = np.clip(i + hi + 1, 0, n)
    cnt = cc[b] - cc[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs[b] - cs[a]) / np.maximum(cnt, 1), np.nan)


def _last_index(mask):
    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)


def _next_index(mask):
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def _defends_left_evidence(match, f):
    """> 0 when, in frames f, home defends the x = 0 goal. Each team's rearmost player (normally
    the goalkeeper) stays near the goal it defends, whoever dominates the ball, so compare how close
    each team's extreme players sit to each goal line (medians over frames: robust to ghosts)."""
    L = match["pitch"][0]
    x = match["pos"][f][:, :, 0]
    out = 0.0
    for t, sign in ((0, 1.0), (1, -1.0)):
        xt = x[:, match["team"] == t]
        xt = xt[~np.isnan(xt).all(1)]                     # frames with this team tracked
        lo = np.median(np.nanmin(xt, axis=1))             # distance of the leftmost player to x = 0
        hi = L - np.median(np.nanmax(xt, axis=1))         # distance of the rightmost player to x = L
        out += sign * (hi - lo)                           # home: near left goal -> +; away mirrored
    return out


def attack_direction(match):
    """+1 where home attacks towards x = length, -1 otherwise, per frame.

    Decided from goalkeeper-like positions (see _defends_left_evidence) pooled over all periods,
    using the fact that teams change ends at half-time. The old rule (home team's mean x below
    the midline) is fragile when the home side dominates the ball: on the challenge match it was
    right by only 3.5 m in the first half."""
    dirs = np.ones(len(match["period"]), np.int8)
    periods = [int(p) for p in np.unique(match["period"]) if p > 0]
    if not periods:
        return dirs
    first = periods[0]
    score = sum(_defends_left_evidence(match, match["period"] == p) * (1 if (p - first) % 2 == 0 else -1)
                for p in periods)
    d_first = 1 if score >= 0 else -1
    for p in periods:
        dirs[match["period"] == p] = d_first if (p - first) % 2 == 0 else -d_first
    # frames outside periods keep +1 (never graded, irrelevant for labels)
    return dirs


def compute(match):
    L, W = match["pitch"]
    pos, ball, team = match["pos"], match["ball"], match["team"]
    n = len(ball)
    d = attack_direction(match).astype(np.float32)

    # home-attack coordinates: home always attacks towards +x
    xa = np.where(d[:, None] > 0, pos[:, :, 0], L - pos[:, :, 0])
    bxa = np.where(d > 0, ball[:, 0], L - ball[:, 0])
    tracked = ~np.isnan(pos[:, :, 0])
    home, away = (team == 0), (team == 1)
    seen = ~np.isnan(ball[:, 0])

    f = {}
    # ---------------------------------------------------------------- ball
    f["seen"] = seen.astype(np.float32)
    f["ball_x"] = bxa / L
    f["ball_y_off"] = np.abs(ball[:, 1] - W / 2) / (W / 2)
    out_x = np.maximum(0, np.maximum(-ball[:, 0], ball[:, 0] - L))
    out_y = np.maximum(0, np.maximum(-ball[:, 1], ball[:, 1] - W))
    f["ball_out"] = np.where(seen, np.hypot(out_x, out_y), np.nan)
    f["ball_to_goalline"] = np.minimum(ball[:, 0], L - ball[:, 0])
    f["ball_to_touchline"] = np.minimum(ball[:, 1], W - ball[:, 1])
    step = np.full(n, np.nan, np.float32)
    step[2:-2] = np.hypot(*(ball[4:] - ball[:-4]).T) * FPS / 4
    bspeed = pd.Series(np.clip(step, 0, 40)).rolling(5, center=True, min_periods=1).median().to_numpy()
    f["ball_speed"] = bspeed
    last, nxt = _last_index(seen), _next_index(seen)
    i = np.arange(n)
    f["t_since_seen"] = np.where(last >= 0, np.minimum((i - last) / FPS, 60), 60)
    f["t_to_seen"] = np.where(nxt < n, np.minimum((nxt - i) / FPS, 60), 60)
    f["gap_len"] = np.where(seen, 0, np.minimum(f["t_since_seen"] + f["t_to_seen"], 120))

    # ---------------------------------------------------------------- contact
    dist = np.hypot(pos[:, :, 0] - ball[:, None, 0], pos[:, :, 1] - ball[:, None, 1])
    dist = np.where(tracked, dist, np.inf)
    dh = np.sort(dist[:, home], axis=1)
    da = np.sort(dist[:, away], axis=1)
    inf2nan = lambda a: np.where(np.isfinite(a), a, np.nan)
    f["dmin_h"], f["dmin_a"] = inf2nan(dh[:, 0]), inf2nan(da[:, 0])
    f["d2_h"], f["d2_a"] = inf2nan(dh[:, 1]), inf2nan(da[:, 1])
    f["dmin_diff"] = np.clip(f["dmin_h"] - f["dmin_a"], -30, 30)
    for r in (3, 10):
        nh = (dist[:, home] < r).sum(1).astype(np.float32)
        na = (dist[:, away] < r).sum(1).astype(np.float32)
        f["n%d_h" % r] = np.where(seen, nh, np.nan)
        f["n%d_a" % r] = np.where(seen, na, np.nan)
    near = np.where(np.isnan(f["dmin_diff"]), np.nan, np.sign(-f["dmin_diff"]))  # +1 home nearest
    f["near"] = near
    ok = ~np.isnan(near)
    ln, nn = _last_index(ok), _next_index(ok)
    f["near_last"] = np.where(ln >= 0, near[np.maximum(ln, 0)], np.nan)
    f["near_next"] = np.where(nn < n, near[np.minimum(nn, n - 1)], np.nan)
    f["diff_last"] = np.where(ln >= 0, f["dmin_diff"][np.maximum(ln, 0)], np.nan)
    f["diff_next"] = np.where(nn < n, f["dmin_diff"][np.minimum(nn, n - 1)], np.nan)

    # ---------------------------------------------------------------- shape
    bx_ff = bxa.copy()
    bx_ff[~seen] = np.nan
    bx_ff = pd.Series(bx_ff).ffill().bfill().to_numpy()
    nh_t = tracked[:, home].sum(1).astype(np.float32)
    na_t = tracked[:, away].sum(1).astype(np.float32)
    behind_h = ((xa[:, home] < bx_ff[:, None]) & tracked[:, home]).sum(1)
    behind_a = ((xa[:, away] > bx_ff[:, None]) & tracked[:, away]).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["behind_h"] = np.where(nh_t > 0, behind_h / nh_t, np.nan)
        f["behind_a"] = np.where(na_t > 0, behind_a / na_t, np.nan)
    f["behind_diff"] = f["behind_h"] - f["behind_a"]
    f["n_tracked_h"], f["n_tracked_a"] = nh_t, na_t
    f["cent_h"] = _nanmean_axis1(xa[:, home], tracked[:, home]) / L
    f["cent_a"] = _nanmean_axis1(xa[:, away], tracked[:, away]) / L
    f["cent_mid"] = (f["cent_h"] + f["cent_a"]) / 2
    f["spread_x_h"] = _nanstd_axis1(xa[:, home], tracked[:, home])
    f["spread_x_a"] = _nanstd_axis1(xa[:, away], tracked[:, away])
    f["spread_y_h"] = _nanstd_axis1(pos[:, home, 1], tracked[:, home])
    f["spread_y_a"] = _nanstd_axis1(pos[:, away, 1], tracked[:, away])
    f["ball_minus_cent"] = f["ball_x"] - f["cent_mid"]

    # ---------------------------------------------------------------- motion
    vx = np.full(xa.shape, np.nan, np.float32)
    vy = np.full(xa.shape, np.nan, np.float32)
    vx[1:-1] = (xa[2:] - xa[:-2]) * FPS / 2
    vy[1:-1] = (pos[2:, :, 1] - pos[:-2, :, 1]) * FPS / 2
    sp = np.hypot(vx, vy)
    good = np.isfinite(sp) & (sp < MAX_SPEED)
    f["speed_h"] = _nanmean_axis1(sp[:, home], good[:, home])
    f["speed_a"] = _nanmean_axis1(sp[:, away], good[:, away])
    f["speed_all"] = _nanmean_axis1(sp, good)
    f["vx_h"] = _nanmean_axis1(vx[:, home], good[:, home])        # + = home moving forward
    f["vx_a"] = -_nanmean_axis1(vx[:, away], good[:, away])       # + = away moving forward
    f["vx_diff"] = f["vx_h"] - f["vx_a"]

    # ---------------------------------------------------------------- context
    core = ["near", "dmin_diff", "seen", "ball_speed", "speed_all", "vx_diff", "behind_diff",
            "ball_x", "ball_out"]
    f["n3_diff"] = f["n3_h"] - f["n3_a"]
    core.append("n3_diff")
    windows = {"c13": (-6, 6), "c51": (-25, 25), "c125": (-62, 62), "c375": (-187, 187),
               "p25": (-25, -1), "p75": (-75, -1), "f25": (1, 25), "f75": (1, 75)}
    for name in core:
        for w, (lo, hi) in windows.items():
            f["%s_%s" % (name, w)] = roll_mean(f[name], lo, hi)

    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})
