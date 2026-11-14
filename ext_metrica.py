"""Load Metrica's PUBLIC sample games (other matches) and derive H/A/D possession labels from events.

Why this exists: the organisers' rules say anything we tune must be tuned on EXTERNAL data. These three
public matches are our dev set. Nothing here touches the challenge match.

    from ext_metrica import load_game
    g = load_game(1, root="external/metrica/data")     # games 1, 2 (CSV) and 3 (EPTS, needs kloppy)
    g["home"], g["away"]   -> float32 arrays [n_frames, n_players, 2], coordinates 0..1, NaN = not tracked
    g["ball"]              -> [n_frames, 2], NaN = not tracked
    g["labels"]            -> int8 [n_frames]: 0 = H, 1 = A, 2 = D, 3 = X (no label)
    g["period"]            -> int8 [n_frames]

Label rules (written from football logic, NOT tuned on any data), applied to the event stream in time order:
  * PASS / SHOT / CARRY / RECOVERY / SET PIECE by team T  -> T has the ball from that event's start frame.
  * BALL LOST, CHALLENGE (won/lost), CARD                  -> no change. A loose ball stays with the team that last
                                                              had it until the other team controls it (RECOVERY).
  * BALL OUT                  -> dead from the event's END frame (the ball crossing the line).
  * goal / offside / foul / end of half -> dead (goal and offside from the event's end, foul from its start).
  * dead ends at the next SET PIECE (restart), which also gives the possession to the restarting team.
Event data about OTHER matches is allowed by the rules. These labels differ a little from the organisers' definition
(which comes from corrected tracking), so treat scores on them as a guide, not as the challenge score.
"""
import json
import os

import numpy as np
import pandas as pd

H, A, D, X = 0, 1, 2, 3
OWNING = {"PASS", "SHOT", "CARRY", "RECOVERY", "SET PIECE"}


def _dead_from(etype, subs, start, end):
    """Frame at which the ball becomes dead because of this event, or None."""
    if etype == "BALL OUT":
        return max(start, end)
    if etype == "SHOT" and "GOAL" in subs:
        return max(start, end)
    if etype == "BALL LOST" and ({"OFFSIDE", "HAND BALL"} & subs):
        return max(start, end)
    if etype == "BALL LOST" and "END HALF" in subs:
        return start
    if etype == "PASS" and "OFFSIDE" in subs:
        return max(start, end)
    if etype == "FAULT RECEIVED":
        return start
    if etype == "CHALLENGE" and "FAULT" in subs and "ADVANTAGE" not in subs:
        return start
    return None


def events_to_labels(events, n_frames, first_frame=1):
    """events: DataFrame with columns team ('H'/'A'), type, subs (set of upper-case str), start, end (frames).

    Returns int8 labels indexed by (frame - first_frame). Frames before the first owning event stay X."""
    labels = np.full(n_frames, X, np.int8)
    ev = events.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    state, since = X, 0  # current label and the index from which it applies
    pending_dead = []

    def idx(frame):
        return int(min(max(frame - first_frame, 0), n_frames))

    def set_state(new, frame):
        nonlocal state, since
        i = idx(frame)
        if i > since:
            labels[since:i] = state
        state, since = new, max(i, since)

    for row in ev.itertuples(index=False):
        # a dead-ball moment scheduled by an earlier event takes effect once its frame is reached
        for frame in sorted(f for f in pending_dead if f <= row.start):
            set_state(D, frame)
            pending_dead.remove(frame)
        dead_at = _dead_from(row.type, row.subs, row.start, row.end)
        if row.type in OWNING:
            set_state(H if row.team == "H" else A, row.start)
        if dead_at is not None:
            pending_dead.append(dead_at)
    for frame in sorted(pending_dead):
        set_state(D, frame)
    labels[since:] = state
    return labels


# ---------------------------------------------------------------- games 1 and 2 (CSV)

def _read_tracking_csv(path):
    raw = pd.read_csv(path, header=None, skiprows=3)
    period = raw.iloc[:, 0].to_numpy(np.int8)
    frame = raw.iloc[:, 1].to_numpy(np.int64)
    values = raw.iloc[:, 3:].to_numpy(np.float32)
    players = values[:, :-2].reshape(len(raw), -1, 2)
    return period, frame, players, values[:, -2:]


def _load_csv_game(game, root):
    folder = os.path.join(root, "Sample_Game_%d" % game)
    stem = os.path.join(folder, "Sample_Game_%d_" % game)
    period, frame, home, ball = _read_tracking_csv(stem + "RawTrackingData_Home_Team.csv")
    _, _, away, _ = _read_tracking_csv(stem + "RawTrackingData_Away_Team.csv")
    raw = pd.read_csv(stem + "RawEventsData.csv")
    events = pd.DataFrame({
        "team": raw["Team"].map({"Home": "H", "Away": "A"}),
        "type": raw["Type"].str.upper(),
        "subs": [set(str(s).upper().replace("-", " ").split()) if isinstance(s, str) else set()
                 for s in raw["Subtype"]],
        "start": raw["Start Frame"], "end": raw["End Frame"]})
    # subtype words like "GOAL KICK" would split into {"GOAL","KICK"}; keep a joined copy for goal tests
    events["subs"] = [s | ({"GOAL"} if "GOAL" in sub and typ == "SHOT" else set())
                      for s, sub, typ in zip(events["subs"], raw["Subtype"].fillna("").str.upper(), events["type"])]
    return period, frame, home, away, ball, events


# ---------------------------------------------------------------- game 3 (EPTS + JSON)

def _load_epts_game(root):
    from kloppy import metrica  # pip install kloppy==3.18.0
    folder = os.path.join(root, "Sample_Game_3")
    ds = metrica.load_tracking_epts(meta_data=os.path.join(folder, "Sample_Game_3_metadata.xml"),
                                    raw_data=os.path.join(folder, "Sample_Game_3_tracking.txt"))
    df = ds.to_df()
    home_team, away_team = ds.metadata.teams[0], ds.metadata.teams[1]
    def block(team):
        cols = [p.player_id for p in team.players]
        xs = np.stack([df["%s_x" % c].to_numpy(np.float32) for c in cols if "%s_x" % c in df.columns], axis=1)
        ys = np.stack([df["%s_y" % c].to_numpy(np.float32) for c in cols if "%s_y" % c in df.columns], axis=1)
        return np.stack([xs, ys], axis=2)
    home, away = block(home_team), block(away_team)
    ball = np.stack([df["ball_x"].to_numpy(np.float32), df["ball_y"].to_numpy(np.float32)], axis=1)
    frame = df["frame_id"].to_numpy(np.int64)
    period = df["period_id"].to_numpy(np.int8)
    team_of = {home_team.team_id: "H", away_team.team_id: "A"}
    data = json.load(open(os.path.join(folder, "Sample_Game_3_events.json")))["data"]
    rows = []
    for e in data:
        subs = e.get("subtypes")
        names = [s["name"] for s in subs] if isinstance(subs, list) else ([subs["name"]] if subs else [])
        words = set(" ".join(names).upper().replace("-", " ").split())
        if "ON TARGET" in " ".join(names).upper() and "GOAL" in names:
            words.add("GOAL")
        rows.append((team_of.get(e["team"]["id"]), e["type"]["name"].upper(), words,
                     e["start"]["frame"], e["end"]["frame"]))
    events = pd.DataFrame(rows, columns=["team", "type", "subs", "start", "end"])
    return period, frame, home, away, ball, events


def load_game(game, root="external/metrica/data"):
    if game in (1, 2):
        period, frame, home, away, ball, events = _load_csv_game(game, root)
    elif game == 3:
        period, frame, home, away, ball, events = _load_epts_game(root)
    else:
        raise ValueError("games are 1, 2, 3")
    labels = events_to_labels(events.dropna(subset=["team"]), len(frame), first_frame=int(frame[0]))
    return {"game": game, "fps": 25, "frame": frame, "period": period, "home": home, "away": away,
            "ball": ball, "labels": labels, "events": events}


def label_summary(labels, fps=25):
    """Possessions as the organisers' scorer counts them: maximal runs of H or of A (D splits them)."""
    out = {"frames": len(labels), "D_share": float((labels == D).mean()), "X_share": float((labels == X).mean())}
    runs, i, lengths = 0, 0, []
    while i < len(labels):
        if labels[i] in (H, A):
            j = i
            while j < len(labels) and labels[j] == labels[i]:
                j += 1
            lengths.append(j - i)
            i = j
        else:
            i += 1
    out["possessions"] = len(lengths)
    out["mean_duration_s"] = float(np.mean(lengths) / fps) if lengths else 0.0
    out["home_share_of_alive"] = float((labels == H).sum() / max(1, ((labels == H) | (labels == A)).sum()))
    return out


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "external/metrica/data"
    for game in (1, 2, 3):
        try:
            g = load_game(game, root)
        except ImportError:
            print("game %d skipped: kloppy not installed" % game)
            continue
        s = label_summary(g["labels"])
        seen = float((~np.isnan(g["ball"][:, 0])).mean())
        print("game %d: %d frames, ball tracked %.1f%%, players/frame home %.1f away %.1f | possessions %d, "
              "mean %.1f s, dead %.1f%%, unlabelled %.1f%%, home share of live %.2f" % (
                  game, s["frames"], 100 * seen, float((~np.isnan(g["home"][:, :, 0])).sum(1).mean()),
                  float((~np.isnan(g["away"][:, :, 0])).sum(1).mean()), s["possessions"],
                  s["mean_duration_s"], 100 * s["D_share"], 100 * s["X_share"], s["home_share_of_alive"]))
