"""Load matches into one common shape.

A match is a dict:
    name    str
    pos     float32 [n, S, 2]  player positions in metres (x along the length), NaN = not tracked
    team    int8    [S]        0 = home, 1 = away, for each slot
    ball    float32 [n, 2]     metres, NaN = not tracked
    period  int8    [n]        1 / 2 on frames that matter (graded), 0 elsewhere
    labels  int8    [n] | None H=0 A=1 D=2 X=3 (Metrica only; the challenge match has none here)
    pitch   (length, width) in metres
    fps     25

Slots are anonymous. Nothing downstream relies on a slot being one person, so the same code
works on Metrica (one slot per player) and on the challenge ATD (46 fragmenting tracklets).
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from . import paths
from . import H, A, D, X

FPS = 25


# ------------------------------------------------------------------------------------ Metrica

def load_metrica(game, cache=paths.CACHE):
    """Metrica public sample game 1, 2 or 3 with event-derived labels (see ext_metrica.py)."""
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, "metrica_%d.npz" % game)
    if not os.path.exists(f):
        sys.path.insert(0, paths.SOLUTION)
        from ext_metrica import load_game
        g = load_game(game, paths.METRICA)
        np.savez_compressed(f, home=g["home"], away=g["away"], ball=g["ball"],
                            labels=g["labels"], period=g["period"], frame=g["frame"])
    z = np.load(f)
    length, width = 105.0, 68.0
    scale = np.array([length, width], np.float32)
    home, away = z["home"] * scale, z["away"] * scale
    pos = np.concatenate([home, away], axis=1).astype(np.float32)
    team = np.r_[np.zeros(home.shape[1], np.int8), np.ones(away.shape[1], np.int8)]
    labels = z["labels"].astype(np.int8).copy()
    period = z["period"].astype(np.int8).copy()
    # frames before the first labelled event are not part of the evaluation
    period[labels == X] = 0
    return {"name": "metrica_%d" % game, "pos": pos, "team": team,
            "ball": (z["ball"] * scale).astype(np.float32), "period": period,
            "labels": labels, "pitch": (length, width), "fps": FPS}


def gt_codes(match):
    """Labels as the organisers' scorer wants them: X outside the graded periods, and one X frame
    between periods so the scorer treats each half as its own run (as it does for BAR-ATH)."""
    gt = np.where(match["period"] > 0, match["labels"], X).astype(np.int8)
    change = np.flatnonzero(np.diff(match["period"].astype(int)) != 0) + 1
    gt[change] = X
    return gt


# ------------------------------------------------------------------------- challenge (BAR-ATH)

def parse_atd(raw_path, n_frames, n_slots, pitch):
    """Raw FIFA/EPTS ATD lines '<frame>:<slot>;<slot>;...:<ball>' -> positions in metres.

    Each group is 'x,y[,speed]' or NaN. frame_id counts from 1, video frames from 0.
    """
    length, width = pitch
    pos = np.full((n_frames, n_slots, 2), np.nan, np.float32)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n_frames:
                continue
            row = pos[i]
            for slot, group in enumerate(parts[1].split(";")):
                if slot < n_slots and group and group[0] != "N":
                    v = group.split(",")
                    row[slot, 0] = float(v[0]) * length
                    row[slot, 1] = float(v[1]) * width
            b = parts[2]
            if b and b[0] != "N":
                v = b.split(",")
                ball[i] = (float(v[0]) * length, float(v[1]) * width)
    return pos, ball


def load_challenge(cache=paths.CACHE):
    """The provided match (tactical source). Read locally; nothing here leaves this machine."""
    paths.organisers_on_path()
    from games import GAME, GAMES, half_video_blocks, require, source_frames
    from starter.baseline import team_of_each_track

    n = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, "%s_tactical.npz" % GAME)
    meta, raw = require(["atd_meta", "atd_raw"], GAME, None)
    team = team_of_each_track(meta)
    if not os.path.exists(f):
        pos, ball = parse_atd(raw, n, len(team), pitch)
        np.savez_compressed(f, pos=pos, ball=ball)
    z = np.load(f)
    period = np.zeros(n, np.int8)
    for half, first, last in half_video_blocks(GAME):
        period[first:last + 1] = half
    return {"name": GAME, "pos": z["pos"], "team": team.astype(np.int8), "ball": z["ball"],
            "period": period, "labels": None, "pitch": tuple(pitch), "fps": FPS}


def swap_teams(match):
    """Same match with home and away exchanged (labels too). Used for symmetric augmentation."""
    out = dict(match)
    out["team"] = (1 - match["team"]).astype(np.int8)
    if match["labels"] is not None:
        lab = match["labels"].copy()
        lab[match["labels"] == H] = A
        lab[match["labels"] == A] = H
        out["labels"] = lab
    out["name"] = match["name"] + "_swapped"
    return out
