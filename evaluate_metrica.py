"""Reproduce the Metrica results in MBDS.md (section 5). Public data only; the provided match is not read.

Leave-one-game-out on Metrica sample games 1-3: train on two games (clean + three degraded copies,
both team orientations), predict the third, score with the organisers' possession_metrics.py.
Reports, per kind of test copy: the organisers' baseline (nearest player + majority vote over 50
frames, re-implemented on our features), our frame classifier alone (argmax), and the submitted
system (classifier + Viterbi decoder with LAM_TEAM / LAM_DEAD), then the decoder-cost grid.

    python evaluate_metrica.py          # ~10 min
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from possession import model, pipeline  # noqa: E402


def main():
    sets = pipeline.metrica_sets()
    oof = pipeline.cross_validate(sets)
    rows = []
    for (g, kind), p in oof.items():
        s = sets[(g, kind)]
        for system, lab in (("organisers' baseline (smooth50)", model.baseline_labels(s["F"], s["period"], 50)),
                            ("frame classifier alone", model.argmax_labels(p, s["period"])),
                            ("classifier + decoder (submitted)",
                             model.decode(p, s["period"], pipeline.LAM_TEAM, pipeline.LAM_DEAD))):
            r = model.score(s["gt"], lab)
            rows.append({"game": g, "kind": kind, "system": system, "spellF1": r["spell_f1_mean"],
                         "macroF1": r["macro_f1"], "possessions": r["n_pred_spells"], "true": r["n_gt_spells"]})
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["system"]))
    t = df.pivot_table(index="system", columns="kind", values="spellF1", aggfunc="mean").loc[order]
    t["mean of degraded"] = t[list(pipeline.DEGRADED_KINDS)].mean(axis=1)
    m = df[df["kind"] != "clean"].groupby("system")["macroF1"].mean().loc[order]
    t["macroF1 (degraded)"] = m
    print("\nspellF1, leave-one-game-out on Metrica (mean of the 3 held-out games)\n")
    print(t[["clean"] + list(pipeline.DEGRADED_KINDS) + ["mean of degraded", "macroF1 (degraded)"]].round(3).to_markdown())
    sub = df[(df["system"] == order[-1]) & (df["kind"] != "clean")]
    print("\nsubmitted system, possessions predicted / true (degraded copies, all games): %d / %d"
          % (sub["possessions"].sum(), sub["true"].sum()))
    grid = pd.DataFrame(pipeline.tune_decoder(sets, oof)).groupby(["lam_team", "lam_dead"])["spell_f1"].mean()
    print("\ndecoder costs: spellF1 (mean of degraded copies and held-out games)\n")
    print(grid.unstack().round(3).to_markdown())


if __name__ == "__main__":
    main()
