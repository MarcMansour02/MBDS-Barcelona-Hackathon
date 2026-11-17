"""Regenerate the submission file from scratch.

    python make_submission.py --team OUR_TEAM_NAME

Trains the tree classifier on the three Metrica public games, adds the sequence models, predicts the
challenge match, writes submissions/<team>.csv next to the workspace (not inside this repo), then
runs the organisers' validator on it. By default the sequence models are loaded from
models/sequence_metrica.pt (shipped in this repository), which reproduces the submitted file exactly;
--retrain trains them again from the Metrica games and overwrites that file. Every random choice is
seeded.

Prints only file-level facts (path, checksum, validation verdict); nothing about the match data.
"""
import argparse
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from possession import model, paths, pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="our_team")
    ap.add_argument("--out", default=paths.OUTPUT)
    ap.add_argument("--retrain", action="store_true",
                    help="retrain the sequence models (~30 min on a CPU) instead of loading models/sequence_metrica.pt")
    ap.add_argument("--trees-only", action="store_true", help="the tree classifier alone (previous system)")
    args = ap.parse_args()

    sets = pipeline.metrica_sets()
    clf = pipeline.train_final(sets)
    print("tree classifier trained on Metrica games %s" % (pipeline.METRICA_GAMES,), flush=True)
    seq_models = None
    if not args.trees_only:
        from possession import sequence
        saved = os.path.join(HERE, "models", "sequence_metrica.pt")
        if not args.retrain and os.path.exists(saved):
            seq_models = sequence.load(saved)
        else:
            seq_models = pipeline.train_sequence(list(sets.values()))
            os.makedirs(os.path.dirname(saved), exist_ok=True)
            sequence.save(saved, seq_models)
        print("%d sequence models ready" % len(seq_models), flush=True)
    labels, _, _ = pipeline.predict_challenge(clf, seq_models)
    path = model.write_submission(os.path.join(args.out, "%s.csv" % args.team), labels)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print("wrote %s\nsha256 %s" % (path, digest))

    validator = os.path.join(paths.ORGANISERS, "evaluation", "validate_submission.py")
    ok = subprocess.run([sys.executable, validator, path], cwd=paths.ORGANISERS,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    print("organisers' validator: %s" % ("PASS" if ok else "FAIL -- run it by hand to see why"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
