"""Regenerate the submission file from scratch.

    python make_submission.py --team OUR_TEAM_NAME

Trains on the three Metrica public games, predicts the challenge match, writes
submissions/<team>.csv next to the workspace (not inside this repo), then runs the organisers'
validator on it. Deterministic: two runs produce the same file.

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
    args = ap.parse_args()

    sets = pipeline.metrica_sets()
    clf = pipeline.train_final(sets)
    print("model trained on Metrica games %s" % (pipeline.METRICA_GAMES,), flush=True)
    labels, _, _ = pipeline.predict_challenge(clf)
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
