"""Where things live. Every location can be overridden with an environment variable; otherwise the
first existing candidate next to this repository (the "workspace" folder) is used."""
import os
import sys

SOLUTION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # this repository
WORKSPACE = os.environ.get("MTAH_WORKSPACE", os.path.dirname(SOLUTION))          # folder that contains it


def _first(env, candidates):
    if os.environ.get(env):
        return os.environ[env]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


# organisers' repository (games.py, starter/, evaluation/)
ORGANISERS = _first("MTAH_REPO", [os.path.join(WORKSPACE, d) for d in
                                  ("more-than-a-hack-2027-main", "more-than-a-hack-2027")])
# Metrica public sample data: the data/ folder of github.com/metrica-sports/sample-data
METRICA = _first("METRICA_DIR", [os.path.join(WORKSPACE, *p) for p in
                                 (("external", "metrica", "data"), ("sample-data", "data"),
                                  ("sample-data-master", "data"))])
# Caches and outputs hold data-derived files. They live OUTSIDE this repository so they can never be
# committed (derived frame-level tracking counts as redistribution).
CACHE = os.environ.get("MTAH_CACHE", os.path.join(WORKSPACE, "cache"))
OUTPUT = os.environ.get("MTAH_OUTPUT", os.path.join(WORKSPACE, "submissions"))
# provided match data: the folder that contains tactical/ (read by the organisers' games.py)
os.environ.setdefault("MTAH_DATA_DIR", _first("MTAH_DATA_DIR", [os.path.join(WORKSPACE, "data"), WORKSPACE]))
if not os.path.isdir(os.path.join(os.environ["MTAH_DATA_DIR"], "tactical")) and \
        os.path.isdir(os.path.join(WORKSPACE, "tactical")):
    os.environ["MTAH_DATA_DIR"] = WORKSPACE


def organisers_on_path():
    """Make the organisers' games.py, starter/ and evaluation/ importable."""
    for p in (ORGANISERS, os.path.join(ORGANISERS, "evaluation")):
        if p not in sys.path:
            sys.path.insert(0, p)
