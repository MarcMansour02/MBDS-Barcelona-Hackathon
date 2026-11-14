"""Possession model for More than a Hack 2027 (Phase 1).

Pipeline: tracking -> collective per-frame features -> gradient-boosted H/A/D probabilities
-> Viterbi decoding with switching costs -> one label per frame.

Trained and tuned only on Metrica's public sample games (other matches). The challenge match is
used only at inference time.
"""
H, A, D, X = 0, 1, 2, 3
LETTERS = "HADX"
