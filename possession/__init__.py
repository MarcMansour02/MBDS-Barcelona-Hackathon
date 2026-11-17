"""Possession model for More than a Hack 2027 (Phase 1).

Pipeline: tracking -> collective per-frame features -> H/A/D probabilities from gradient-boosted
trees and a temporal convolutional network, averaged -> Viterbi decoding with switching costs ->
one label per frame.

Trained and tuned only on Metrica's public sample games (other matches). The challenge match is
used only at inference time.
"""
H, A, D, X = 0, 1, 2, 3
LETTERS = "HADX"
