"""Sequence model: a temporal convolutional network (TCN) over the per-frame features.

The tree classifier sees each frame through its features, whose rolling windows reach up to +/-7.5 s.
Possession is a property of whole phases of play, so a second model reads the feature sequence
itself: the 124 features at 5 frames per second, through 9 residual blocks of dilated 1-D
convolutions (kernel 5, dilations 1-32), which gives every prediction up to about +/-34 s of context
(168 steps at 5 fps). It is trained on the same games, copies and team orientations as the trees,
and its probabilities are averaged with theirs before the decoder.

At 25 fps a match is read as five interleaved 5 fps sequences (frames 0, 5, 10, ...; 1, 6, 11, ...),
each predicted in both team orientations; the outputs are put back in frame order and averaged.
Missing values (ball not seen, etc.) enter as zeros plus a missing-value indicator per feature.
All randomness is seeded; training runs on the CPU.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import X

STEP = 5                          # 25 fps -> 5 fps
SWAP = [1, 0, 2]                  # H <-> A
STEPS, BATCH, WINDOW, LR = 1500, 16, 512, 2e-3


# ------------------------------------------------------------------------------------ inputs
class Scaler:
    """Robust scaling (median / interquartile range), clipped, plus missing-value indicators."""

    def fit(self, X):
        med = np.nanmedian(X, 0)
        q25, q75 = np.nanpercentile(X, [25, 75], 0)
        sd = np.nanstd(X, 0)
        iqr = np.where(q75 - q25 > 1e-6, q75 - q25, np.where(sd > 1e-6, sd, 1.0))
        self.med, self.iqr = np.nan_to_num(med).astype(np.float32), iqr.astype(np.float32)
        return self

    def __call__(self, F):
        Z = (F - self.med) / self.iqr
        miss = np.isnan(Z)
        Z = np.clip(np.where(miss, 0, Z), -8, 8)
        return np.concatenate([Z, miss.astype(np.float32)], 1).astype(np.float32)

    def state(self):
        return {"med": self.med, "iqr": self.iqr}

    @classmethod
    def from_state(cls, st):
        s = cls()
        s.med, s.iqr = np.asarray(st["med"], np.float32), np.asarray(st["iqr"], np.float32)
        return s


# ------------------------------------------------------------------------------------ network
class _ChanNorm(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.ln = nn.LayerNorm(h)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class _Block(nn.Module):
    def __init__(self, h, d, p):
        super().__init__()
        self.n = _ChanNorm(h)
        self.c1 = nn.Conv1d(h, h, 5, dilation=d, padding=2 * d)
        self.c2 = nn.Conv1d(h, h, 1)
        self.dp = nn.Dropout(p)

    def forward(self, x):
        return x + self.c2(self.dp(Fn.gelu(self.c1(self.n(x)))))


class TCN(nn.Module):
    def __init__(self, cin, h=64, dil=(1, 2, 4, 8, 16, 32, 1, 4, 16), p=0.15):
        super().__init__()
        self.inp = nn.Conv1d(cin, h, 1)
        self.drop_in = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[_Block(h, d, p) for d in dil])
        self.out = nn.Sequential(_ChanNorm(h), nn.Conv1d(h, 3, 1))

    def forward(self, x):                                   # (batch, channels, time)
        return self.out(self.blocks(self.drop_in(self.inp(x))))


# ------------------------------------------------------------------------------------ data
def _runs(period):
    from .model import period_runs
    return period_runs(period)


def sequences(F, labels, period, copy_index, orientation, n_phase=2):
    """5 fps training sequences (one per half and phase) from one copy in one team orientation."""
    out = []
    for j in range(n_phase):
        ph = (copy_index * 2 + orientation + 2 * j) % STEP
        for s, e in _runs(period):
            idx = np.arange(s + ph, e, STEP)
            out.append((F[idx], labels[idx].astype(np.int64)))
    return out


def training_data(train_sets):
    """Sequences + scaler sample from prepared sets (pipeline.prepare), both orientations."""
    seqs, sample = [], []
    for k, s in enumerate(train_sets):
        for o, (F, y) in enumerate(((s["F"], s["labels"]), (s["F_swap"], s["labels_swap"]))):
            F = F.to_numpy(np.float32) if hasattr(F, "to_numpy") else F
            seqs += sequences(F, y, s["period"], k, o)
            graded = np.flatnonzero(s["period"] > 0)
            sample.append(F[graded[(graded + k) % 16 == 0]])
    return seqs, np.vstack(sample)


# ------------------------------------------------------------------------------------ train / predict
def train(train_sets, seed=0, steps=STEPS, log=None):
    """Fit one TCN (seeded). Returns (network, scaler)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    seqs, sample = training_data(train_sets)
    scaler = Scaler().fit(sample)
    data = [(scaler(F), y) for F, y in seqs]
    lens = np.array([len(y) for _, y in data], float)
    pick = np.clip(lens - WINDOW, 1, None) / np.clip(lens - WINDOW, 1, None).sum()
    nz = data[0][0].shape[1] // 2
    net = TCN(data[0][0].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    net.train()
    for it in range(steps):
        xb, yb = [], []
        for k in rng.choice(len(data), BATCH, p=pick):
            Z, y = data[k]
            s = int(rng.integers(0, max(1, len(y) - WINDOW)))
            z = Z[s:s + WINDOW].copy()
            z[:, :nz] += rng.normal(0, 0.1, (len(z), nz)).astype(np.float32)   # input noise
            xb.append(z.T); yb.append(y[s:s + WINDOW])
        loss = Fn.cross_entropy(net(torch.from_numpy(np.stack(xb))), torch.from_numpy(np.stack(yb)),
                                ignore_index=X)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if log and (it + 1) % 500 == 0:
            log("  tcn seed %d: step %d/%d" % (seed, it + 1, steps))
    net.eval()
    return net, scaler


@torch.no_grad()
def predict_proba(net, scaler, F, F_swap, period):
    """[n, 3] p(H, A, D) at 25 fps: five 5 fps phases x both team orientations, averaged."""
    F = F.to_numpy(np.float32) if hasattr(F, "to_numpy") else F
    F_swap = F_swap.to_numpy(np.float32) if hasattr(F_swap, "to_numpy") else F_swap
    P = np.zeros((len(period), 3), np.float32)
    for s, e in _runs(period):
        for ph in range(STEP):
            idx = np.arange(s + ph, e, STEP)
            for G, cols in ((F, [0, 1, 2]), (F_swap, SWAP)):
                z = torch.from_numpy(scaler(G[idx]).T[None])
                P[idx] += torch.softmax(net(z)[0], 0).numpy().T[:, cols] / 2
    return P


def save(path, models):
    torch.save([{"net": n.state_dict(), **sc.state()} for n, sc in models], path)


def load(path, cin=248):
    out = []
    for st in torch.load(path, weights_only=False):
        net = TCN(cin); net.load_state_dict(st["net"]); net.eval()
        out.append((net, Scaler.from_state(st)))
    return out
