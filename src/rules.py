"""
A generator of random technical trading rules on one price series.
Every rule produces a position in {-1, 0, 1} decided with information up to the close of day t and applied to the
return of day t + 1 (no look-ahead). Transaction costs are charged on every change of position.
"""

import numpy as np
import pandas as pd

FAMILIES = ["ma_crossover", "momentum", "mean_reversion", "breakout"]


class RuleFactory:
    def __init__(self, close):
        self.close = pd.Series(np.asarray(close, float))
        self._ma, self._max, self._min, self._sd = {}, {}, {}, {}

    def ma(self, w):
        if w not in self._ma:
            self._ma[w] = self.close.rolling(w).mean().to_numpy()
        return self._ma[w]

    def rmax(self, w):
        if w not in self._max:
            self._max[w] = self.close.rolling(w).max().to_numpy()
        return self._max[w]

    def rmin(self, w):
        if w not in self._min:
            self._min[w] = self.close.rolling(w).min().to_numpy()
        return self._min[w]

    def rsd(self, w):
        if w not in self._sd:
            self._sd[w] = self.close.rolling(w).std().to_numpy()
        return self._sd[w]

    def signal(self, rule):
        c = self.close.to_numpy()
        f, short = rule["family"], rule["short"]
        low = -1.0 if short else 0.0
        if f == "ma_crossover":
            fast, slow = self.ma(rule["fast"]), self.ma(rule["slow"])
            pos = np.where(fast > slow, 1.0, low)
            pos[np.isnan(slow)] = 0
        elif f == "momentum":
            L = rule["lookback"]
            past = np.r_[np.full(L, np.nan), c[:-L]]
            pos = np.where(c > past, 1.0, low)
            pos[np.isnan(past)] = 0
        elif f == "mean_reversion":
            L, k = rule["lookback"], rule["threshold"]
            z = (c - self.ma(L)) / self.rsd(L)
            raw = np.where(z < -k, 1.0, np.where(z > k, low, np.nan))
            raw[:L] = 0
            pos = pd.Series(raw).ffill().fillna(0).to_numpy() if rule["hold"] else np.nan_to_num(raw)
        elif f == "breakout":
            N = rule["window"]
            up = c >= self.rmax(N)
            dn = c <= self.rmin(max(5, N // 2))
            raw = np.where(up, 1.0, np.where(dn, low, np.nan))
            raw[:N] = 0
            pos = pd.Series(raw).ffill().fillna(0).to_numpy()
        else:
            raise ValueError(f)
        return pos

    def returns(self, rule, asset_returns, cost_bps):
        """Strategy returns: yesterday's position times today's return, minus costs on position changes."""
        pos = self.signal(rule)
        held = np.r_[0.0, pos[:-1]]
        turnover = np.abs(np.diff(np.r_[0.0, held]))
        return held * asset_returns - turnover * cost_bps / 1e4, held


def random_rules(n, seed=0):
    rng = np.random.default_rng(seed)
    rules = []
    for i in range(n):
        f = FAMILIES[i % len(FAMILIES)]
        r = {"id": i, "family": f, "short": bool(rng.random() < 0.5)}
        if f == "ma_crossover":
            fast = int(rng.integers(2, 100)); r.update(fast=fast, slow=int(rng.integers(fast + 5, 301)))
        elif f == "momentum":
            r.update(lookback=int(rng.integers(5, 301)))
        elif f == "mean_reversion":
            r.update(lookback=int(rng.integers(5, 101)), threshold=round(float(rng.uniform(0.25, 2.5)), 2), hold=bool(rng.random() < 0.5))
        else:
            r.update(window=int(rng.integers(10, 301)))
        rules.append(r)
    return rules


def describe(rule):
    s = "long/short" if rule["short"] else "long/flat"
    f = rule["family"]
    if f == "ma_crossover":
        return f"MA crossover {rule['fast']}/{rule['slow']} days, {s}"
    if f == "momentum":
        return f"Momentum, {rule['lookback']}-day lookback, {s}"
    if f == "mean_reversion":
        return f"Mean reversion, {rule['lookback']}-day z-score beyond {rule['threshold']}" + (", held" if rule["hold"] else "") + f", {s}"
    return f"Breakout, {rule['window']}-day channel, {s}"
