"""
Statistics of backtest overfitting, written from scratch:
Sharpe ratio, Probabilistic Sharpe Ratio (PSR), expected maximum Sharpe under the null, Deflated Sharpe Ratio (DSR),
effective number of independent trials, and the Probability of Backtest Overfitting (PBO) by CSCV.

References
- Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. Journal of Risk, 15(2).
- Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias,
  Backtest Overfitting and Non-Normality. Journal of Portfolio Management, 40(5).
- Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The Probability of Backtest Overfitting.
  Journal of Computational Finance, 20(4).
"""

from itertools import combinations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329
DAYS = 252


def sharpe(r, axis=0):
    """Per-period Sharpe ratio (not annualized): mean / standard deviation."""
    r = np.asarray(r, float)
    sd = r.std(axis=axis, ddof=1)
    return np.divide(r.mean(axis=axis), sd, out=np.zeros_like(sd, dtype=float), where=sd > 0)


def annualize(sr):
    return np.asarray(sr) * np.sqrt(DAYS)


def psr(sr_hat, sr_star, T, skew, kurt):
    """
    Probabilistic Sharpe Ratio: probability that the true Sharpe exceeds sr_star, given an estimate sr_hat
    from T observations with the given skewness and (non-excess) kurtosis. All Sharpe ratios per period.
    PSR = Phi( (sr_hat - sr_star) * sqrt(T - 1) / sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat^2) )
    """
    denom = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2)
    return float(stats.norm.cdf((sr_hat - sr_star) * np.sqrt(T - 1) / denom))


def expected_max_sharpe(var_sr, n_trials):
    """
    Expected maximum of n_trials Sharpe ratios when every true Sharpe is zero (False Strategy Theorem):
    E[max SR] ~ sqrt(V[SR]) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)))
    """
    n = max(float(n_trials), 1.0 + 1e-9)
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
                                     + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe(best_returns, all_sharpes, n_trials=None):
    """
    Deflated Sharpe Ratio of the selected strategy: its PSR measured against the Sharpe that the best of
    n_trials worthless strategies would be expected to reach, instead of against zero.
    """
    r = np.asarray(best_returns, float)
    sr = float(sharpe(r))
    n = len(all_sharpes) if n_trials is None else n_trials
    sr0 = expected_max_sharpe(np.var(all_sharpes, ddof=1), n)
    sk, ku = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    return {"sharpe": sr, "sharpe_annual": float(annualize(sr)), "sr0": sr0, "sr0_annual": float(annualize(sr0)),
            "n_trials": float(n), "skew": sk, "kurtosis": ku, "T": len(r),
            "psr_vs_zero": psr(sr, 0.0, len(r), sk, ku), "dsr": psr(sr, sr0, len(r), sk, ku)}


def effective_trials(returns_matrix):
    """
    Number of effectively independent strategies: participation ratio of the eigenvalues of their correlation
    matrix, (sum lambda)^2 / sum lambda^2. Equals N for uncorrelated strategies and 1 for identical ones.
    """
    C = np.corrcoef(returns_matrix, rowvar=False)
    C = np.nan_to_num(C)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum()), lam[::-1]


def pbo_cscv(returns_matrix, n_blocks=16):
    """
    Probability of Backtest Overfitting by Combinatorially Symmetric Cross-Validation.
    The sample is cut into n_blocks; for every way of choosing half the blocks as in-sample, the strategy with
    the best in-sample Sharpe is located in the out-of-sample ranking. PBO = share of splits where it lands
    below the median. Block sums are precomputed so all C(16, 8) = 12,870 splits are evaluated exactly.
    """
    M = np.asarray(returns_matrix, float)
    T, N = M.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    s1 = np.array([M[b].sum(0) for b in blocks])
    s2 = np.array([(M[b] ** 2).sum(0) for b in blocks])
    cnt = np.array([len(b) for b in blocks], float)
    logits, is_best, oos_of_best, oos_rank = [], [], [], []
    for combo in combinations(range(n_blocks), n_blocks // 2):
        m = np.zeros(n_blocks, bool); m[list(combo)] = True
        sr = []
        for mask in (m, ~m):
            n = cnt[mask].sum(); a = s1[mask].sum(0); b = s2[mask].sum(0)
            mean = a / n; var = (b - n * mean ** 2) / (n - 1)
            sr.append(np.divide(mean, np.sqrt(var), out=np.zeros(N), where=var > 0))
        best = int(np.argmax(sr[0]))
        w = (stats.rankdata(sr[1])[best]) / (N + 1)
        logits.append(np.log(w / (1 - w)))
        is_best.append(sr[0][best]); oos_of_best.append(sr[1][best]); oos_rank.append(w)
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "logits": logits, "is_sharpe": np.array(is_best),
            "oos_sharpe": np.array(oos_of_best), "oos_rank": np.array(oos_rank), "splits": len(logits)}


def min_track_record_length(sr_hat, sr_star, skew, kurt, prob=0.95):
    """Observations needed for PSR(sr_star) to reach `prob` (Bailey and Lopez de Prado, 2012)."""
    if sr_hat <= sr_star:
        return float("inf")
    z = stats.norm.ppf(prob)
    return float(1 + (1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2) * (z / (sr_hat - sr_star)) ** 2)
