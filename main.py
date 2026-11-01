"""
Backtest Overfitting Detector: is the Sharpe ratio of my best strategy real, or luck from trying 1,000 things?
Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014) and Probability of Backtest Overfitting (CSCV) on the S&P 500.
Run:  python main.py      (about two minutes on a laptop CPU)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats

from src.rules import RuleFactory, random_rules, describe
from src.stats import (sharpe, annualize, psr, expected_max_sharpe, deflated_sharpe, pbo_cscv, min_track_record_length)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BG, CARD, TEXT, TEXT2, MUTED, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#8a8980", "#3a3936"
BLUE, ORANGE, AQUA, AMBER, PINK, VIOLET, RED = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767"
FAMILY_COLORS = {"ma_crossover": BLUE, "momentum": AMBER, "mean_reversion": VIOLET, "breakout": AQUA}
FAMILY_NAMES = {"ma_crossover": "MA crossover", "momentum": "Momentum", "mean_reversion": "Mean reversion", "breakout": "Breakout"}
COST_BPS = 5
N_RULES = 1000
SPLIT = "2000-01-01"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": TEXT,
    "axes.labelcolor": TEXT2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "grid.color": GRID, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def plotly_layout(**kw):
    base = dict(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family="Inter, Segoe UI, Arial", color=TEXT, size=13),
                hoverlabel=dict(bgcolor=CARD, bordercolor=GRID, font=dict(color=TEXT)), margin=dict(l=20, r=20, t=70, b=20))
    base.update(kw)
    return base


def title(ax, text, size=12.5):
    ax.set_title(text, loc="left", fontsize=size, fontweight="bold", color=TEXT)


def drawdown(r):
    eq = np.exp(np.cumsum(r))
    return eq / np.maximum.accumulate(eq) - 1


def main():
    df = pd.read_csv(ROOT / "data" / "sp500_daily.csv", parse_dates=["date"])
    dates = df["date"]
    close = df["close"].to_numpy()
    r = np.r_[0.0, np.diff(np.log(close))]
    IS = (dates < SPLIT).to_numpy()
    OOS = ~IS
    is_idx = np.where(IS)[0][1:]
    oos_idx = np.where(OOS)[0]
    # Detrended returns: remove each period's average daily return, so buy-and-hold has a Sharpe of exactly zero
    # and any Sharpe left is market timing, not the equity premium (Aronson, 2007).
    rd = r.copy()
    rd[is_idx] -= r[is_idx].mean(); rd[oos_idx] -= r[oos_idx].mean(); rd[0] = 0
    print(f"S&P 500 daily, {dates.iloc[0].date()} to {dates.iloc[-1].date()}: in-sample {len(is_idx):,} days (1950-1999), "
          f"out-of-sample {len(oos_idx):,} days (2000-2018)")

    # ------------------------------------------------------------- 1,000 random rules
    factory = RuleFactory(close)
    rules = random_rules(N_RULES, seed=0)
    R = np.array([factory.returns(rule, rd, COST_BPS)[0] for rule in rules]).T
    R_raw = np.array([factory.returns(rule, r, COST_BPS)[0] for rule in rules]).T
    sr_is, sr_oos = sharpe(R[is_idx]), sharpe(R[oos_idx])
    best = int(np.argmax(sr_is))
    best_rule = rules[best]
    print(f"Best in-sample rule: {describe(best_rule)}; Sharpe {annualize(sr_is[best]):.2f} in-sample, {annualize(sr_oos[best]):.2f} out-of-sample")

    # ------------------------------------------------------------- pure-luck benchmark: coin-flip strategies
    rng = np.random.default_rng(1)
    luck = []
    for _ in range(N_RULES):
        n_seg = len(r) // 10
        lengths = rng.geometric(1 / 20, n_seg)
        pos = np.repeat(rng.choice([-1.0, 1.0], n_seg), lengths)[:len(r)]
        held = np.r_[0.0, pos[:-1]]
        ret = held * rd - np.abs(np.diff(np.r_[0.0, held])) * COST_BPS / 1e4
        luck.append(sharpe(ret[is_idx]))
    luck = np.array(luck)

    # ------------------------------------------------------------- Deflated Sharpe Ratio
    dsr = deflated_sharpe(R[is_idx, best], sr_is, N_RULES)
    C = np.nan_to_num(np.corrcoef(R[is_idx], rowvar=False))
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    cum = np.cumsum(lam) / lam.sum()
    q = N_RULES / len(is_idx)
    mp_edge = (1 + np.sqrt(q)) ** 2
    n_eff = {"90% of variance": int(np.searchsorted(cum, 0.90) + 1), "95% of variance": int(np.searchsorted(cum, 0.95) + 1),
             "Eigenvalues above the noise edge": int((lam > mp_edge).sum())}
    dsr_eff = {k: deflated_sharpe(R[is_idx, best], sr_is, v)["dsr"] for k, v in n_eff.items()}
    n_grid = np.unique(np.round(np.logspace(0, 3, 60)).astype(int))
    dsr_curve = np.array([deflated_sharpe(R[is_idx, best], sr_is, n)["dsr"] for n in n_grid])
    n_pass = int(n_grid[dsr_curve >= 0.95].max()) if (dsr_curve >= 0.95).any() else 1
    print(f"DSR: naive PSR {dsr['psr_vs_zero']:.4f}; expected max Sharpe of 1,000 worthless rules {annualize(dsr['sr0']):.2f}; "
          f"DSR(N=1000) {dsr['dsr']:.3f}; effective trials {n_eff} -> DSR {dsr_eff}; passes 0.95 only if N <= {n_pass}")

    # ------------------------------------------------------------- PBO by CSCV
    pbo = pbo_cscv(R[1:], n_blocks=16)
    corr_is_oos = float(np.corrcoef(sr_is, sr_oos)[0, 1])
    top_decile = sr_is >= np.quantile(sr_is, 0.9)
    print(f"PBO {pbo['pbo']:.3f} over {pbo['splits']:,} splits; correlation between in-sample and out-of-sample Sharpe {corr_is_oos:.2f}")

    # ------------------------------------------------------------- calibration on pure noise
    sim_rng = np.random.default_rng(7)
    T_sim, N_sim, n_sims = 2520, 200, 300
    sim_rules = random_rules(N_sim, seed=3)
    naive_hits, dsr_hits, best_sr_sim, sr0_sim, dsr_sim = 0, 0, [], [], []
    for _ in range(n_sims):
        eps = stats.t.rvs(df=4, size=T_sim, random_state=sim_rng) * 0.01 / np.sqrt(2)
        price = 100 * np.exp(np.cumsum(eps))
        ret = np.r_[0.0, np.diff(np.log(price))]
        fac = RuleFactory(price)
        M = np.array([fac.returns(rule, ret, COST_BPS)[0][300:] for rule in sim_rules]).T
        s = sharpe(M)
        b = int(np.argmax(s))
        d_ = deflated_sharpe(M[:, b], s, N_sim)
        naive_hits += d_["psr_vs_zero"] > 0.95
        dsr_hits += d_["dsr"] > 0.95
        best_sr_sim.append(annualize(s[b])); sr0_sim.append(annualize(d_["sr0"])); dsr_sim.append(d_["dsr"])
    calib = {"simulations": n_sims, "days": T_sim, "rules": N_sim, "naive_false_positive_rate": naive_hits / n_sims,
             "dsr_false_positive_rate": dsr_hits / n_sims, "mean_best_sharpe": float(np.mean(best_sr_sim)),
             "mean_predicted_max_sharpe": float(np.mean(sr0_sim))}
    print("calibration on noise:", calib)

    # ------------------------------------------------------------- my own pre-registered strategy
    own_rule = {"family": "ma_crossover", "short": False, "fast": 1, "slow": 200}
    own_raw, own_pos = factory.returns(own_rule, r, COST_BPS)
    own_det, _ = factory.returns(own_rule, rd, COST_BPS)
    bh = r.copy()
    own = {}
    for lab, idx in (("in_sample", is_idx), ("out_of_sample", oos_idx)):
        x = own_raw[idx]; xd = own_det[idx]
        dd_own, dd_bh = drawdown(x).min(), drawdown(bh[idx]).min()
        own[lab] = {"sharpe": float(annualize(sharpe(x))), "buy_hold_sharpe": float(annualize(sharpe(bh[idx]))),
                    "annual_return": float(np.expm1(x.mean() * 252)), "buy_hold_annual_return": float(np.expm1(bh[idx].mean() * 252)),
                    "annual_vol": float(x.std() * np.sqrt(252)), "buy_hold_vol": float(bh[idx].std() * np.sqrt(252)),
                    "max_drawdown": float(dd_own), "buy_hold_max_drawdown": float(dd_bh),
                    "timing_sharpe": float(annualize(sharpe(xd))), "share_of_days_invested": float(np.r_[0, own_pos[:-1]][idx].mean())}
        dd_ = deflated_sharpe(xd, sr_is, 1)
        own[lab]["timing_psr"] = dd_["psr_vs_zero"]
        own[lab]["min_track_record_years"] = min_track_record_length(dd_["sharpe"], 0, dd_["skew"], dd_["kurtosis"]) / 252
    print("own strategy:", json.dumps(own, indent=1))

    # ------------------------------------------------------------- chart 1: 3D overfitting landscape (MA crossover grid)
    fasts = np.unique(np.round(np.geomspace(2, 100, 26)).astype(int))
    slows = np.unique(np.round(np.geomspace(20, 300, 26)).astype(int))
    Zi = np.full((len(slows), len(fasts)), np.nan); Zo = np.full_like(Zi, np.nan)
    for i, s_ in enumerate(slows):
        for j, f_ in enumerate(fasts):
            if f_ < s_:
                ret, _ = factory.returns({"family": "ma_crossover", "short": True, "fast": int(f_), "slow": int(s_)}, rd, COST_BPS)
                Zi[i, j] = annualize(sharpe(ret[is_idx])); Zo[i, j] = annualize(sharpe(ret[oos_idx]))
    pi, pj = np.unravel_index(np.nanargmax(Zi), Zi.shape)
    fig3d = go.Figure()
    fig3d.add_trace(go.Surface(x=np.log10(fasts), y=np.log10(slows), z=Zi, name="In-sample 1950-1999", showscale=False, opacity=0.95,
                               colorscale=[[0, "#10233b"], [0.5, BLUE], [1, "#d6e6fb"]], showlegend=True,
                               customdata=np.dstack(np.meshgrid(fasts, slows)),
                               hovertemplate="<b>In-sample</b><br>fast %{customdata[0]} / slow %{customdata[1]} days<br>Sharpe %{z:.2f}<extra></extra>"))
    fig3d.add_trace(go.Surface(x=np.log10(fasts), y=np.log10(slows), z=Zo, name="Out-of-sample 2000-2018", showscale=False, opacity=0.8,
                               colorscale=[[0, "#4a1d0f"], [0.5, ORANGE], [1, "#f6c3a6"]], showlegend=True,
                               customdata=np.dstack(np.meshgrid(fasts, slows)),
                               hovertemplate="<b>Out-of-sample</b><br>fast %{customdata[0]} / slow %{customdata[1]} days<br>Sharpe %{z:.2f}<extra></extra>"))
    fig3d.add_trace(go.Scatter3d(x=[np.log10(fasts[pj])] * 2, y=[np.log10(slows[pi])] * 2, z=[Zi[pi, pj], Zo[pi, pj]], mode="lines+markers",
                                 line=dict(color=TEXT, width=6), marker=dict(size=6, color=[BLUE, ORANGE]),
                                 name=f"Best in-sample ({fasts[pj]}/{slows[pi]}): {Zi[pi, pj]:.2f} -> {Zo[pi, pj]:.2f}"))
    ticks = lambda v: dict(tickvals=np.log10(v), ticktext=[str(t) for t in v])
    axis = dict(backgroundcolor=BG, gridcolor=GRID, zerolinecolor=MUTED, color=TEXT2)
    fig3d.update_layout(**plotly_layout(
        title=dict(text="The overfitting landscape: Sharpe of 300+ moving-average crossovers before (blue) and after (orange) 2000", x=0.02),
        scene=dict(xaxis=dict(title="Fast average (days)", **ticks(fasts[::5]), **axis), yaxis=dict(title="Slow average (days)", **ticks(slows[::5]), **axis),
                   zaxis=dict(title="Annualized timing Sharpe", **axis), camera=dict(eye=dict(x=1.6, y=-1.45, z=0.75))),
        legend=dict(font=dict(color=TEXT2), x=0.01, y=0.92), height=780))
    fig3d.write_html(OUT / "01_overfitting_landscape_3d.html", include_plotlyjs="cdn")
    try:
        fig3d.write_image(OUT / "01_overfitting_landscape_3d.png", width=1600, height=900, scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (static 3D export skipped: {exc.__class__.__name__})")

    # ------------------------------------------------------------- chart 2: histogram of Sharpes
    fig, ax = plt.subplots(figsize=(15, 6))
    bins = np.linspace(min(annualize(sr_is).min(), annualize(luck).min()) - 0.02, max(annualize(sr_is).max(), annualize(luck).max()) + 0.05, 70)
    ax.hist(annualize(luck), bins=bins, color=MUTED, alpha=0.65, label="1,000 coin-flip strategies (pure luck)")
    ax.hist(annualize(sr_is), bins=bins, color=BLUE, alpha=0.75, label="1,000 random technical rules")
    ax.axvline(annualize(sr_is[best]), color=AMBER, lw=2.2)
    ax.axvline(annualize(dsr["sr0"]), color=RED, lw=1.8, ls="--")
    ymax = ax.get_ylim()[1]
    ax.text(annualize(sr_is[best]) - 0.015, ymax * 0.9, f"Best rule: {annualize(sr_is[best]):.2f}\n{describe(best_rule)}", color=AMBER, fontsize=10, ha="right")
    ax.text(annualize(dsr["sr0"]) + 0.01, ymax * 0.62, f"Best Sharpe expected from\n1,000 worthless rules: {annualize(dsr['sr0']):.2f}", color=RED, fontsize=10)
    ax.set_xlabel("Annualized timing Sharpe ratio, 1950-1999 (detrended returns, after 5 bp costs)"); ax.set_ylabel("Strategies")
    ax.legend(frameon=False, labelcolor=TEXT2, loc="upper left"); ax.grid(axis="y", lw=0.5)
    title(ax, "Try 1,000 ideas and the best one looks brilliant, yet it falls short of the best Sharpe luck alone would be expected to produce")
    fig.tight_layout()
    fig.savefig(OUT / "02_sharpe_histogram.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 3: DSR vs number of trials + calibration
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    axes[0].semilogx(n_grid, dsr_curve, color=BLUE, lw=2.4)
    axes[0].axhline(0.95, color=MUTED, ls=":", lw=1.2)
    axes[0].text(1.1, 0.965, "95% confidence", color=MUTED, fontsize=9)
    for (k, v), col in zip(n_eff.items(), [AQUA, VIOLET, PINK]):
        axes[0].scatter([v], [dsr_eff[k]], color=col, s=60, zorder=5, label=f"Effective trials, {k}: N = {v}, DSR = {dsr_eff[k]:.2f}")
    axes[0].scatter([N_RULES], [dsr["dsr"]], color=RED, s=70, zorder=5, label=f"All 1,000 rules: DSR = {dsr['dsr']:.2f}")
    axes[0].scatter([1], [dsr["psr_vs_zero"]], color=AMBER, s=70, zorder=5, label=f"Ignoring the search (N = 1): {dsr['psr_vs_zero']:.4f}")
    axes[0].set_xlabel("Number of independent strategies tried (log scale)"); axes[0].set_ylabel("Deflated Sharpe Ratio")
    axes[0].set_ylim(0, 1.05); axes[0].legend(frameon=False, labelcolor=TEXT2, fontsize=8.5, loc="lower left"); axes[0].grid(lw=0.5)
    title(axes[0], f"The same Sharpe, judged by how many ideas were tried (passes only if N <= {n_pass})")
    cats = ["Naive test\n(PSR vs zero)", "Deflated Sharpe\nRatio"]
    vals = [100 * calib["naive_false_positive_rate"], 100 * calib["dsr_false_positive_rate"]]
    bars = axes[1].bar(cats, vals, color=[AMBER, AQUA], width=0.5)
    for b_, v in zip(bars, vals):
        axes[1].text(b_.get_x() + b_.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", color=TEXT, fontsize=13, fontweight="bold")
    axes[1].axhline(5, color=MUTED, ls=":", lw=1.2); axes[1].text(1.35, 6, "5% target", color=MUTED, fontsize=9)
    axes[1].set_ylabel("Share of worthless 'best strategies' declared significant (%)"); axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis="x", colors=TEXT, length=0); axes[1].grid(axis="y", lw=0.5)
    title(axes[1], f"Validation on {n_sims} fake markets with no edge (best of {N_sim} rules each)")
    fig.tight_layout()
    fig.savefig(OUT / "03_deflated_sharpe.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 4: in-sample vs out-of-sample
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    for fam, col in FAMILY_COLORS.items():
        m = np.array([ru["family"] == fam for ru in rules])
        axes[0].scatter(annualize(sr_is[m]), annualize(sr_oos[m]), s=11, color=col, alpha=0.7, lw=0, label=FAMILY_NAMES[fam])
    axes[0].scatter([annualize(sr_is[best])], [annualize(sr_oos[best])], s=160, facecolors="none", edgecolors=TEXT, lw=2, zorder=5)
    axes[0].annotate("best in-sample rule", (annualize(sr_is[best]), annualize(sr_oos[best])), xytext=(-150, -10), textcoords="offset points",
                     color=TEXT, arrowprops=dict(arrowstyle="-", color=TEXT))
    lim = [min(annualize(sr_is).min(), annualize(sr_oos).min()) - 0.05, max(annualize(sr_is).max(), annualize(sr_oos).max()) + 0.05]
    axes[0].plot(lim, lim, color=GRID, ls="--", lw=1)
    axes[0].axhline(0, color=MUTED, lw=0.8); axes[0].axvline(0, color=MUTED, lw=0.8)
    axes[0].set_xlabel("Sharpe 1950-1999 (in-sample)"); axes[0].set_ylabel("Sharpe 2000-2018 (out-of-sample)")
    axes[0].legend(frameon=False, labelcolor=TEXT2, loc="upper right", markerscale=2); axes[0].grid(lw=0.5)
    title(axes[0], f"What worked before 2000 tended to fail after it (correlation {corr_is_oos:.2f})")
    axes[1].hist(pbo["logits"], bins=60, color=BLUE, alpha=0.85)
    axes[1].axvline(0, color=RED, lw=1.8, ls="--")
    axes[1].text(0.97, 0.92, f"PBO = {pbo['pbo']:.0%}", ha="right", transform=axes[1].transAxes, color=TEXT, fontsize=16, fontweight="bold")
    axes[1].text(0.97, 0.84, f"of {pbo['splits']:,} train/test splits, the in-sample\nwinner ranked below the median out-of-sample", transform=axes[1].transAxes,
                 color=TEXT2, fontsize=9.5, va="top", ha="right")
    axes[1].set_xlabel("Logit of the winner's out-of-sample rank (left of 0 = below median)"); axes[1].set_ylabel("Splits"); axes[1].grid(axis="y", lw=0.5)
    title(axes[1], "Probability of Backtest Overfitting (CSCV, 16 blocks)")
    fig.tight_layout()
    fig.savefig(OUT / "04_in_sample_vs_out_of_sample.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 5: the best rule and my own strategy
    fig, axes = plt.subplots(2, 1, figsize=(16, 9.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    best_raw = R_raw[:, best]
    for series, lab, col, lw in [(bh, "Buy and hold S&P 500 (price index)", MUTED, 1.6), (own_raw, "My strategy: 200-day moving average, long or cash", AQUA, 2.0),
                                 (best_raw, f"Best of 1,000 random rules: {describe(best_rule)}", AMBER, 1.6)]:
        axes[0].semilogy(dates, np.exp(np.cumsum(series)), color=col, lw=lw, label=lab)
    axes[0].axvline(pd.Timestamp(SPLIT), color=TEXT2, ls="--", lw=1)
    axes[0].text(pd.Timestamp(SPLIT), axes[0].get_ylim()[1] * 0.5, "  out-of-sample →", color=TEXT2)
    axes[0].text(pd.Timestamp("1999-06-01"), axes[0].get_ylim()[1] * 0.5, "← selection period  ", color=TEXT2, ha="right")
    axes[0].set_ylabel("Growth of 1 (log scale, before dividends)"); axes[0].legend(frameon=False, labelcolor=TEXT2, loc="upper left"); axes[0].grid(lw=0.5)
    title(axes[0], "The best backtest collapses after it is chosen; a simple rule decided in advance keeps behaving as designed")
    axes[1].fill_between(dates, 100 * drawdown(bh), 0, color=MUTED, alpha=0.5, label="Buy and hold")
    axes[1].plot(dates, 100 * drawdown(own_raw), color=AQUA, lw=1.4, label="200-day moving average")
    axes[1].set_ylabel("Drawdown (%)"); axes[1].legend(frameon=False, labelcolor=TEXT2, loc="lower left"); axes[1].grid(lw=0.5)
    title(axes[1], f"Its real value is risk: maximum drawdown {100 * own['out_of_sample']['max_drawdown']:.0f}% vs {100 * own['out_of_sample']['buy_hold_max_drawdown']:.0f}% for the index after 2000")
    fig.tight_layout()
    fig.savefig(OUT / "05_best_rule_vs_own_strategy.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- metrics
    fam_stats = {FAMILY_NAMES[f]: {"rules": int(sum(ru["family"] == f for ru in rules)),
                                   "median_is_sharpe": float(np.median(annualize(sr_is[[ru["family"] == f for ru in rules]]))),
                                   "median_oos_sharpe": float(np.median(annualize(sr_oos[[ru["family"] == f for ru in rules]])))} for f in FAMILY_COLORS}
    metrics = {
        "data": {"series": "S&P 500 price index, daily close", "start": str(dates.iloc[0].date()), "end": str(dates.iloc[-1].date()),
                 "in_sample_days": len(is_idx), "out_of_sample_days": len(oos_idx), "cost_bps_per_trade": COST_BPS,
                 "buy_hold_sharpe_in_sample": float(annualize(sharpe(r[is_idx]))), "buy_hold_sharpe_out_of_sample": float(annualize(sharpe(r[oos_idx])))},
        "random_rules": {"n": N_RULES, "share_positive_in_sample": float((sr_is > 0).mean()),
                         "sharpe_percentiles_5_50_95": annualize(np.percentile(sr_is, [5, 50, 95])).round(3).tolist(),
                         "coin_flip_sharpe_percentiles_5_50_95": annualize(np.percentile(luck, [5, 50, 95])).round(3).tolist(),
                         "coin_flip_max": float(annualize(luck.max())), "by_family": fam_stats},
        "best_rule": {"rule": describe(best_rule), "in_sample_sharpe": float(annualize(sr_is[best])), "out_of_sample_sharpe": float(annualize(sr_oos[best])),
                      "out_of_sample_rank_percentile": float(stats.percentileofscore(sr_oos, sr_oos[best]))},
        "deflated_sharpe": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in dsr.items()},
        "effective_trials": n_eff, "dsr_with_effective_trials": dsr_eff, "max_trials_to_pass_95": n_pass,
        "pbo": {"value": pbo["pbo"], "splits": pbo["splits"], "median_logit": float(np.median(pbo["logits"])),
                "mean_is_sharpe_of_winner": float(annualize(pbo["is_sharpe"].mean())), "mean_oos_sharpe_of_winner": float(annualize(pbo["oos_sharpe"].mean()))},
        "is_oos_sharpe_correlation": corr_is_oos,
        "top_decile_in_sample": {"median_is": float(np.median(annualize(sr_is[top_decile]))), "median_oos": float(np.median(annualize(sr_oos[top_decile])))},
        "calibration_on_noise": calib, "own_strategy_200d_ma": own,
        "ma_grid_best": {"fast": int(fasts[pj]), "slow": int(slows[pi]), "in_sample": float(Zi[pi, pj]), "out_of_sample": float(Zo[pi, pj])},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # ------------------------------------------------------------- dashboard
    kpis = {"Best in-sample Sharpe": f"{annualize(sr_is[best]):.2f}", "Same rule after 2000": f"{annualize(sr_oos[best]):.2f}",
            "Deflated Sharpe Ratio": f"{dsr['dsr']:.2f}", "Probability of overfitting": f"{pbo['pbo']:.0%}"}
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in kpis.items())
    sec = lambda t, n_, f: f'<section><h2>{t}</h2><p class="note">{n_}</p>{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</section>'
    img = lambda t, n_, src: f'<section><h2>{t}</h2><p class="note">{n_}</p><img src="outputs/{src}" alt="{t}"></section>'
    sd_trials = float(annualize(np.std(sr_is, ddof=1)))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest Overfitting Detector</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>:root{{--bg:{BG};--card:{CARD};--text:{TEXT};--text2:{TEXT2};--muted:{MUTED};--line:{GRID}}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif}}
header,.kpis,footer{{max-width:1240px;margin:auto;padding:0 32px}}header{{padding-top:48px}}h1{{font-size:40px;margin:0 0 8px}}
header p{{color:var(--text2);max-width:880px;line-height:1.55;margin:0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:24px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.kpi span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.8px}}.kpi b{{display:block;font-size:24px;margin-top:6px}}
section{{max-width:1240px;margin:28px auto;padding:20px 24px;background:var(--card);border:1px solid var(--line);border-radius:14px}}
section img{{width:100%;border-radius:8px}}h2{{margin:4px 0 6px;font-size:20px}}.note{{color:var(--text2);margin:0 0 10px;font-size:14px}}
.calc{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}label{{display:grid;grid-template-columns:210px 1fr 70px;gap:10px;align-items:center;color:var(--text2);font-size:13px;margin:9px 0}}
input[type=range]{{width:100%;accent-color:{BLUE}}}output{{text-align:right;color:var(--text);font-variant-numeric:tabular-nums}}
.res{{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:18px}}.big{{font-size:52px;font-weight:800;margin:4px 0}}
.row{{display:flex;justify-content:space-between;color:var(--text2);font-size:14px;margin:6px 0}}.row b{{color:var(--text)}}.verdict{{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:600;font-size:13px}}
footer{{color:var(--muted);font-size:13px;margin:24px auto 48px}}a{{color:#7fb0f0}}
@media(max-width:820px){{.calc{{grid-template-columns:1fr}}header,.kpis,footer{{padding:0 16px}}section{{margin:16px;padding:12px}}h1{{font-size:30px}}label{{grid-template-columns:130px 1fr 56px}}}}</style></head><body>
<header><h1>Backtest Overfitting Detector</h1><p>Is the Sharpe ratio of the best strategy real, or the luck of trying 1,000 ideas? 1,000 random technical rules are backtested on 69 years
of the S&P 500 and the winner is judged with the Deflated Sharpe Ratio (Bailey and L&oacute;pez de Prado, 2014) and the Probability of Backtest Overfitting.</p></header>
<div class="kpis">{kpi_html}</div>
<section><h2>Deflated Sharpe Ratio calculator</h2><p class="note">Enter the backtest of any strategy. The defaults are this project's best rule. The calculator answers: given how many things you tried, what is the probability that the true Sharpe is above zero?</p>
<div class="calc"><div>
<label>Annualized Sharpe of the best strategy<input type="range" id="sr" min="0" max="3" step="0.01" value="{annualize(sr_is[best]):.2f}"><output id="sr_o"></output></label>
<label>Years of daily data<input type="range" id="yr" min="1" max="60" step="1" value="50"><output id="yr_o"></output></label>
<label>Strategies tried (N)<input type="range" id="n" min="0" max="4" step="0.01" value="3"><output id="n_o"></output></label>
<label>Std. dev. of the tried Sharpes (annual)<input type="range" id="sd" min="0.05" max="1.5" step="0.01" value="{sd_trials:.2f}"><output id="sd_o"></output></label>
<label>Skewness of returns<input type="range" id="sk" min="-3" max="3" step="0.1" value="{dsr['skew']:.1f}"><output id="sk_o"></output></label>
<label>Kurtosis of returns<input type="range" id="ku" min="3" max="60" step="0.5" value="{min(dsr['kurtosis'], 60):.1f}"><output id="ku_o"></output></label></div>
<div class="res"><div class="note">Deflated Sharpe Ratio</div><div class="big" id="dsr"></div><span class="verdict" id="verdict"></span>
<div class="row" style="margin-top:16px"><span>Best Sharpe expected from N worthless strategies</span><b id="sr0"></b></div>
<div class="row"><span>Naive probability (ignoring the search)</span><b id="psr"></b></div>
<div class="row"><span>Years needed to be 95% sure (vs. that hurdle)</span><b id="mtrl"></b></div></div></div></section>
{sec("The overfitting landscape in 3D", "Drag to rotate. Each point is a moving-average crossover rule. The blue peaks of 1950-1999 do not line up with the orange surface of 2000-2018: the white line shows where the best in-sample rule ended up.", fig3d)}
{img("1,000 backtests", "The distribution of Sharpe ratios from random rules and from coin flips.", "02_sharpe_histogram.png")}
{img("Deflated Sharpe Ratio", "How the verdict depends on the number of trials, and proof that the test works on markets with no edge.", "03_deflated_sharpe.png")}
{img("In-sample versus out-of-sample", "What happened to every rule after it would have been chosen.", "04_in_sample_vs_out_of_sample.png")}
{img("The winner versus a rule chosen in advance", "Equity curves and drawdowns.", "05_best_rule_vs_own_strategy.png")}
<footer>Tom&aacute;s Alarc&oacute;n &middot; <a href="https://github.com/tomasalarcon10/backtest-overfitting-detector">Source code</a></footer>
<script>
function erf(x){{const s=Math.sign(x);x=Math.abs(x);const t=1/(1+0.3275911*x);const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);return s*y}}
const Phi=x=>0.5*(1+erf(x/Math.SQRT2));
function invPhi(p){{let lo=-10,hi=10;for(let i=0;i<100;i++){{const m=(lo+hi)/2;if(Phi(m)<p)lo=m;else hi=m}}return (lo+hi)/2}}
const G=0.5772156649;
function calc(){{const v=id=>+document.getElementById(id).value,srA=v("sr"),yr=v("yr"),N=Math.round(Math.pow(10,v("n"))),sdA=v("sd"),sk=v("sk"),ku=v("ku");
const T=yr*252,sr=srA/Math.sqrt(252),sd=sdA/Math.sqrt(252);
const sr0=N<=1?0:sd*((1-G)*invPhi(1-1/N)+G*invPhi(1-1/(N*Math.E)));const den=Math.sqrt(1-sk*sr+(ku-1)/4*sr*sr);
const dsr=Phi((sr-sr0)*Math.sqrt(T-1)/den),ps=Phi(sr*Math.sqrt(T-1)/den);const mtrl=sr>sr0?(1+(1-sk*sr+(ku-1)/4*sr*sr)*Math.pow(1.6449/(sr-sr0),2))/252:Infinity;
const set=(id,t)=>document.getElementById(id).textContent=t;set("sr_o",srA.toFixed(2));set("yr_o",yr);set("n_o",N.toLocaleString("en-US"));set("sd_o",sdA.toFixed(2));set("sk_o",sk.toFixed(1));set("ku_o",ku.toFixed(1));
set("dsr",dsr.toFixed(3));set("sr0",(sr0*Math.sqrt(252)).toFixed(2));set("psr",ps.toFixed(4));set("mtrl",isFinite(mtrl)?mtrl.toFixed(1):"never");
const vd=document.getElementById("verdict"),ok=dsr>=0.95;vd.textContent=ok?"Likely a real edge (DSR ≥ 0.95)":"Cannot rule out luck";vd.style.background=ok?"{AQUA}33":"{RED}33";vd.style.color=ok?"{AQUA}":"{RED}"}}
document.querySelectorAll("input").forEach(e=>e.oninput=calc);calc();
</script></body></html>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("best_rule", "deflated_sharpe", "pbo", "top_decile_in_sample", "ma_grid_best")}, indent=1, default=float))


if __name__ == "__main__":
    main()
