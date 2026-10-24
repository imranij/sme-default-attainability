"""Draws fig_frontier.pdf and fig_identified.pdf, both real datasets.

Same serif style and palette as the three designed figures in figures/.
fig_frontier has an SBA row and a Home Credit row, and fig_identified has an SBA panel and
a Home Credit panel. If a dataset's results file is absent the figure is
drawn from whatever is present and the build script's caption is matched to it.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGS, RESULTS

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "font.size": 9.5, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
NAVY, TEAL, RUST, GREY = "#1b2a4a", "#2a7f7f", "#b5541f", "#8a8a8a"
PAL = [NAVY, TEAL, RUST, "#6a4c93", GREY]

ROWS = [
    ("SBA National", "SBA_Q1_cells.json",
     [("F", "firmographics"), ("F+G", "firmographics + guarantee"),
      ("F+G+B", "all three sources")]),
    ("Home Credit", "HC_Q1_cells.json",
     [("A", "application"), ("A+U", "application + bureau"),
      ("A+U+I+P", "all four sources")]),
]


def load(n):
    p = os.path.join(RESULTS, n)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_frontier():
    rows = [(nm, load(f), panels) for nm, f, panels in ROWS]
    rows = [r for r in rows if r[1]]
    if not rows:
        print("no Q1 cells; fig_frontier not drawn")
        return
    nr, nc = len(rows), 3
    fig, axes = plt.subplots(nr, nc, figsize=(7.8, 2.9 * nr), squeeze=False)
    for r, (dsname, cells, panels) in enumerate(rows):
        have = [(t, lab) for t, lab in panels if any(c["coalition"] == t for c in cells)]
        ylo, yhi = np.inf, -np.inf
        for k, (tag, lab) in enumerate(have):
            ax = axes[r][k]
            sub = sorted([c for c in cells if c["coalition"] == tag], key=lambda c: c["a"])
            a = [c["a"] for c in sub]
            lo = [100 * c["DI_star_lo"] for c in sub]
            hi = [100 * c["DI_star_hi"] for c in sub]
            pt = [100 * c["DI_star_point"] for c in sub]
            ach = [100 * c["DI_achieved"] for c in sub]
            di0 = [100 * c["DI0_tuned"] for c in sub]
            tgt = [0.85 * v for v in di0]
            ax.fill_between(a, lo, hi, color=TEAL, alpha=0.18, linewidth=0, label="ceiling band")
            ax.plot(a, pt, color=TEAL, lw=1.3, label=r"ceiling $\mathrm{DI}^\ast(S,a)$")
            ax.plot(a, ach, "o-", color=NAVY, lw=1.3, ms=3.2,
                    label=r"achieved $\mathrm{DI}(\hat f,a)$")
            ax.plot(a, di0, "s--", color=GREY, lw=1.1, ms=3.0, label="tuned incumbent")
            ax.plot(a, tgt, ":", color=RUST, lw=1.2, label=r"15\% reduction line")
            ax.set_xlabel("approval rate $a$")
            ax.set_title(f"{dsname}: {lab}", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
            ylo = min(ylo, min(lo + tgt)); yhi = max(yhi, max(hi + di0 + ach))
        pad = 0.05 * (yhi - ylo) if yhi > ylo else 1.0
        for k in range(len(have)):
            axes[r][k].set_ylim(ylo - pad, yhi + pad)
            if k:
                axes[r][k].set_yticklabels([])
        for k in range(len(have), nc):
            axes[r][k].axis("off")
        axes[r][0].set_ylabel("default inflow (\\%)")
        axes[r][0].legend(fontsize=6.6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_frontier.pdf"))
    print(f"wrote fig_frontier.pdf ({nr} dataset rows)")


def _panel_identified(ax, s, title):
    Mg = np.linspace(1, 4, 200)
    sw, mu, d0 = s["swap_share_a06"], s["mu_in_a06"], s["DI0_a06"]
    hi = 1 - mu / d0
    lo = 1 - mu * (1 + sw * (Mg - 1)) / d0
    ax.fill_between(Mg, lo, hi, color=NAVY, alpha=0.18, linewidth=0)
    ax.plot(Mg, lo, color=NAVY, lw=1.4, label="lower endpoint")
    ax.axhline(hi, color=NAVY, lw=1.0, ls="-")
    ax.axhline(0.15, color=RUST, lw=0.9, ls="--")
    ax.text(3.95, 0.152, "15\\% target", ha="right", va="bottom", fontsize=7.5, color=RUST)
    ms = s["M_star_015_a06"]
    ci = s["M_star_015_a06_ci"]
    inrange = 1 <= ms <= 4
    if inrange:
        ax.axvline(ms, color=TEAL, lw=1.1)
        ax.axvspan(max(1, ci[0]), min(4, ci[1]), color=TEAL, alpha=0.18, linewidth=0)
    sub = (f"$M^\\ast(0.15) = {ms:.2f}$" + ("" if inrange else " (off scale)")
           + f"\nbootstrap $[{ci[0]:.2f},\\,{ci[1]:.2f}]$")
    ax.set_title(title + "\n" + sub, fontsize=7.8)
    ax.set_xlabel("bounded risk ratio $M$")
    ax.spines[["top", "right"]].set_visible(False)


def fig_identified():
    sim = load("Q2_sim_mechanisms.json")
    sba = load("SBA_Q2_summary.json")
    hc = load("HC_Q2_summary.json")
    real = [(sba, "SBA National ($a=0.6$, $a_0=0.7$)"),
            (hc, "Home Credit ($a=0.6$, $a_0=0.7$)")]
    real = [(s, t) for s, t in real if s]
    npan = len(real) + 2
    fig, axes = plt.subplots(1, npan, figsize=(2.9 * npan, 3.2))
    axes = np.atleast_1d(axes)
    for i, (s, t) in enumerate(real):
        _panel_identified(axes[i], s, t)
    axes[0].set_ylabel("identified set for DIR")

    Mg = np.linspace(1, 4, 200)
    by = {}
    for r in (sim or []):
        by.setdefault(r["mechanism"], []).append(r)
    ax = axes[len(real)]
    for i, (mech, v) in enumerate(by.items()):
        s = np.mean([x["swap_share"] for x in v])
        mu = np.mean([x["mu_in"] for x in v])
        d0 = np.mean([x["DI0"] for x in v])
        hi = 1 - mu / d0
        lo = 1 - mu * (1 + s * (Mg - 1)) / d0
        ax.fill_between(Mg, lo, hi, alpha=0.14, color=PAL[i % 5], linewidth=0)
        ax.plot(Mg, lo, color=PAL[i % 5], lw=1.1, label=mech.replace("_", " "))
    ax.axhline(0.15, color=RUST, lw=0.9, ls="--")
    ax.set_xlabel("bounded risk ratio $M$")
    ax.set_title("Simulator, five selection mechanisms", fontsize=9)
    ax.legend(fontsize=6.4, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[len(real) + 1]
    names = list(by)
    trueM = [np.mean([x["true_M"] for x in by[m]]) for m in names]
    mstar = [np.mean([x["M_star_015"] for x in by[m]]) for m in names]
    for s, lab in ((sba, "SBA"), (hc, "Home Credit")):
        if s:
            names = names + [f"{lab} (true $M$)"]
            trueM = trueM + [s["true_M_a06"]]
            mstar = mstar + [s["M_star_015_a06"]]
    yy = np.arange(len(names))
    ax.barh(yy - 0.18, trueM, 0.34, color=NAVY, label="true $M$")
    ax.barh(yy + 0.18, np.maximum(mstar, 1e-3), 0.34, color=TEAL, label="$M^\\ast(0.15)$")
    ax.set_yticks(yy)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("risk ratio (log scale)")
    ax.set_title("True $M$ vs. $M^\\ast(0.15)$", fontsize=9)
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_identified.pdf"))
    print(f"wrote fig_identified.pdf ({npan} panels)")


if __name__ == "__main__":
    fig_frontier()
    fig_identified()
