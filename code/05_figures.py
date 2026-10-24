"""Figures, in the serif style of the three designed figures.

fig_identified_sim_only.pdf
                     : the identified set of Proposition 2 against M, drawn from the
                       SIMULATOR's five selection mechanisms alone. It is not in the
                       paper. 13_figures_sba.py draws the figure the paper uses
                       (fig_identified.pdf), whose third and fourth panels are exactly
                       this figure's two panels, alongside an SBA and a Home Credit
                       panel. This script writes the simulator-only version under its
                       own name so that run order cannot overwrite fig_identified.pdf.
figS4_ceiling.pdf    : estimator validation, bias and band width by cell (simulator).
figS_power.pdf       : the three pilot-power surfaces (simulator).
fig_frontier.pdf     : NOT produced here -- it needs achieved inflow and a ceiling band
                       on the real datasets, so 13_figures_sba.py draws it.
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


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_identified_sim_only():
    rows = load("Q2_sim_mechanisms.json")
    if not rows:
        return
    by = {}
    for r in rows:
        by.setdefault(r["mechanism"], []).append(r)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    Mg = np.linspace(1, 4, 100)
    ax = axes[0]
    for i, (mech, v) in enumerate(by.items()):
        s = np.mean([x["swap_share"] for x in v])
        mu = np.mean([x["mu_in"] for x in v])
        d0 = np.mean([x["DI0"] for x in v])
        hi = 1 - mu / d0
        lo = 1 - mu * (1 + s * (Mg - 1)) / d0
        ax.fill_between(Mg, lo, hi, alpha=0.16, color=PAL[i % 5], linewidth=0)
        ax.plot(Mg, lo, color=PAL[i % 5], lw=1.2, label=mech.replace("_", " "))
    ax.axhline(0.15, color="k", lw=0.8, ls="--")
    ax.text(3.95, 0.155, "15\\% target", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("bounded risk ratio $M$"); ax.set_ylabel("identified set for DIR")
    ax.set_title("Identified set vs $M$ (simulator, $a=0.6$, $a_0=0.7$)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    names = list(by)
    trueM = [np.mean([x["true_M"] for x in by[m]]) for m in names]
    mstar = [np.mean([x["M_star_015"] for x in by[m]]) for m in names]
    yy = np.arange(len(names))
    ax.barh(yy - 0.18, trueM, 0.34, color=NAVY, label="true $M$")
    ax.barh(yy + 0.18, mstar, 0.34, color=TEAL, label="$M^\\ast(0.15)$")
    ax.set_yticks(yy); ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlabel("risk ratio (log scale)")
    ax.set_title("True $M$ against the verifiability threshold", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    # NOT fig_identified.pdf: that name belongs to 13_figures_sba.py, which supersedes
    # this figure with the two-dataset version the paper prints.
    fig.savefig(os.path.join(FIGS, "fig_identified_sim_only.pdf"))
    print("wrote fig_identified_sim_only.pdf")


def fig_ceiling():
    cells = load("S4_ceiling_cells.json")
    if not cells:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ns = sorted({c["n"] for c in cells})
    brs = sorted({c["base_rate"] for c in cells})
    for i, br in enumerate(brs):
        b = [np.mean([c["bias_pp"] for c in cells if c["n"] == n and c["base_rate"] == br]) for n in ns]
        w = [np.mean([c["band_width_pp"] for c in cells if c["n"] == n and c["base_rate"] == br]) for n in ns]
        axes[0].plot(ns, b, "o-", color=PAL[i], lw=1.2, ms=3.5, label=f"base rate {br:.0%}")
        axes[1].plot(ns, w, "o-", color=PAL[i], lw=1.2, ms=3.5, label=f"base rate {br:.0%}")
    axes[0].axhline(0, color="k", lw=0.7)
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    axes[0].set_xlabel("$n$"); axes[0].set_ylabel("bias of the point estimator (pp)")
    axes[1].set_xlabel("$n$"); axes[1].set_ylabel("ceiling-band width (pp)")
    axes[0].set_title("Bias against the known ceiling", fontsize=9)
    axes[1].set_title("Band width", fontsize=9)
    for a in axes:
        a.legend(fontsize=7.5, frameon=False); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "figS4_ceiling_validation.pdf"))
    print("wrote figS4_ceiling_validation.pdf")


def fig_power():
    rows = load("Q4_power_surfaces.json")
    if not rows:
        return
    designs = ["observational", "randomised_band", "full_randomisation"]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9), sharey=True)
    for k, d in enumerate(designs):
        sub = [r for r in rows if r["design"] == d]
        ns = sorted({r["n"] for r in sub}); ms = sorted({r["months"] for r in sub})
        for i, m in enumerate(ms):
            p = [next(r["power"] for r in sub if r["n"] == n and r["months"] == m) for n in ns]
            axes[k].plot(ns, p, "o-", color=PAL[i], lw=1.2, ms=3, label=f"{m} months")
        axes[k].axhline(0.8, color="k", ls="--", lw=0.8)
        axes[k].set_xscale("log"); axes[k].set_xlabel("cohort size $n$")
        axes[k].set_title(d.replace("_", " "), fontsize=9)
        axes[k].spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("power at a true 15\\% reduction")
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "figS_pilot_power.pdf"))
    print("wrote figS_pilot_power.pdf")


if __name__ == "__main__":
    fig_identified_sim_only(); fig_ceiling(); fig_power()
