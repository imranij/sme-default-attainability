"""Supplementary Figure: source-level value of information, drawn from the Q3 outputs.

Everything here is read from
results/SBA_Q3_shapley.json and results/HC_Q3_shapley.json, which 12_sba_q3.py and 18_hc_q3.py
wrote; nothing is computed. The simulator's seven-source set is not drawn because the sim-to-real
gate failed and the paper withholds it. Same style as 13_figures_sba.py.
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
                     "font.size": 8.5, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
NAVY, TEAL, RUST, GREY = "#1b2a4a", "#2a7f7f", "#b5541f", "#8a8a8a"

DATA = [("SBA National", "SBA_Q3_shapley.json"), ("Home Credit", "HC_Q3_shapley.json")]


def panel_values(ax, d, title):
    src = d["ranking_attainable"]                       # Shapley order, largest first
    att = np.array([d["shapley_attainable"][s] for s in src])
    ach = np.array([d["shapley_achieved"][s] for s in src])
    lo = np.array([d["shapley_attainable_ci"][s][0] for s in src])
    hi = np.array([d["shapley_attainable_ci"][s][1] for s in src])
    y = np.arange(len(src))
    ax.barh(y + 0.19, 100 * att, 0.36, color=NAVY, label=r"attainable $v^\ast$", zorder=3)
    ax.errorbar(100 * att, y + 0.19, xerr=[100 * (att - lo), 100 * (hi - att)], fmt="none",
                ecolor="white", elinewidth=1.4, capsize=0, zorder=4)
    ax.errorbar(100 * att, y + 0.19, xerr=[100 * (att - lo), 100 * (hi - att)], fmt="none",
                ecolor=GREY, elinewidth=0.8, capsize=2, zorder=5)
    ax.barh(y - 0.19, 100 * ach, 0.36, color=TEAL, alpha=0.85,
            label=r"achieved $v_{\mathrm{ach}}$", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(src); ax.invert_yaxis()
    ax.set_xlabel("Shapley share of the reduction (%)")
    ax.set_title(title, loc="left", color=NAVY)
    for yi, v in zip(y, att):
        ax.text(100 * v + 1.0, yi + 0.19, f"{100 * v:.1f}", va="center", fontsize=7, color=NAVY)
    ax.set_xlim(0, 100 * max(att.max(), ach.max()) * 1.22)


def panel_loso(ax, d, title):
    src = d["ranking_attainable"]
    sh_rank = {s: i + 1 for i, s in enumerate(src)}
    lo_rank = {s: i + 1 for i, s in enumerate(d["loso_ranking_attainable"])}
    differs = set(d.get("sources_whose_rank_differs", []))
    n = len(src)
    for s in src:
        col = RUST if s in differs else GREY
        ax.plot([0, 1], [sh_rank[s], lo_rank[s]], "-", color=col, lw=1.6 if s in differs else 1.0,
                zorder=2)
        ax.plot([0, 1], [sh_rank[s], lo_rank[s]], "o", color=col, ms=4.5, zorder=3)
        ax.text(-0.08, sh_rank[s], s, ha="right", va="center", fontsize=7.5, color=NAVY)
        ax.text(1.06, lo_rank[s], f"{100 * d['loso_attainable'][s]:.1f}%", ha="left",
                va="center", fontsize=7, color=col)
    ax.set_xlim(-0.75, 1.55); ax.set_ylim(n + 0.6, 0.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Shapley rank", "leave-one-out rank"])
    ax.set_yticks(range(1, n + 1)); ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
    ax.set_ylabel("")
    ax.set_title(title, loc="left", color=NAVY)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ds = {name: json.load(open(os.path.join(RESULTS, f))) for name, f in DATA
          if os.path.exists(os.path.join(RESULTS, f))}
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
    for row, (name, d) in enumerate(ds.items()):
        letter = "ab"[row], "cd"[row]
        panel_values(axes[row, 0], d, f"({letter[0]}) {name}: Shapley value by source")
        panel_loso(axes[row, 1], d,
                   f"({letter[1]}) {name}: leave-one-out rank")
        if row == 0:
            axes[row, 0].legend(frameon=False, fontsize=7, loc="lower right")
    for a in axes.ravel():
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(length=2.5, width=0.7)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_source_shapley.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out, "for", ", ".join(ds))


if __name__ == "__main__":
    main()
