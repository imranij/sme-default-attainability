"""
Figure 1 -- The attainability frontier (schematic).
Illustrates the paper's central device: an achieved default-inflow curve
lying above a ceiling curve, with the vertical gap at the operating
approval rate a=0.7 read as the "attainability gap" (Section 1 / Section 3,
sec:ceiling). This is a conceptual illustration, not measured data -- no
experiment has been run at the time of writing. No numbers here are drawn
from, or intended to match, any dataset or result in the paper.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 9.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

NAVY = "#1b2a4a"
SLATE = "#5b6b82"
ACCENT = "#c1652f"   # burnt orange -- survives grayscale as a mid-dark tone
LIGHT_GRID = "#d9dde3"

fig, ax = plt.subplots(figsize=(0.9 * 5.396, 3.45))

a = np.linspace(0.5, 0.9, 200)

# Illustrative, stylized curves only -- shapes chosen for a clear, readable
# schematic, not fitted to any data. Default inflow falls as approval rate
# falls (approve only the safest applicants at low a).
achieved = 0.145 - 0.145 * (a - 0.5) + 0.055 * (a - 0.5) ** 2
ceiling = 0.075 - 0.075 * (a - 0.5) + 0.02 * (a - 0.5) ** 2.2

ax.plot(a, achieved, color=NAVY, lw=2.4, solid_capstyle="round", zorder=3)
ax.plot(a, ceiling, color=ACCENT, lw=2.4, ls=(0, (6, 2.5)), solid_capstyle="round", zorder=3)

ax.fill_between(a, ceiling, achieved, color=SLATE, alpha=0.12, zorder=1)

# operating point
a_op = 0.7
y_ach = np.interp(a_op, a, achieved)
y_ceil = np.interp(a_op, a, ceiling)
ax.plot([a_op, a_op], [0, 0.16], color=LIGHT_GRID, lw=1.1, zorder=0)
ax.plot([a_op, a_op], [y_ceil, y_ach], color="#333333", lw=0.9, zorder=2)
ax.plot([a_op], [y_ach], marker="o", ms=4.5, color=NAVY, zorder=4)
ax.plot([a_op], [y_ceil], marker="o", ms=4.5, color=ACCENT, zorder=4)

arrow = FancyArrowPatch(
    (a_op + 0.052, y_ceil), (a_op + 0.052, y_ach),
    arrowstyle="-|>,head_width=3.2,head_length=5", mutation_scale=1.0,
    color="#333333", lw=0.9, zorder=2, shrinkA=0, shrinkB=0,
)
ax.plot([a_op, a_op + 0.052], [y_ach, y_ach], color="#333333", lw=0.6, ls=":", zorder=2)
ax.plot([a_op, a_op + 0.052], [y_ceil, y_ceil], color="#333333", lw=0.6, ls=":", zorder=2)
ax.add_patch(arrow)
ax.annotate(
    "attainability\ngap", xy=(a_op + 0.058, (y_ach + y_ceil) / 2),
    xytext=(a_op + 0.062, (y_ach + y_ceil) / 2),
    va="center", ha="left", fontsize=8.6, color="#222222",
)

ax.text(a_op, 0.164, r"operating rate $a=0.7$", ha="center", va="bottom",
        fontsize=8.4, color="#444444")

# labels on curves
ax.annotate("achieved $\\mathrm{DI}(f,a)$\n(a deployed model)", xy=(0.535, achieved[np.argmin(np.abs(a-0.535))]),
            xytext=(0.505, 0.148), fontsize=8.7, color=NAVY, ha="left", va="bottom")
ax.annotate("ceiling $\\mathrm{DI}^\\ast(S,a)$\n(Bayes-optimal ranker)", xy=(0.62, ceiling[np.argmin(np.abs(a-0.62))]),
            xytext=(0.60, 0.006), fontsize=8.7, color=ACCENT, ha="left", va="bottom")

ax.set_xlim(0.5, 0.9)
ax.set_ylim(0, 0.175)
ax.set_xlabel("approval rate $a$")
ax.set_ylabel("default inflow $\\mathrm{DI}(\\cdot,a)$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#888888")
ax.spines["bottom"].set_color("#888888")
ax.tick_params(colors="#444444")
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])

fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.text(0.02, 0.015, "schematic: not measured data; no experiment has been run",
          fontsize=7.6, style="italic", color="#666666", ha="left", va="bottom")
fig.savefig("fig1_frontier.pdf", bbox_inches="tight")
print("wrote fig1_frontier.pdf")
