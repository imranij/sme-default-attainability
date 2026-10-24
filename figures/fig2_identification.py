"""
Figure 2 -- Partial identification of the default-inflow reduction (schematic).
Left panel: incumbent's approved set A0 and new model's approved set A as
two overlapping regions, with the identified overlap (mu_in) and the
unidentified swap-in region (mu_out, bounded by the ratio M of
Assumption 1). Right panel: how the identified interval for DIR narrows as
M falls toward 1 and widens as the swap share s grows, at hypothetical
illustrative values of s and M -- not estimates from any dataset.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

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
ACCENT = "#c1652f"
LIGHT = "#eef0f3"

fig = plt.figure(figsize=(0.95 * 5.396, 3.85))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.42)
axL = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1])

# ---------- Left panel: Venn-style set diagram ----------
axL.set_xlim(0, 10)
axL.set_ylim(0, 10)
axL.axis("off")
axL.set_title("approved sets", fontsize=9.5, pad=14)

e0 = Ellipse((4.0, 5.0), width=6.4, height=6.0, facecolor=SLATE, alpha=0.30,
             edgecolor=NAVY, lw=1.4, zorder=2)
e1 = Ellipse((6.2, 5.0), width=6.0, height=5.6, facecolor=ACCENT, alpha=0.28,
             edgecolor=ACCENT, lw=1.4, zorder=2)
axL.add_patch(e0)
axL.add_patch(e1)

axL.text(1.9, 8.1, r"$A_0$ (incumbent)", fontsize=8.6, color=NAVY, ha="center")
axL.text(8.3, 8.0, r"$A$ (new model)", fontsize=8.6, color=ACCENT, ha="center")

axL.text(3.7, 5.0, r"$A \cap A_0$" "\nidentified\n" r"$\mu_{\mathrm{in}}$",
          fontsize=8.3, color="#1a1a1a", ha="center", va="center", zorder=3)
axL.text(7.7, 5.0, r"$A \setminus A_0$" "\nunidentified\n" r"$\mu_{\mathrm{out}} \leq M\mu_{\mathrm{in}}$",
          fontsize=8.0, color="#1a1a1a", ha="center", va="center", zorder=3)
axL.text(1.3, 3.2, r"$A_0 \setminus A$", fontsize=7.6, color="#333333", ha="center", va="center")

axL.text(4.9, 0.9, "overlap width $\\propto (1-s)$;  right lobe width $\\propto s$",
          fontsize=7.3, color="#555555", ha="center", style="italic")

# ---------- Right panel: identified interval as a function of M ----------
DI0 = 1.0
mu_in = 0.62  # illustrative, hypothetical value only

def interval(M, s):
    lo = 1 - mu_in * (1 + s * (M - 1)) / DI0
    hi = 1 - mu_in / DI0
    return lo, hi

M_grid = np.linspace(1, 4, 200)
s_values = [0.10, 0.20, 0.30]
s_colors = [NAVY, "#6b4fa0", ACCENT]

for s, col in zip(s_values, s_colors):
    lo = [interval(M, s)[0] for M in M_grid]
    hi = interval(M_grid[0], s)[1]
    axR.plot(M_grid, lo, color=col, lw=1.9, label=f"$s={s:.2f}$")
axR.axhline(interval(1, 0.2)[1], color="#888888", lw=1.0, ls=(0, (4, 2)))
axR.text(3.85, interval(1, 0.2)[1] + 0.012, "upper endpoint $1-\\mu_{\\mathrm{in}}/\\mathrm{DI}_0$\n(does not depend on $M$)",
          fontsize=7.0, color="#555555", ha="right", va="bottom")

axR.axvline(1.0, color="#bbbbbb", lw=1.0)
axR.set_xlim(1, 4)
ylo_all = min(interval(4, 0.30)[0], interval(4, 0.10)[0]) - 0.05
axR.set_ylim(ylo_all, interval(1, 0.1)[1] + 0.10)

axR.set_xlabel("risk ratio $M$")
axR.set_ylabel("identified interval for $\\mathrm{DIR}(f,a\\mid f_0)$")
axR.set_title("identified interval width", fontsize=9.5, pad=14)
axR.legend(loc="lower left", fontsize=7.6, frameon=False, handlelength=1.6)
axR.spines["top"].set_visible(False)
axR.spines["right"].set_visible(False)
axR.spines["left"].set_color("#888888")
axR.spines["bottom"].set_color("#888888")
axR.tick_params(colors="#444444", labelsize=8)

fig.tight_layout(rect=[0, 0.0, 1, 0.98])
fig.savefig("fig2_identification.pdf", bbox_inches="tight")
print("wrote fig2_identification.pdf")
