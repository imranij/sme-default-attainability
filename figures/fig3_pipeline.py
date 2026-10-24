"""
Figure 3 -- Overview of the estimation pipeline.
The boxes use the terms the paper defines: GBDT fused with a sequence encoder, the
rate-constrained objective, DIR and the identified set, margin-based abstention with
review budget b, and SHAP reason codes with their stability check.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8.6,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

NAVY = "#1b2a4a"
BOXFACE = "#eef1f5"
BOXFACE2 = "#f7ece3"
EDGE = NAVY

fig, ax = plt.subplots(figsize=(5.396, 3.75))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")


def box(x, y, w, h, text, face=BOXFACE, edge=EDGE, fs=7.4, zorder=3):
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.35,rounding_size=2.2",
                        linewidth=1.1, edgecolor=edge, facecolor=face, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fs, color="#1a1a1a", zorder=zorder + 1, linespacing=1.3)
    return (x, y, w, h)


def arrow(b1, b2, side1="right", side2="left", color="#444444", lw=1.1,
          style="-|>", connector="arc3,rad=0.0"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts = {"right": (x1 + w1, y1 + h1 / 2), "left": (x1, y1 + h1 / 2),
           "top": (x1 + w1 / 2, y1 + h1), "bottom": (x1 + w1 / 2, y1)}
    pts2 = {"right": (x2 + w2, y2 + h2 / 2), "left": (x2, y2 + h2 / 2),
            "top": (x2 + w2 / 2, y2 + h2), "bottom": (x2 + w2 / 2, y2)}
    p1, p2 = pts[side1], pts2[side2]
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=9,
                         lw=lw, color=color, zorder=2, shrinkA=1, shrinkB=1,
                         connectionstyle=connector)
    ax.add_patch(a)


# Row 1: data source
b_data = box(2, 42.5, 19, 12, "raw data\n(public datasets\n+ SaudiSME-Sim)", face=BOXFACE2, fs=7.0)

# Row 1: feature split
b_static = box(26, 48, 18, 7.5, "static features\n(ratios, firmo-\ngraphics, sources $S$)", fs=6.6)
b_seq = box(26, 37.5, 18, 7.5, "monthly cash-flow\nsequences\n(where available)", fs=6.6)

arrow(b_data, b_static, connector="arc3,rad=-0.18")
arrow(b_data, b_seq, connector="arc3,rad=0.18")

# Row 1: learners
b_gbdt = box(49, 48, 15.5, 7.5, "GBDT\n(LightGBM)", fs=7.4)
b_seqenc = box(49, 37.5, 15.5, 7.5, "sequence\nencoder", fs=7.4)

arrow(b_static, b_gbdt)
arrow(b_seq, b_seqenc)

# fused score box
b_fuse = box(69.5, 42.5, 17, 12, "fused score $f$\n(sequence encoder\nfused with the\ntabular model)", face=BOXFACE2, fs=6.6)
arrow(b_gbdt, b_fuse, connector="arc3,rad=-0.18")
arrow(b_seqenc, b_fuse, connector="arc3,rad=0.18")

# objective box
b_obj = box(58, 29, 28, 9, "cross-entropy\nor rate-constrained\nobjective", face=BOXFACE2, fs=7.2)
arrow(b_fuse, b_obj, side1="bottom", side2="top")

# threshold box
b_thr = box(58, 18.5, 28, 7.5, "threshold at\noperating rate $a$", face=BOXFACE2, fs=7.2)
arrow(b_obj, b_thr, side1="bottom", side2="top")

# three downstream branches
b_dir = box(2, 3, 27, 13,
            "DIR $=1-\\mathrm{DI}(f,a)/\\mathrm{DI}(f_0,a)$\n"
            "+ identified set (Prop.~2)\n"
            "swap set, bootstrap CIs",
            fs=6.7)
b_abst = box(34, 3, 27, 13,
             "margin-based abstention\n$|f(x)-\\tau_a|$,\nreview budget $b$",
             fs=6.7)
b_shap = box(66, 3, 32, 13,
             "SHAP reason codes $\\to$\nstability (RBO, published\nindices) and fidelity",
             fs=6.7)

arrow(b_thr, b_dir, side1="bottom", side2="top", connector="arc3,rad=0.30")
arrow(b_thr, b_abst, side1="bottom", side2="top")
arrow(b_thr, b_shap, side1="bottom", side2="top", connector="arc3,rad=0.30")

fig.tight_layout()
fig.savefig("fig3_pipeline.pdf", bbox_inches="tight")
print("wrote fig3_pipeline.pdf")
