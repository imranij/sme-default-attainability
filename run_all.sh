#!/bin/sh
# Runs every analysis in the order the results depend on, then redraws the figures.
# Usage:  sh run_all.sh            the full run (needs the SBA National and Home Credit files)
#         sh run_all.sh --fast     the simulator and Berka steps only (no hand-downloaded data)
# Paths are relative to this folder. Seeds are fixed inside each script.
set -e
cd "$(dirname "$0")/code"
MODE="${1:---all}"
run() { echo "== $1"; python3 -W ignore "$1"; }

sh fetch_data.sh                 # Berka files, checked against SHA-256 sums
run 02_selection_mechanisms.py   # simulator: five selection mechanisms (Q2, simulator side)
run 01_ceiling_validation.py     # simulator: ceiling-estimator validation (Supplementary S4)
run 04_berka_coverage_gate.py    # Berka demonstration, bootstrap coverage (S5), gate placeholder
run 03_pilot_power.py            # Q4 pilot power surfaces (reads the output of 02)

if [ "$MODE" = "--fast" ]; then
  run 05_figures.py
  echo "Fast run finished. The SBA National and Home Credit steps were skipped."
  exit 0
fi

run 07_sba_prepare.py            # SBA National: label, maturity filter, sources, split
run 08_sba_q1.py                 # SBA: Q1 ceiling band and achieved inflow; coalition fits
run 09_sba_q2.py                 # SBA: Q2 identified set and coverage cell
run 12_sba_q3.py                 # SBA: Q3 source-level Shapley values (reads 08)
run 11_sba_components.py         # SBA: objectives, ablations, abstention, fairness tables
run 10_gate_kendall.py           # sim-to-real gate, SBA arm and simulator arm
run 14_lender_scale.py           # SBA: origination volume per lender
run 20_s4_large_cells.py         # simulator: the two large ceiling cells (S4)
run 15_hc_prepare.py             # Home Credit: aggregation of child tables, sources, split
run 16_hc_q1.py                  # Home Credit: Q1 ceiling band and achieved inflow
run 17_hc_q2.py                  # Home Credit: Q2 identified set and coverage cell
run 19_gate_hc.py                # sim-to-real gate, both arms, and the decision
run 18_hc_q3.py                  # Home Credit: Q3 Shapley values over four sources
run 21_interaction_index.py      # Shapley-Taylor interaction index
run 05_figures.py                # simulator and supplement figures
run 13_figures_sba.py            # fig_frontier.pdf and fig_identified.pdf
run 25_fig_shapley.py            # supplementary Shapley figure
echo "Full run finished. Results are in results/, figures in figures/."
