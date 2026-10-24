# Machine learning, attainability bounds and data value for small and medium-sized enterprise default reduction

Pir Noman Ahmad and Imran, King Fahd University of Petroleum and Minerals, Dhahran, Saudi Arabia.

This repository holds the code and the result files behind the paper. The paper asks whether a
target of a 15% cut in new default inflow from lending to small and medium-sized enterprises can
be reached at a fixed approval rate, whether such a cut could be verified from accepted-loan data,
and which data sources would contribute to it. The scripts compute the ceiling on default inflow,
the identified set of the reduction and a Shapley value over data sources. They do this on two
public loan books (SBA National and Home Credit), on the Berka bank data and on SaudiSME-Sim, a
synthetic generator that ships with the code. Every number reported in the paper comes from a
file in `results/`, and `results/slot_table.md` lists each reported quantity with the file it was
read from.

## What is in each folder

| folder | contents |
|---|---|
| `code/` | the analysis scripts, numbered roughly in the order they were written, plus `common.py` (shared paths and helpers), `simfeat.py` (simulator features and selection mechanisms) and `fetch_data.sh` |
| `lib/src/` | the library the scripts import, namely the simulator (`data/saudisme_sim.py`), the default-inflow metric and its bootstrap (`eval/dir_metric.py`) and the rate-constrained objective (`models/dial_objective.py`) |
| `lib/tests/` | unit tests for the metric and the objective |
| `results/` | every JSON file the scripts wrote, together with `slot_table.md` and `slot_counts.json`, which map the reported numbers to these files |
| `figures/` | the figure PDFs, the three scripts that draw the schematic figures, and the outputs of the plotting scripts in `code/` |
| `data/` | empty apart from `data/README.md`, which says where each dataset comes from and where it must be placed |
| `patches/` | a record of what was changed in the scripts for this public copy |

## Setup

The code needs Python 3.11 or later, and the published results were produced with the package
versions pinned in `requirements.txt`, which install with

```
pip install -r requirements.txt
```

The Berka files are downloaded by `code/fetch_data.sh`, which also checks their SHA-256 sums.
SBA National and Home Credit have to be downloaded by hand. Home Credit is distributed under the
Kaggle competition rules and SBA National with the supplementary material of its article, so we
point to both sources and do not copy the files. They go in these places:

```
data/raw/sba/SBAnational.csv
data/raw/home_credit/application_train.csv
data/raw/home_credit/bureau.csv
data/raw/home_credit/bureau_balance.csv
data/raw/home_credit/installments_payments.csv
data/raw/home_credit/POS_CASH_balance.csv
```

`data/README.md` gives the source of each file and the SHA-256 sum of the copy we used. The
five Home Credit tables take about 1.8 GB on disk, and the Home Credit preparation step was
written to stay within 7 GB of memory. Scripts find the repository root from their own location, so they can be run from
any folder. Setting the environment variable `SME_ATTAIN_ROOT` points them at another copy.

## Reproducing the results

`run_all.sh` runs every script in the order the results depend on. A run without the two
hand-downloaded datasets is also possible, and it covers the simulator and Berka steps:

```
sh run_all.sh            # full run
sh run_all.sh --fast     # simulator and Berka steps only
```

The full order is given below with the run time of each step. The first time column is from the
original run on a cloud machine with two processor cores. The second is from the check described
in the next section, on a cloud machine with two cores and 7 GB of memory. A dash means the step
was not timed or not re-run.

| step | script | what it produces | original run | check |
|---|---|---|---|---|
| 1 | `fetch_data.sh` | Berka files, checked against SHA-256 sums | - | 4 s |
| 2 | `02_selection_mechanisms.py` | simulator side of the identification question, five selection mechanisms | 152 s | 3 s |
| 3 | `01_ceiling_validation.py` | ceiling-estimator validation on the simulator (Supplementary S4) | 542 s | 225 s |
| 4 | `04_berka_coverage_gate.py` | Berka demonstration and bootstrap coverage (Supplementary S5) | 297 s | 112 s |
| 5 | `03_pilot_power.py` | pilot power surfaces | - | under 1 s |
| 6 | `07_sba_prepare.py` | SBA National cohort, sources and split | - | - |
| 7 | `08_sba_q1.py` | SBA ceiling band and achieved inflow | 222 s | - |
| 8 | `09_sba_q2.py` | SBA identified set | 511 s | - |
| 9 | `12_sba_q3.py` | SBA Shapley values over three sources | - | - |
| 10 | `11_sba_components.py` | SBA objectives, ablations, abstention and fairness tables | 410 s | - |
| 11 | `10_gate_kendall.py` | sim-to-real gate, SBA arm | 225 s | - |
| 12 | `14_lender_scale.py` | origination volume per SBA lender | - | - |
| 13 | `20_s4_large_cells.py` | the two large ceiling cells on the simulator (Supplementary S4) | 163 s | 107 s |
| 14 | `15_hc_prepare.py` | Home Credit aggregation, sources and split | - | - |
| 15 | `16_hc_q1.py` | Home Credit ceiling band and achieved inflow | 544 s | - |
| 16 | `17_hc_q2.py` | Home Credit identified set | 170 s | - |
| 17 | `19_gate_hc.py` | sim-to-real gate on both real datasets, and its decision | 131 s | - |
| 18 | `18_hc_q3.py` | Home Credit Shapley values over four sources | - | - |
| 19 | `21_interaction_index.py` | Shapley-Taylor interaction index | - | 1 s |
| 20 | `05_figures.py`, `13_figures_sba.py`, `25_fig_shapley.py` | figures drawn from the result files | - | 4 s |

The timed steps of the original run add up to about an hour. The schematic figures are drawn
separately, by running `python3 fig1_frontier.py` and the other two scripts from inside `figures/`.
The unit tests run from inside `lib/` with `python3 -m pytest`.

Step 4 writes a placeholder `results/gate_status.json`, which steps 11 and 17 replace. The file
therefore only matches the committed copy once the full run has finished.

## What was checked by re-running

We unpacked the repository into an empty folder and ran the steps that do not need the two
hand-downloaded datasets, with Python 3.13, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, scikit-learn
1.9.1, LightGBM 4.7.0 and Matplotlib 3.11.2. The Berka download passed its checksum test on all
eight files. Each output was then compared with the committed file field by field.

The outputs of steps 3, 4, 5, 13 and 19 were identical to the committed files in every field.
This covers `S4_ceiling_cells.json`, `S4_ceiling_summary.json`, `S4_ceiling_large_cells.json`,
`berka_sequence_demo.json`, `S5_bootstrap_coverage.json`, `S5_bootstrap_summary.json`,
`Q4_power_summary.json`, `Q4_power_surfaces.json` and `interaction_indices.json`. In step 2,
`Q2_sim_summary.json` was identical, and in `Q2_sim_mechanisms.json` all simulator quantities
were identical. Two estimates in that file moved slightly, the Heckman-type correction by up to
1.04e-4 and the reweighting estimate by up to 2.4e-9, which we put down to the newer scikit-learn.
None of these changes moves an estimate across the edge of an identified set. The plotting
scripts redrew every data figure with the same text and numbers. The tick spacing of the two
schematic figures changed with the Matplotlib version.

Every script passed a Python syntax check, and the unit tests gave 13 passes and one skip. The
skipped test needs PyTorch, which is optional. Steps 6 to 12 and 14 to 18 were not re-run in this
check, because they need the SBA National and Home Credit files. Their outputs in `results/` are
the ones the original run wrote. All 79 quantities listed in `results/slot_table.md` were found
in the paper with the value the table gives.

## What is not included

The SBA National and Home Credit files are not included, for the reasons given under Setup, and
Berka is downloaded from its mirror by `fetch_data.sh`. The intermediate files in
`data/processed/` are not included either, since steps 6, 7, 14 and 15 rebuild them. The paper's
LaTeX source and our working notes are not part of the release.

Ten figures in the data section of the paper describe the three datasets as collected (for
example the rows per Home Credit table and the SBA loan vintages). They were drawn outside this
pipeline, so their scripts are not in this repository. These figures report descriptive
statistics of the raw files, and they are not among the quantities traced in
`results/slot_table.md`. One of the two versions of the pipeline schematic in the paper was also
made outside the pipeline, and the version in `figures/fig3_pipeline.py` is the other.

The design also names Taiwan and Polish bankruptcy datasets, but neither was used for any
reported number, so no code for them is included. Library modules that were planned but never
used by a reported result were left out as well.

No licence is attached to this repository, which means the code may be viewed and cited while
any reuse or redistribution needs the authors' permission.

## How to cite

Please cite both the paper and this archived copy of the code. The paper is

> Ahmad, P. N. and Imran. Machine learning, attainability bounds and data value for small and
> medium-sized enterprise default reduction. Working paper, King Fahd University of Petroleum and
> Minerals, 2026.

The archived copy of this repository is on Zenodo at
[10.5281/zenodo.23622378](https://doi.org/10.5281/zenodo.23622378), a DOI that always resolves to the
latest archived version. `CITATION.cff` holds the same details in a machine-readable form.
