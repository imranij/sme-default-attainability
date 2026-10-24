| # | slot | status | value | source file |
|---|---|---|---|---|
| 1 | `Kendall tau, SBA` | computed (Home Credit + SBA) | $-0.60$ | `results/gate_status.json` |
| 2 | `Kendall tau, Home Credit` | computed (Home Credit + SBA) | $-0.20$ | `results/gate_status.json` |
| 3 | `agreement among the top three` | computed (Home Credit + SBA) | 2 of 3 pairwise comparisons on SBA and 2 of 3 on Home Credit | `results/gate_status.json` |
| 4 | `admitted or withheld` | computed (Home Credit + SBA) | withheld: both $\tau$ values fall below the threshold and the sign condition fails on each real dataset, so the gate fails and the source-level analysis is confined to the real datasets | `results/gate_status.json` |
| 5 | `plug-in bias` | computed (simulator) | 0.01 percentage points | `results/S4_ceiling_cells.json` |
| 6 | `cells with a sign change` | computed (simulator) | 6 of the 9 sample-size by base-rate groups | `results/S4_ceiling_cells.json` |
| 7 | `band coverage` | computed (simulator) | 92\% | `results/S4_ceiling_cells.json` |
| 8 | `band width, SBA` | computed (simulator) | 0.39 percentage points | `results/S4_ceiling_large_cells.json` |
| 9 | `band width, Home Credit` | computed (simulator) | 0.26 percentage points | `results/S4_ceiling_large_cells.json` |
| 10 | `cells with an uninformative band` | computed (simulator) | 36 of 36 cells | `results/S4_ceiling_cells.json` |
| 11 | `the method used` | computed (simulator) | the paired bootstrap | `results/S5_bootstrap_summary.json` |
| 12 | `coverage of the chosen method` | computed (simulator) | 88\% | `results/S5_bootstrap_summary.json` |
| 13 | `datasets and rates without coverage` | computed (simulator) | the simulator cells $n = 600$, base rate 8\%; $n = 5000$, base rate 8\% | `results/S5_bootstrap_summary.json` |
| 14 | `gap, SBA` | computed (SBA) | 0.6 percentage points | `results/SBA_Q1_summary.json` |
| 15 | `gap, Home Credit` | computed (Home Credit) | $-0.1$ percentage points | `results/HC_Q1_summary.json` |
| 16 | `band, SBA` | computed (SBA) | $[20.1,\,21.0]$ percentage points | `results/SBA_Q1_summary.json` |
| 17 | `band, Home Credit` | computed (Home Credit) | $[2.6,\,3.8]$ percentage points | `results/HC_Q1_summary.json` |
| 18 | `gap at a = 0.5, SBA` | computed (SBA) | 1.7 percentage points | `results/SBA_Q1_summary.json` |
| 19 | `gap at a = 0.5, Home Credit` | computed (Home Credit) | $-0.1$ percentage points | `results/HC_Q1_summary.json` |
| 20 | `attainable reduction, SBA` | computed (SBA) | 9.3\% | `results/SBA_Q1_summary.json` |
| 21 | `attainable reduction, Home Credit` | computed (Home Credit) | 9.7\% | `results/HC_Q1_summary.json` |
| 22 | `configurations that clear the line` | computed (Home Credit + SBA) | none of the seven source configurations on SBA and none of the fifteen source configurations on Home Credit | `results/SBA_Q1_summary.json` + `results/HC_Q1_summary.json` |
| 23 | `attainable reduction, untuned incumbent` | computed (Home Credit + SBA) | 40.9\% on SBA and 19.9\% on Home Credit | `results/SBA_Q1_summary.json` + `results/HC_Q1_summary.json` |
| 24 | `difference between the two incumbents` | computed (Home Credit + SBA) | 31.6 percentage points on SBA and 10.3 percentage points on Home Credit | `results/SBA_Q1_summary.json` + `results/HC_Q1_summary.json` |
| 25 | `model or data, by configuration` | computed (Home Credit + SBA) | the data on every configuration of both datasets -- no source set on either has a ceiling that clears the 15\% line | `results/SBA_Q1_summary.json` + `results/HC_Q1_summary.json` |
| 26 | `swap share, SBA` | computed (SBA) | 3.0\% | `results/SBA_Q2_summary.json` |
| 27 | `swap share, Home Credit` | computed (Home Credit) | 3.4\% | `results/HC_Q2_summary.json` |
| 28 | `mu-in, SBA` | computed (SBA) | 9.6\% | `results/SBA_Q2_summary.json` |
| 29 | `mu-in, Home Credit` | computed (Home Credit) | 2.9\% | `results/HC_Q2_summary.json` |
| 30 | `width per unit of M, SBA` | computed (SBA) | 2.2 percentage points of $\DIR$ | `results/SBA_Q2_summary.json` |
| 31 | `width per unit of M, Home Credit` | computed (Home Credit) | 3.0 percentage points | `results/HC_Q2_summary.json` |
| 32 | `M-star, SBA` | computed (SBA) | 6.47 | `results/SBA_Q2_summary.json` |
| 33 | `M-star, Home Credit` | computed (Home Credit) | 0.20 | `results/HC_Q2_summary.json` |
| 34 | `M-star interval, SBA` | computed (SBA) | $[6.12,\,6.79]$ | `results/SBA_Q2_summary.json` |
| 35 | `M-star interval, Home Credit` | computed (Home Credit) | $[-0.64,\,1.34]$ | `results/HC_Q2_summary.json` |
| 36 | `swap share at a = 0.7` | computed (Home Credit + SBA) | 8.9\% on SBA and 7.7\% on Home Credit | `results/SBA_Q2_summary.json` + `results/HC_Q2_summary.json` |
| 37 | `M-star at a = 0.7` | computed (Home Credit + SBA) | 3.53 on SBA and 1.40 on Home Credit | `results/SBA_Q2_summary.json` + `results/HC_Q2_summary.json` |
| 38 | `smallest true M` | computed (simulator) | 24.1 | `results/Q2_sim_summary.json` |
| 39 | `largest true M` | computed (simulator) | 1585.7 | `results/Q2_sim_summary.json` |
| 40 | `mechanisms satisfying the lower bound` | computed (simulator) | all five mechanisms | `results/Q2_sim_summary.json` |
| 41 | `top source, Home Credit` | computed (Home Credit) | application | `results/HC_Q3_shapley.json` |
| 42 | `top source value, Home Credit` | computed (Home Credit) | 57.3\% | `results/HC_Q3_shapley.json` |
| 43 | `top source, SBA` | computed (SBA) | guarantee | `results/SBA_Q3_shapley.json` |
| 44 | `top source value, SBA` | computed (SBA) | 69.3\% | `results/SBA_Q3_shapley.json` |
| 45 | `top-rank probability, Home Credit` | computed (Home Credit) | 100\% | `results/HC_Q3_shapley.json` |
| 46 | `top-rank probability, SBA` | computed (SBA) | 100\% | `results/SBA_Q3_shapley.json` |
| 47 | `ranks that reshuffle` | computed (Home Credit + SBA) | no rank on SBA and no rank on Home Credit | `results/SBA_Q3_shapley.json` + `results/HC_Q3_shapley.json` |
| 48 | `ranks that hold across draws` | computed (Home Credit + SBA) | the first, second and third ranks on SBA and the first, second, third and fourth ranks on Home Credit | `results/SBA_Q3_shapley.json` + `results/HC_Q3_shapley.json` |
| 49 | `achieved-function ordering` | computed (Home Credit + SBA) | guarantee, bank, firmographics on SBA and application, bureau, instalments, POS-cash on Home Credit | `results/SBA_Q3_shapley.json` + `results/HC_Q3_shapley.json` |
| 50 | `distance between attainable and achieved` | computed (Home Credit + SBA) | 2.3 percentage points on SBA and 4.3 percentage points on Home Credit | `results/SBA_Q3_shapley.json` + `results/HC_Q3_shapley.json` |
| 51 | `largest interacting pair` | computed (Home Credit + SBA) | guarantee and bank on SBA and application and bureau on Home Credit | `results/interaction_indices.json` |
| 52 | `largest interaction` | computed (Home Credit + SBA) | $-24.2$ percentage points and $-42.2$ percentage points respectively | `results/interaction_indices.json` |
| 53 | `sources whose rank differs` | computed (Home Credit + SBA) | no source on SBA and bureau, instalments and POS-cash on Home Credit | `results/SBA_Q3_shapley.json` + `results/HC_Q3_shapley.json` |
| 54 | `simulator source ranking` | computed (Home Credit + SBA) | not reported: the gate fails, so under the pre-specified rule the source-level analysis is confined to the real datasets | `results/gate_status.json` |
| 55 | `cohort size, observational` | computed (simulator) | 100,000 | `results/Q4_power_summary.json` |
| 56 | `duration, observational` | computed (simulator) | 6 | `results/Q4_power_summary.json` |
| 57 | `cohort size, randomised band` | computed (simulator) | 50,000 | `results/Q4_power_summary.json` |
| 58 | `duration, randomised band` | computed (simulator) | 6 | `results/Q4_power_summary.json` |
| 59 | `randomisation share` | computed (simulator) | 10\% | `results/Q4_power_summary.json` |
| 60 | `cohort size, full randomisation` | computed (simulator) | 10,000 | `results/Q4_power_summary.json` |
| 61 | `duration, full randomisation` | computed (simulator) | 6 | `results/Q4_power_summary.json` |
| 62 | `cohort saving from the band` | computed (simulator) | 50\% | `results/Q4_power_summary.json` |
| 63 | `standing of the observational cohort` | computed (SBA) | beyond the six-month book of every lender in the SBA record: at $a = 0.7$ the design's 100{,}000 applicants imply about 70,000 bookings in six months, against a largest observed six-month book of 6,893 loans, a 99th percentile of 179 and a median of 2 | `results/SBA_lender_scale.json` |
| 64 | `difference between objectives` | computed (SBA) | 3.8 percentage points | `results/SBA_components.json` |
| 65 | `minimum detectable difference` | computed (SBA) | 0.4 percentage points | `results/SBA_components.json` |
| 66 | `powered or underpowered` | computed (SBA) | powered | `results/SBA_components.json` |
| 67 | `best kappa and lambda` | computed (SBA) | $\kappa = 0.5$, $\lambda = 0.5$ | `results/SBA_kappa_lambda_sweep.json` |
| 68 | `omitted gradient mass` | computed (SBA) | 0.0024\% | `results/SBA_components.json` |
| 69 | `consistently or not` | computed (SBA) | inconsistently | `results/SBA_auc_dir.json` |
| 70 | `reversing pairs` | computed (SBA) | 5 of 140 | `results/SBA_auc_dir.json` |
| 71 | `stability across bootstrap refits` | computed (SBA) | 0.85 | `results/SBA_components.json` |
| 72 | `stability across out-of-time refits` | computed (SBA) | 0.51 | `results/SBA_components.json` |
| 73 | `stability shift under the rate-constrained objective` | computed (SBA) | $-0.09$ in rank-biased overlap | `results/SBA_components.json` |
| 74 | `selective-prediction gain` | computed (SBA) | 2.80 percentage points of default inflow | `results/SBA_components.json` |
| 75 | `over-represented groups in the routed set` | computed (SBA) | no group by more than 1.5 times its share, the largest excesses falling on individual borrower states | `results/SBA_fairness.json` |
| 76 | `approval-rate gaps` | computed (SBA) | 29.1, 43.7 and 18.8 percentage points respectively | `results/SBA_fairness.json` |
| 77 | `default-rate gaps` | computed (SBA) | 17.1, 31.6 and 5.1 percentage points respectively | `results/SBA_fairness.json` |
| 78 | `mechanisms where a standard method fails` | computed (simulator) | (sentence recast to per-method sets: MAR / Heckman / fuzzy) | `results/Q2_sim_summary.json` |
| 79 | `sources with a negative value` | computed (SBA) | (sentence recast: no source takes a negative value) | `results/SBA_Q3_shapley.json` |