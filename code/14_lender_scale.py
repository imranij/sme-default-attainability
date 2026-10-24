"""How the Q4 observational cohort stands against what one lender books.

Section 7.5 asks for the observational pilot requirement "against the cohorts a single lender books
in a programme period". SBA records the originating lender (`Bank`) and the approval date, so the
six-month origination count per lender is directly observable. Computed on the matured analysis
population's approval years (1991-2010, the years with complete six-month windows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA, dump
import os

def main():
    df = pd.read_csv(os.path.join(DATA, "sba", "SBAnational.csv"),
                     usecols=["Bank", "ApprovalDate", "MIS_Status"], dtype="string")
    d = pd.to_datetime(df["ApprovalDate"], format="%d-%b-%y", errors="coerce")
    d = d.where(d < pd.Timestamp("2020-01-01"), d - pd.DateOffset(years=100))
    ok = d.notna() & df["Bank"].notna() & df["MIS_Status"].str.strip().isin(["CHGOFF", "P I F"])
    df, d = df[ok], d[ok]
    half = d.dt.year.astype(str) + "H" + ((d.dt.month > 6).astype(int) + 1).astype(str)
    cnt = df.assign(half=half).groupby(["Bank", "half"]).size()
    # keep windows in years with complete coverage
    yrs = d.dt.year
    keep_years = set(range(1991, 2011))
    cnt = cnt[[h for h in cnt.index.get_level_values("half")
               .map(lambda s: int(s[:4]) in keep_years)]]
    v = cnt.to_numpy()
    q = {f"p{p}": float(np.percentile(v, p)) for p in (50, 75, 90, 95, 99, 99.9)}
    need = 100_000                                  # Q4 observational cohort (results/Q4_power_summary.json)
    share_at_or_above = float((v >= need).mean())
    out = dict(source="SBA", n_lender_half_years=int(len(v)), n_lenders=int(cnt.index.get_level_values(0).nunique()),
               years="1991-2010", loans_per_lender_per_six_months=q,
               max_six_month_book=int(v.max()),
               required_observational_cohort=need,
               share_of_lender_half_years_at_or_above_requirement=share_at_or_above,
               ratio_requirement_to_median=need / max(q["p50"], 1.0),
               ratio_requirement_to_p99=need / max(q["p99"], 1.0))
    booked = int(round(0.7 * need))   # the pilot books the top a = 0.7 of its applicants
    out["booked_equivalent_at_a07"] = booked
    out["standing_phrase"] = (
        "beyond the six-month book of every lender in the SBA record: at $a = 0.7$ the design's "
        f"100{{,}}000 applicants imply about {booked:,} bookings in six months, against a largest "
        f"observed six-month book of {int(v.max()):,} loans, a 99th percentile of {q['p99']:.0f} "
        f"and a median of {q['p50']:.0f}")
    dump("SBA_lender_scale.json", out)
    print(out["standing_phrase"])


if __name__ == "__main__":
    main()
