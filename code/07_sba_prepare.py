"""SBA National: load, maturity filter, source grouping, out-of-time split.

The column-to-source grouping defined below was fixed before any model was fitted.
Outputs data/processed/sba.npz (features, label, split, group columns) and
results/SBA_cohort.json (row accounting and the censoring rate of excluded cohorts).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from common import DATA, RESULTS, dump

from common import PROC  # the repository root is resolved once, in common.py
os.makedirs(PROC, exist_ok=True)

STR = "string"
DTYPES = {
    "LoanNr_ChkDgt": STR, "Name": STR, "City": STR, "State": STR, "Zip": STR,
    "Bank": STR, "BankState": STR, "NAICS": STR, "ApprovalDate": STR,
    "ApprovalFY": STR, "Term": "int32", "NoEmp": "int32", "NewExist": STR,
    "CreateJob": "int32", "RetainedJob": "int32", "FranchiseCode": STR,
    "UrbanRural": STR, "RevLineCr": STR, "LowDoc": STR, "ChgOffDate": STR,
    "DisbursementDate": STR, "DisbursementGross": STR, "BalanceGross": STR,
    "MIS_Status": STR, "ChgOffPrinGr": STR, "GrAppv": STR, "SBA_Appv": STR,
}

# --- the fixed grouping ------------------------------------------------------
SOURCES = ["firmographics", "guarantee", "bank"]
SRC_LABEL = {"firmographics": "firmographics", "guarantee": "guarantee", "bank": "bank"}


def money(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(r"[$,\s]", "", regex=True), errors="coerce")


def dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%d-%b-%y", errors="coerce")


def fix_century(d: pd.Series) -> pd.Series:
    """%y maps 62 -> 2062; the extract spans 1962-2014, so push future dates back 100y."""
    cutoff = pd.Timestamp("2020-01-01")
    return d.where(d < cutoff, d - pd.DateOffset(years=100))


def main():
    path = os.path.join(DATA, "sba", "SBAnational.csv")
    df = pd.read_csv(path, dtype=DTYPES, low_memory=False)
    n_raw = len(df)
    acct = {"rows_raw": n_raw, "cols_raw": df.shape[1]}

    df["ApprovalDate"] = fix_century(dates(df["ApprovalDate"]))
    df["DisbursementDate"] = fix_century(dates(df["DisbursementDate"]))
    df["ChgOffDate"] = fix_century(dates(df["ChgOffDate"]))
    extract = max(df["ApprovalDate"].max(), df["DisbursementDate"].max(), df["ChgOffDate"].max())
    acct["extract_date"] = str(extract.date())

    for c in ["DisbursementGross", "GrAppv", "SBA_Appv", "BalanceGross", "ChgOffPrinGr"]:
        df[c] = money(df[c])

    # ---- label -------------------------------------------------------------
    st = df["MIS_Status"].str.strip()
    acct["label_counts"] = {str(k): int(v) for k, v in st.value_counts(dropna=False).items()}
    keep = st.isin(["CHGOFF", "P I F"])
    acct["rows_missing_label"] = int((~keep).sum())
    df = df[keep].copy()
    df["y"] = (st[keep] == "CHGOFF").astype(np.int8).to_numpy()

    # ---- maturity filter --------------------------------------------------
    start = df["DisbursementDate"].fillna(df["ApprovalDate"])
    term = df["Term"].astype("float64")
    ok_dates = start.notna() & (term > 0)
    acct["rows_no_date_or_zero_term"] = int((~ok_dates).sum())
    df = df[ok_dates].copy()
    start = start[ok_dates]
    term = term[ok_dates]
    end = start + pd.to_timedelta(term * 30.4375, unit="D")
    matured = end <= extract
    acct["rows_after_label_and_date"] = int(len(df))
    acct["rows_matured"] = int(matured.sum())
    acct["rows_not_matured"] = int((~matured).sum())
    acct["censoring_rate_excluded_cohorts"] = float((~matured).mean())
    acct["chargeoff_rate_all"] = float(df["y"].mean())
    acct["chargeoff_rate_matured"] = float(df.loc[matured, "y"].mean())
    acct["chargeoff_rate_not_matured"] = float(df.loc[~matured, "y"].mean())
    df = df[matured].copy()
    df["end_date"] = end[matured]

    # ---- approval fiscal year (time axis) ----------------------------------
    fy = pd.to_numeric(df["ApprovalFY"].str.extract(r"(\d{4})")[0], errors="coerce")
    fy = fy.fillna(df["ApprovalDate"].dt.year)
    df["fy"] = fy.astype("int32")
    acct["fy_min"], acct["fy_max"] = int(df["fy"].min()), int(df["fy"].max())

    # ---- out-of-time split: last cohorts are the test window ---------------
    # target ~25% of matured rows in the test window, cut on a fiscal-year boundary
    cnt = df["fy"].value_counts().sort_index()
    cum = cnt[::-1].cumsum()[::-1] / len(df)
    cut = int(cum[cum <= 0.25].index.min()) if (cum <= 0.25).any() else int(df["fy"].max())
    df["test"] = (df["fy"] >= cut).astype(np.int8)
    acct["split_test_from_fy"] = cut
    acct["n_train"] = int((df["test"] == 0).sum())
    acct["n_test"] = int((df["test"] == 1).sum())
    acct["base_rate_train"] = float(df.loc[df.test == 0, "y"].mean())
    acct["base_rate_test"] = float(df.loc[df.test == 1, "y"].mean())

    tr = df["test"].to_numpy() == 0

    # ---- features by source ------------------------------------------------
    def ordcode(s: pd.Series) -> np.ndarray:
        """Ordinal code fixed on the TRAIN window; unseen test levels -> -1."""
        lv = pd.Index(sorted(s[tr].dropna().unique()))
        return s.map({v: i for i, v in enumerate(lv)}).fillna(-1).to_numpy(dtype="float32")

    def freq(s: pd.Series) -> np.ndarray:
        """Level frequency counted on the TRAIN window only."""
        f = s[tr].value_counts()
        return s.map(f).fillna(0).to_numpy(dtype="float32")

    naics2 = df["NAICS"].str.slice(0, 2)
    naics2 = naics2.where(df["NAICS"].str.len() == 6, other=pd.NA)
    franchise = (~df["FranchiseCode"].isin(["0", "1", "00000", "00001"])).astype("float32")
    urb = pd.to_numeric(df["UrbanRural"], errors="coerce").fillna(-1).astype("float32")
    newex = pd.to_numeric(df["NewExist"], errors="coerce")
    newex = newex.where(newex.isin([1.0, 2.0])).fillna(0).astype("float32")

    F = {
        "naics2": ordcode(naics2),
        "log_noemp": np.log1p(df["NoEmp"].clip(lower=0)).to_numpy("float32"),
        "new_exist": newex.to_numpy(),
        "create_job": df["CreateJob"].to_numpy("float32"),
        "retained_job": df["RetainedJob"].to_numpy("float32"),
        "log_create_job": np.log1p(df["CreateJob"].clip(lower=0)).to_numpy("float32"),
        "log_retained_job": np.log1p(df["RetainedJob"].clip(lower=0)).to_numpy("float32"),
        "franchise": franchise.to_numpy(),
        "urban_rural": urb.to_numpy(),
        "state": ordcode(df["State"]),
    }
    gr = df["GrAppv"].to_numpy("float64")
    sba = df["SBA_Appv"].to_numpy("float64")
    dis = df["DisbursementGross"].to_numpy("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        gshare = np.where(gr > 0, sba / gr, np.nan)
        dratio = np.where(gr > 0, dis / gr, np.nan)
    G = {
        "term": df["Term"].to_numpy("float32"),
        "log_grappv": np.log1p(np.nan_to_num(gr, nan=0.0)).astype("float32"),
        "log_sba_appv": np.log1p(np.nan_to_num(sba, nan=0.0)).astype("float32"),
        "log_disb": np.log1p(np.nan_to_num(dis, nan=0.0)).astype("float32"),
        "guarantee_share": np.nan_to_num(gshare, nan=-1.0).astype("float32"),
        "disb_to_appv": np.nan_to_num(dratio, nan=-1.0).astype("float32"),
        "revline": df["RevLineCr"].str.strip().map({"Y": 1.0, "N": 0.0}).fillna(-1).to_numpy("float32"),
        "lowdoc": df["LowDoc"].str.strip().map({"Y": 1.0, "N": 0.0}).fillna(-1).to_numpy("float32"),
    }
    B = {
        "bank": ordcode(df["Bank"]),
        "bank_freq": freq(df["Bank"]),
        "bank_state": ordcode(df["BankState"]),
        "same_state": (df["BankState"].fillna("") == df["State"].fillna("")).astype("float32").to_numpy(),
    }
    cols, src_of, blocks = [], [], {"firmographics": F, "guarantee": G, "bank": B}
    mats = []
    for s in SOURCES:
        for k, v in blocks[s].items():
            cols.append(k); src_of.append(s); mats.append(v)
    X = np.column_stack(mats).astype("float32")
    acct["n_features"] = X.shape[1]
    acct["features_by_source"] = {s: [c for c, u in zip(cols, src_of) if u == s] for s in SOURCES}

    # incumbent field set: F plus Term and log GrAppv
    inc_cols = ["naics2", "log_noemp", "new_exist", "create_job", "retained_job",
                "franchise", "urban_rural", "term", "log_grappv"]
    acct["incumbent_fields"] = inc_cols

    # fairness group columns: sector (2-digit NAICS), region (borrower state), size band
    noemp = df["NoEmp"].to_numpy()
    size_band = np.digitize(noemp, [1, 5, 20, 100])  # 0:0 emp,1:1-4,2:5-19,3:20-99,4:100+
    groups = {"sector": F["naics2"].astype("int32"),
              "region": F["state"].astype("int32"),
              "size_band": size_band.astype("int32")}
    acct["size_band_edges"] = "NoEmp in {0},{1-4},{5-19},{20-99},{100+}"

    np.savez_compressed(
        os.path.join(PROC, "sba.npz"),
        X=X, y=df["y"].to_numpy("int8"), test=df["test"].to_numpy("int8"),
        fy=df["fy"].to_numpy("int32"),
        cols=np.array(cols), src=np.array(src_of), inc_cols=np.array(inc_cols),
        g_sector=groups["sector"], g_region=groups["region"], g_size=groups["size_band"],
    )
    dump("SBA_cohort.json", acct)
    print(json.dumps({k: acct[k] for k in
                      ["rows_raw", "rows_matured", "censoring_rate_excluded_cohorts",
                       "chargeoff_rate_matured", "n_train", "n_test", "split_test_from_fy",
                       "n_features"]}, indent=1))


if __name__ == "__main__":
    main()
