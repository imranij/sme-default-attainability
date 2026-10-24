"""Home Credit: load, aggregate the child tables to SK_ID_CURR, source grouping,
application-order split.

The table-to-source grouping and the aggregation set defined below were both fixed before
this script was first run, as were the label and the split.

Sources: A application | U bureau (+bureau_balance) | I instalments | P POS-cash.
The three large child tables are reduced with an accumulating chunked group-by; they are
never merged raw (7 GB ceiling).

Outputs data/processed/hc.npz and results/HC_cohort.json.
"""
from __future__ import annotations

import gc
import os
import time

import numpy as np
import pandas as pd

from common import DATA, dump

HC = os.path.join(DATA, "home_credit")
from common import PROC  # the repository root is resolved once, in common.py
os.makedirs(PROC, exist_ok=True)
SOURCES = ["application", "bureau", "instalments", "pos_cash"]
SRC_LETTER = {"application": "A", "bureau": "U", "instalments": "I", "pos_cash": "P"}
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


class Acc:
    """Accumulate associative group-by reductions across chunks, keyed on one id column."""

    def __init__(self, key):
        self.key = key
        self.sum = None
        self.max = None
        self.min = None
        self.cnt = None

    def add(self, df, sum_cols=(), max_cols=(), min_cols=()):
        g = df.groupby(self.key, sort=False)
        if sum_cols:
            s = g[list(sum_cols)].sum()
            self.sum = s if self.sum is None else self.sum.add(s, fill_value=0.0)
        c = g.size().rename("_n")
        self.cnt = c if self.cnt is None else self.cnt.add(c, fill_value=0)
        if max_cols:
            m = g[list(max_cols)].max()
            self.max = m if self.max is None else self.max.combine(m, np.fmax)
        if min_cols:
            m = g[list(min_cols)].min()
            self.min = m if self.min is None else self.min.combine(m, np.fmin)


def agg_bureau():
    """Source U: bureau.csv aggregated to SK_ID_CURR, plus bureau_balance via SK_ID_BUREAU."""
    cols = ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_TYPE", "DAYS_CREDIT",
            "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_MAX_OVERDUE",
            "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT_UPDATE",
            "AMT_ANNUITY"]
    b = pd.read_csv(os.path.join(HC, "bureau.csv"), usecols=cols)
    log(f"bureau.csv {len(b):,} rows")
    b["_active"] = (b["CREDIT_ACTIVE"] == "Active").astype("float32")
    b["_closed"] = (b["CREDIT_ACTIVE"] == "Closed").astype("float32")
    sumc = ["_active", "_closed", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY", "DAYS_CREDIT",
            "DAYS_CREDIT_UPDATE", "DAYS_CREDIT_ENDDATE"]
    g = b.groupby("SK_ID_CURR", sort=True)
    out = g[sumc].sum()
    n = g.size().rename("n_credits")
    out = out.join(n)
    out["n_credit_types"] = g["CREDIT_TYPE"].nunique()
    out["max_days_overdue"] = g["CREDIT_DAY_OVERDUE"].max()
    out["max_amt_max_overdue"] = g["AMT_CREDIT_MAX_OVERDUE"].max()
    out["min_days_credit"] = g["DAYS_CREDIT"].min()
    out["max_days_credit"] = g["DAYS_CREDIT"].max()
    for c in ["DAYS_CREDIT", "DAYS_CREDIT_UPDATE", "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_SUM",
              "AMT_CREDIT_SUM_DEBT", "AMT_ANNUITY"]:
        out["mean_" + c] = out[c] / out["n_credits"]
    out = out.rename(columns={"_active": "n_active", "_closed": "n_closed"})
    b2c = pd.Series(b["SK_ID_CURR"].to_numpy(), index=b["SK_ID_BUREAU"].to_numpy())
    del b, g
    gc.collect()

    # --- bureau_balance in chunks, reduced on SK_ID_BUREAU then summed to SK_ID_CURR ---
    bb_sum = None
    bb_min = None
    dpd = {"1", "2", "3", "4", "5"}
    for ch in pd.read_csv(os.path.join(HC, "bureau_balance.csv"),
                          usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
                          dtype={"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16",
                                 "STATUS": "string"},
                          chunksize=6_000_000):
        st = ch["STATUS"]
        ch["_dpd"] = st.isin(dpd).astype("float32")
        ch["_c"] = (st == "C").astype("float32")
        ch["_x"] = (st == "X").astype("float32")
        ch["_m"] = 1.0
        g = ch.groupby("SK_ID_BUREAU", sort=False)
        s = g[["_dpd", "_c", "_x", "_m"]].sum()
        mn = g["MONTHS_BALANCE"].min()
        bb_sum = s if bb_sum is None else bb_sum.add(s, fill_value=0.0)
        bb_min = mn if bb_min is None else bb_min.combine(mn, np.fmin)
        log(f"  bureau_balance chunk -> {len(bb_sum):,} SK_ID_BUREAU")
        del ch, g, s, mn
        gc.collect()
    bb = bb_sum.join(bb_min.rename("earliest_month"))
    bb["curr"] = bb.index.map(b2c)
    bb = bb[bb["curr"].notna()]
    gb = bb.groupby("curr", sort=True)
    bbc = gb[["_dpd", "_c", "_x", "_m"]].sum()
    bbc["bb_earliest_month"] = gb["earliest_month"].min()
    bbc = bbc.rename(columns={"_dpd": "bb_n_dpd_months", "_c": "bb_n_closed_months",
                              "_x": "bb_n_unknown_months", "_m": "bb_n_months"})
    for c in ["bb_n_dpd_months", "bb_n_closed_months", "bb_n_unknown_months"]:
        bbc[c.replace("n_", "share_")] = bbc[c] / bbc["bb_n_months"].clip(lower=1)
    out = out.join(bbc, how="outer")
    del bb, bb_sum, bb_min, bbc, gb
    gc.collect()
    out.index.name = "SK_ID_CURR"
    log(f"source U: {out.shape[1]} features on {len(out):,} applicants")
    return out


def agg_instalments():
    """Source I: installments_payments.csv aggregated to SK_ID_CURR."""
    acc_sum, acc_max, acc_min, pairs = None, None, None, []
    for ch in pd.read_csv(os.path.join(HC, "installments_payments.csv"),
                          dtype={"SK_ID_PREV": "int32", "SK_ID_CURR": "int32"},
                          chunksize=4_000_000):
        late = (ch["DAYS_ENTRY_PAYMENT"] - ch["DAYS_INSTALMENT"]).astype("float32")
        ratio = (ch["AMT_PAYMENT"] / ch["AMT_INSTALMENT"].replace(0.0, np.nan)).astype("float32")
        ch["_late"] = late
        ch["_late_pos"] = (late > 0).astype("float32")
        ch["_ratio"] = ratio
        ch["_under"] = (ratio < 0.999).astype("float32")
        ch["_n"] = 1.0
        sc = ["_late", "_late_pos", "_ratio", "_under", "_n", "AMT_PAYMENT", "AMT_INSTALMENT",
              "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT"]
        g = ch.groupby("SK_ID_CURR", sort=False)
        s = g[sc].sum()
        mx = g[["_late", "NUM_INSTALMENT_NUMBER"]].max()
        mn = g[["_ratio"]].min()
        acc_sum = s if acc_sum is None else acc_sum.add(s, fill_value=0.0)
        acc_max = mx if acc_max is None else acc_max.combine(mx, np.fmax)
        acc_min = mn if acc_min is None else acc_min.combine(mn, np.fmin)
        pairs.append(ch[["SK_ID_CURR", "SK_ID_PREV"]].drop_duplicates())
        log(f"  installments chunk -> {len(acc_sum):,} applicants")
        del ch, g, s, mx, mn
        gc.collect()
    npv = pd.concat(pairs).drop_duplicates().groupby("SK_ID_CURR").size().rename("i_n_prev")
    o = pd.DataFrame(index=acc_sum.index)
    n = acc_sum["_n"]
    o["i_n_payments"] = n
    o["i_mean_late"] = acc_sum["_late"] / n
    o["i_max_late"] = acc_max["_late"]
    o["i_n_late"] = acc_sum["_late_pos"]
    o["i_share_late"] = acc_sum["_late_pos"] / n
    o["i_mean_ratio"] = acc_sum["_ratio"] / n
    o["i_min_ratio"] = acc_min["_ratio"]
    o["i_n_under"] = acc_sum["_under"]
    o["i_share_under"] = acc_sum["_under"] / n
    o["i_sum_payment"] = acc_sum["AMT_PAYMENT"]
    o["i_sum_instalment"] = acc_sum["AMT_INSTALMENT"]
    o["i_mean_inst_number"] = acc_sum["NUM_INSTALMENT_NUMBER"] / n
    o["i_max_inst_number"] = acc_max["NUM_INSTALMENT_NUMBER"]
    o["i_mean_days_instalment"] = acc_sum["DAYS_INSTALMENT"] / n
    o = o.join(npv)
    o.index.name = "SK_ID_CURR"
    log(f"source I: {o.shape[1]} features on {len(o):,} applicants")
    return o


def agg_pos():
    """Source P: POS_CASH_balance.csv aggregated to SK_ID_CURR."""
    acc_sum, acc_max, acc_min, pairs = None, None, None, []
    for ch in pd.read_csv(os.path.join(HC, "POS_CASH_balance.csv"),
                          dtype={"SK_ID_PREV": "int32", "SK_ID_CURR": "int32",
                                 "MONTHS_BALANCE": "int16", "SK_DPD": "int32",
                                 "SK_DPD_DEF": "int32"},
                          chunksize=4_000_000):
        ch["_n"] = 1.0
        ch["_dpd_pos"] = (ch["SK_DPD"] > 0).astype("float32")
        ch["_active"] = (ch["NAME_CONTRACT_STATUS"] == "Active").astype("float32")
        ch["_completed"] = (ch["NAME_CONTRACT_STATUS"] == "Completed").astype("float32")
        sc = ["_n", "_dpd_pos", "_active", "_completed", "SK_DPD", "SK_DPD_DEF",
              "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "MONTHS_BALANCE"]
        g = ch.groupby("SK_ID_CURR", sort=False)
        s = g[sc].sum()
        mx = g[["SK_DPD", "SK_DPD_DEF", "MONTHS_BALANCE"]].max()
        mn = g[["MONTHS_BALANCE"]].min()
        acc_sum = s if acc_sum is None else acc_sum.add(s, fill_value=0.0)
        acc_max = mx if acc_max is None else acc_max.combine(mx, np.fmax)
        acc_min = mn if acc_min is None else acc_min.combine(mn, np.fmin)
        pairs.append(ch[["SK_ID_CURR", "SK_ID_PREV"]].drop_duplicates())
        log(f"  POS_CASH chunk -> {len(acc_sum):,} applicants")
        del ch, g, s, mx, mn
        gc.collect()
    npv = pd.concat(pairs).drop_duplicates().groupby("SK_ID_CURR").size().rename("p_n_prev")
    o = pd.DataFrame(index=acc_sum.index)
    n = acc_sum["_n"]
    o["p_n_months"] = n
    o["p_mean_dpd"] = acc_sum["SK_DPD"] / n
    o["p_max_dpd"] = acc_max["SK_DPD"]
    o["p_mean_dpd_def"] = acc_sum["SK_DPD_DEF"] / n
    o["p_max_dpd_def"] = acc_max["SK_DPD_DEF"]
    o["p_n_dpd_months"] = acc_sum["_dpd_pos"]
    o["p_share_dpd_months"] = acc_sum["_dpd_pos"] / n
    o["p_n_active"] = acc_sum["_active"]
    o["p_n_completed"] = acc_sum["_completed"]
    o["p_mean_cnt_instalment"] = acc_sum["CNT_INSTALMENT"] / n
    o["p_mean_cnt_future"] = acc_sum["CNT_INSTALMENT_FUTURE"] / n
    o["p_mean_month"] = acc_sum["MONTHS_BALANCE"] / n
    o["p_min_month"] = acc_min["MONTHS_BALANCE"]
    o["p_max_month"] = acc_max["MONTHS_BALANCE"]
    o = o.join(npv)
    o.index.name = "SK_ID_CURR"
    log(f"source P: {o.shape[1]} features on {len(o):,} applicants")
    return o


def main():
    acct = {}
    U = agg_bureau()
    I = agg_instalments()
    P = agg_pos()

    app = pd.read_csv(os.path.join(HC, "application_train.csv"), low_memory=False)
    acct["rows_raw"] = int(len(app))
    acct["cols_raw"] = int(app.shape[1])
    acct["n_positive"] = int(app["TARGET"].sum())
    acct["base_rate"] = float(app["TARGET"].mean())
    log(f"application_train {len(app):,} x {app.shape[1]}  base={acct['base_rate']:.4f}")

    # ---- split: application order == ascending SK_ID_CURR, last 25% is the test window
    app = app.sort_values("SK_ID_CURR", kind="mergesort").reset_index(drop=True)
    cut_ix = int(round(0.75 * len(app)))
    test = np.zeros(len(app), dtype=np.int8)
    test[cut_ix:] = 1
    acct["split_rule"] = "ascending SK_ID_CURR; last 25% of rows form the test window"
    acct["split_test_from_SK_ID_CURR"] = int(app["SK_ID_CURR"].iloc[cut_ix])
    acct["n_train"] = int((test == 0).sum())
    acct["n_test"] = int((test == 1).sum())
    y = app["TARGET"].to_numpy("int8")
    acct["base_rate_train"] = float(y[test == 0].mean())
    acct["base_rate_test"] = float(y[test == 1].mean())
    acct["base_rate_shift_pp"] = 100.0 * (acct["base_rate_test"] - acct["base_rate_train"])
    acct["base_rate_shift_note"] = (
        "SBA's out-of-time split carried a 30.0 pp train/test base-rate shift induced by the "
        "maturity filter. The Home Credit analogue is "
        f"{acct['base_rate_shift_pp']:+.2f} pp; no maturity filter applies.")
    log(f"split: train {acct['n_train']:,} @ {acct['base_rate_train']:.4f} | "
        f"test {acct['n_test']:,} @ {acct['base_rate_test']:.4f} "
        f"(shift {acct['base_rate_shift_pp']:+.2f} pp)")
    tr = test == 0

    # ---- source A: the application table -----------------------------------
    ids = app["SK_ID_CURR"].to_numpy()
    feat = app.drop(columns=["SK_ID_CURR", "TARGET"])
    obj = [c for c in feat.columns
           if not pd.api.types.is_numeric_dtype(feat[c]) or pd.api.types.is_bool_dtype(feat[c])]
    acct["application_categorical_columns"] = obj
    for c in obj:
        lv = pd.Index(sorted(feat.loc[tr, c].dropna().unique()))
        feat[c] = feat[c].map({v: i for i, v in enumerate(lv)}).fillna(-1).astype("float32")
    A = feat.astype("float32")
    acct["n_features_application"] = int(A.shape[1])
    del app, feat
    gc.collect()

    # ---- align the three child sources on the application order ------------
    blocks = {"application": A}
    for nm, tbl, fill0 in [("bureau", U, ("n_credits", "n_active", "n_closed", "n_credit_types",
                                          "bb_n_months", "bb_n_dpd_months", "bb_n_closed_months",
                                          "bb_n_unknown_months")),
                           ("instalments", I, ("i_n_payments", "i_n_late", "i_n_under",
                                               "i_n_prev")),
                           ("pos_cash", P, ("p_n_months", "p_n_dpd_months", "p_n_active",
                                            "p_n_completed", "p_n_prev"))]:
        t = tbl.reindex(ids)
        for c in t.columns:
            t[c] = t[c].fillna(0.0 if c in fill0 else -1.0)
        t = t.astype("float32").reset_index(drop=True)
        blocks[nm] = t
        acct[f"n_features_{nm}"] = int(t.shape[1])
        acct[f"coverage_{nm}"] = float(tbl.reindex(ids).iloc[:, 0].notna().mean())
        log(f"aligned {nm}: {t.shape[1]} features, coverage {acct[f'coverage_{nm}']:.3f}")
    del U, I, P
    gc.collect()

    cols, src, mats = [], [], []
    for s in SOURCES:
        b = blocks[s]
        for c in b.columns:
            cols.append(f"{SRC_LETTER[s]}:{c}")
            src.append(s)
        mats.append(b.to_numpy("float32"))
    X = np.concatenate(mats, axis=1)
    del mats, blocks
    gc.collect()
    acct["n_features"] = int(X.shape[1])
    acct["features_by_source"] = {s: [c for c, u in zip(cols, src) if u == s] for s in SOURCES}
    acct["sources"] = SOURCES
    acct["grouping"] = "fixed in this script before any fit"
    acct["aggregation"] = "fixed in this script before any fit"

    # incumbent field set: source A exactly
    inc_cols = [c for c, u in zip(cols, src) if u == "application"]
    acct["incumbent_fields"] = "all of source A (the application table)"
    acct["n_incumbent_fields"] = len(inc_cols)

    np.savez_compressed(os.path.join(PROC, "hc.npz"),
                        X=X, y=y, test=test, sk_id=ids.astype("int32"),
                        cols=np.array(cols), src=np.array(src),
                        inc_cols=np.array(inc_cols))
    dump("HC_cohort.json", acct)
    log("done", {k: acct[k] for k in ["rows_raw", "base_rate", "n_train", "n_test",
                                      "base_rate_train", "base_rate_test",
                                      "base_rate_shift_pp", "n_features"]})


if __name__ == "__main__":
    main()
