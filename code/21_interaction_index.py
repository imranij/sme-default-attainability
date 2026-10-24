"""The pairwise interaction index the paper names.

The paper's sentence reads "the largest pairwise Shapley--Taylor interaction is ... at
...". `12_sba_q3.py` and `18_hc_q3.py` compute a pairwise index with the weight
|T|!(K-|T|-2)!/(K-1)!, which is the Grabisch--Roubens *Shapley interaction index*, not the
Shapley--Taylor index of Sundararajan, Dhamdhere & Agarwal (2020). At order k = 2 the
Shapley--Taylor index of a top-order pair S is

    I^ST(S) = (k / n) * sum_{T subset N\\S}  delta_S v(T) / C(n-1, |T|),
    delta_S v(T) = v(T + i + j) - v(T + i) - v(T + j) + v(T).

The two weightings differ (at K = 3, 2/3 and 1/3 against 1/2 and 1/2), so the number the
slot was carrying was not the quantity the sentence names. This script recomputes both
indices from the coalition values already stored by 12_sba_q3.py and 18_hc_q3.py -- no refitting -- and
writes them side by side, so the slot can be filled with the named quantity and the
difference from the previously reported figure is on the record.

Writes results/interaction_indices.json.
"""
from __future__ import annotations

import itertools
import json
import math
import os

from common import RESULTS, dump

SPEC = {
    "SBA": dict(file="SBA_Q3_shapley.json", srcs=["F", "G", "B"],
                name={"F": "firmographics", "G": "guarantee", "B": "bank"}),
    "HomeCredit": dict(file="HC_Q3_shapley.json", srcs=["A", "U", "I", "P"],
                       name={"A": "application", "U": "bureau", "I": "instalments",
                             "P": "POS-cash"}),
}


def parse_values(d, srcs):
    """{frozenset: value} from the 'v_attainable'/'v_achieved' tag dicts."""
    out = {}
    for tag, val in d.items():
        C = frozenset() if tag == "EMPTY" else frozenset(tag.split("+"))
        assert C <= set(srcs), tag
        out[C] = val
    return out


def grabisch_roubens(v, srcs):
    K = len(srcs)
    out = {}
    for i, j in itertools.combinations(srcs, 2):
        rest = [s for s in srcs if s not in (i, j)]
        tot = 0.0
        for r in range(len(rest) + 1):
            for C in itertools.combinations(rest, r):
                Cs = frozenset(C)
                delta = v[Cs | {i, j}] - v[Cs | {i}] - v[Cs | {j}] + v[Cs]
                tot += (math.factorial(len(C)) * math.factorial(K - 2 - len(C))
                        / math.factorial(K - 1)) * delta
        out[f"{i}x{j}"] = tot
    return out


def shapley_taylor(v, srcs, k=2):
    """Order-2 Shapley-Taylor index for the top-order pairs."""
    K = len(srcs)
    out = {}
    for i, j in itertools.combinations(srcs, 2):
        rest = [s for s in srcs if s not in (i, j)]
        tot = 0.0
        for r in range(len(rest) + 1):
            for C in itertools.combinations(rest, r):
                Cs = frozenset(C)
                delta = v[Cs | {i, j}] - v[Cs | {i}] - v[Cs | {j}] + v[Cs]
                tot += delta / math.comb(K - 1, len(C))
        out[f"{i}x{j}"] = (k / K) * tot
    return out


def main():
    out = {"definition_shapley_taylor":
           "Sundararajan, Dhamdhere & Agarwal (2020), order k = 2, top-order pairs: "
           "I(S) = (k/n) sum_{T subset N\\S} delta_S v(T) / C(n-1,|T|)",
           "definition_grabisch_roubens":
           "Grabisch & Roubens (1999) Shapley interaction index: weight |T|!(K-|T|-2)!/(K-1)!",
           "note": "recomputed from the coalition values stored by 12_sba_q3.py and 18_hc_q3.py; no refit"}
    for ds, spec in SPEC.items():
        p = os.path.join(RESULTS, spec["file"])
        if not os.path.exists(p):
            print("skip", ds, "(no", spec["file"] + ")")
            continue
        d = json.load(open(p))
        nm = spec["name"]
        ren = lambda k: " and ".join(nm[c] for c in k.split("x"))
        rec = {}
        for fn, key in (("attainable", "v_attainable"), ("achieved", "v_achieved")):
            v = parse_values(d[key], spec["srcs"])
            st = shapley_taylor(v, spec["srcs"])
            gr = grabisch_roubens(v, spec["srcs"])
            top_st = max(st.items(), key=lambda kv: abs(kv[1]))
            top_gr = max(gr.items(), key=lambda kv: abs(kv[1]))
            rec[fn] = dict(
                shapley_taylor={ren(k): val for k, val in st.items()},
                grabisch_roubens={ren(k): val for k, val in gr.items()},
                largest_pair_shapley_taylor=ren(top_st[0]),
                largest_shapley_taylor=top_st[1],
                largest_pair_grabisch_roubens=ren(top_gr[0]),
                largest_grabisch_roubens=top_gr[1],
                same_pair_under_both=bool(top_st[0] == top_gr[0]))
        rec["n_sources"] = len(spec["srcs"])
        out[ds] = rec
        print(ds, json.dumps({k: rec["attainable"][k] for k in
                              ["largest_pair_shapley_taylor", "largest_shapley_taylor",
                               "largest_pair_grabisch_roubens", "largest_grabisch_roubens",
                               "same_pair_under_both"]}, indent=1), flush=True)
    dump("interaction_indices.json", out)


if __name__ == "__main__":
    main()
