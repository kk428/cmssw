#!/usr/bin/env python3
"""b03: per-T5 failure reasons -> per-sim recoverability (upper bounds) for lost sims with a matched T5."""
import pandas as pd, numpy as np, itertools
df = pd.read_pickle("b02_t5fate.pkl")
def reasons(r):
    R = set()
    if r.partOfPT5:
        if r.n_pt5_rightpLS == 0: R.add("pT5_wrongpLS")       # T5 absorbed by pT5(s) built only with other sims' pLS
        elif r.n_pt5_rightpLS_alive == 0: R.add("pT5_dedup")  # right pT5 built but killed by RemoveDupPixelQuintupletsFromMap
        elif r.n_pt5_inTC == 0: R.add("pT5_notTC")
        else: R.add("pT5_inTC_butnotmatched")
    else:
        if r.bits & 1: R.add("T5dup_AfterBuild")
        if r.bits & 2: R.add("T5dup_BeforeTC")
        if r.bits & 16: R.add("CrossCleanT5")
        if not r.tight: R.add("tightCutFlag")
        if not R: R.add("T5_should_be_TC")
    return frozenset(R)
df["R"] = df.apply(reasons, axis=1)
out = []
for reg, sel in [("core dR<0.02", df.dr < 0.02), ("noncore dR>0.1", df.dr >= 0.1)]:
    L = df[sel & ~df.found]
    nsim = L.groupby(["evt","isim"]).ngroups
    out.append(f"== {reg}: lost sims with >=1 matched T5: {nsim}  (matched T5 rows {len(L)})")
    # per-T5 reason combos
    out.append("  per-T5 reason combination counts:")
    for k, v in L.R.value_counts().items(): out.append(f"    {v:6d}  {sorted(k)}")
    # per sim: minimal reason sets
    g = L.groupby(["evt","isim"]).R.apply(list)
    allR = sorted(set().union(*[set(x) for x in L.R]))
    out.append("  sims recoverable if ONLY reason X were removed (some T5 fails solely by X):")
    for x in allR:
        n = sum(1 for Rs in g if any(r == frozenset([x]) for r in Rs))
        out.append(f"    {x:25s} {n}")
    # sim-level 'primary' attribution: pick the T5 with fewest reasons; tie-break by order
    order = ["pT5_wrongpLS","pT5_dedup","pT5_notTC","pT5_inTC_butnotmatched","T5dup_BeforeTC","CrossCleanT5","tightCutFlag","T5dup_AfterBuild","T5_should_be_TC"]
    prim = {}
    for key, Rs in g.items():
        best = min(Rs, key=lambda r: (len(r), min(order.index(x) for x in r)))
        lab = "+".join(sorted(best, key=order.index))
        prim[lab] = prim.get(lab, 0) + 1
    out.append("  sim-level primary label (T5 with fewest failure reasons):")
    for k, v in sorted(prim.items(), key=lambda kv: -kv[1]): out.append(f"    {v:5d}  {k}")
txt = "\n".join(out); print(txt); open("b03_t5reasons.txt","w").write(txt+"\n")
