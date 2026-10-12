#!/usr/bin/env python3
"""b10: who removes the T5s of lost sims in AfterBuild / BeforeTC dedup (from b04 replay)."""
import pandas as pd, numpy as np
k = pd.read_pickle("b04_kills.pkl"); f = pd.read_pickle("b02_t5fate.pkl")
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
for reg, sel in [("core", k.dr2 if False else None)]: pass
f["key"] = list(zip(f.evt, f.isim))
sims_found = f.groupby(["evt","isim"]).found.first()
k = k.join(sims_found, on=["evt","isim"])
dr = f.groupby(["evt","isim"]).dr.first(); k = k.join(dr, on=["evt","isim"])
def kk(r):
    if r.killer_sim == r.isim: return "own-sim T5"
    if r.killer_sim < 0: return "fake T5"
    return "other-sim T5"
k["kkind"] = k.apply(kk, axis=1)
for reg, sel in [("core dR<0.02", k.dr < 0.02), ("noncore dR>0.1", k.dr >= 0.1)]:
    for kind in ["AB", "BTC"]:
        x = k[sel & ~k.found & (k.kind == kind) & (k.v_pof == 0)]
        P(f"== {reg} {kind}: lost sims {x.groupby(['evt','isim']).ngroups}, victim T5s {x.victim.nunique()}, kill rows {len(x)}")
        P("   killer kind:", x.kkind.value_counts().to_dict())
        # killer final status
        st = np.where(x.killer_pof == 1, "killer partOfPT5", np.where(x.killer_bits & 1, "killer AB-dup",
             np.where(x.killer_bits & 2, "killer BTC-dup", np.where(x.killer_bits & 16, "killer xclean",
             np.where(x.killer_tight == 0, "killer !tight", "killer -> T5 TC")))))
        P("   killer final status:", pd.Series(st).value_counts().to_dict())
        if kind == "BTC":
            P("   BTC branch: dR2<1e-3:", int((x.dR2 < 1e-3).sum()), " nShared>=5:", int((x.nshared >= 5).sum()),
              " d2emb<0.1:", int((x.d2emb < 0.1).sum()), " killer pof (pT5-priority):", int((x.killer_pof == 1).sum()))
            P("   d2emb quantiles", np.quantile(x.d2emb, [.1, .5, .9]).round(3).tolist(), " nShared dist", x.nshared.value_counts().sort_index().to_dict())
            # sims where every BTC killer is other/fake (distinct track) 
            g = x.groupby(["evt","isim"]).kkind.apply(lambda z: "own" if (z == "own-sim T5").all() else ("mixed" if (z == "own-sim T5").any() else "only other/fake"))
            P("   per-sim BTC killer kinds:", g.value_counts().to_dict())
        else:
            g = x.groupby(["evt","isim"]).kkind.apply(lambda z: "own" if (z == "own-sim T5").all() else ("mixed" if (z == "own-sim T5").any() else "only other/fake"))
            P("   per-sim AB killer kinds:", g.value_counts().to_dict())
            P("   nShared dist", x.nshared.value_counts().sort_index().to_dict())
open("b10_t5killers.txt", "w").write("\n".join(out) + "\n")
