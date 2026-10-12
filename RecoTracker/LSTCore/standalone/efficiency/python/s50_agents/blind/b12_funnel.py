#!/usr/bin/env python3
"""b12: mutually exclusive first-failure funnel for lost denominator sims (furthest stage reached), and for sims whose
T5s ALL died in AfterBuild: the AB survivors that replaced them (same module, >=7 shared hits with an own T5), with the
number of the sim's own hits they contain (own hits = hits of MDs matched to the sim with both hits; lower bound)."""
import numpy as np, uproot, pandas as pd, awkward as ak
from bcommon import *
sims = pd.read_pickle("b01_simtable.pkl"); low = pd.read_pickle("b05_lower.pkl"); p6 = pd.read_pickle("b06_pt5.pkl")
s8 = pd.read_pickle("b08_pls_state.pkl")
plsAlive = s8.groupby(["evt","isim"]).state.apply(lambda z: any(v.startswith("alive") or v == "pLS_TC" for v in z))
df = sims.merge(low, on=["evt","isim"]).merge(p6[["evt","isim","n_pT5_both","n_pT5_both_alive","n_pT5_ownT5"]], on=["evt","isim"])
df = df.join(plsAlive.rename("plsAlive"), on=["evt","isim"]); df["plsAlive"] = df.plsAlive.fillna(False)
t = uproot.open(F100)["tree"]
lab = {}; abrows = []
for ie, d in enumerate(t.iterate(BR, step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]; S = df[(df.evt == ie) & (df.tcIdx < 0)]
    if len(S) == 0: continue
    o = build(d); h = o["t5h"]; lm = o["t5_lowmod"]
    bits = np.asarray(d.t5_isDupBits); pof = np.asarray(d.t5_partOfPT5).astype(bool); tight = np.asarray(d.t5_tightCutFlag).astype(bool)
    eta = np.asarray(d.t5_eta); phi = np.asarray(d.t5_phi); fake = np.asarray(d.t5_isFake).astype(bool)
    tc_t5 = set(np.asarray(d.tc_t5Idx).tolist()); p5t5 = np.asarray(d.pT5_t5Idx); p5dup = np.asarray(d.pT5_isDupReco)
    tcp5 = set(np.asarray(d.tc_pt5Idx).tolist())
    t5_in_pt5tc = set(p5t5[[k for k in range(len(p5t5)) if k in tcp5]].tolist()) if len(p5t5) else set()
    for r in S.itertuples():
        i = int(r.isim)
        t5s = np.asarray(d.sim_t5IdxAll[i])[np.asarray(d.sim_t5IdxAllFrac[i]) > 0.75]
        if len(t5s) == 0:
            if r.nSimL < 5: L = "0a no T5: <5 OT layers w/ simhits" + (" (no pLS)" if r.nPLS == 0 else (" (quad pLS exists, killed)" if r.pls_anyQuad else " (only triplet pLS)"))
            elif r.nMDL < 5: L = "0b no T5: <5 OT layers with MD"
            elif r.nT3 == 0: L = "0c no T5: no T3"
            else: L = "0d no T5: T3s exist, T5 creation cuts"
        else:
            abs_ = t5s[(bits[t5s] & 1) == 0]
            if r.n_pT5_both > 0: L = "3 own pT5 built, removed by pT5 dedup" if r.n_pT5_both_alive == 0 else "3x own pT5 alive (other)"
            elif len(abs_) == 0: L = "1 all T5s removed in AfterBuild dedup"
            elif (~pof[abs_]).any():
                nonp = abs_[~pof[abs_]]
                if ((bits[nonp] & 2) > 0).all(): L = "4 T5 removed in BeforeTC dedup"
                elif ((bits[nonp] & 0x12) > 0).all(): L = "4b T5 removed BeforeTC/CrossCleanT5"
                elif (~tight[nonp[(bits[nonp] == 0)]]).all(): L = "5 T5 fails tightCutFlag"
                else: L = "6 T5 TC exists?"
            else:
                if r.nPLS == 0: L = "2a T5 absorbed in pT5 with other pLS; sim has no pLS"
                elif not r.plsAlive: L = "2b T5 absorbed in pT5 with other pLS; own pLS killed before pT5 building"
                else: L = "2c T5 absorbed in pT5 with other pLS; own pLS alive but pT5(own,own) not built"
            if L.startswith("1"):
                own_md = set(np.asarray(d.sim_mdIdxAll[i])[np.asarray(d.sim_mdIdxAllFrac[i]) > 0.75].tolist())
                own_h = set(o["mdh"][list(own_md)].ravel().tolist()) if own_md else set()
                surv = np.flatnonzero((bits & 1) == 0)
                best = None
                for v in t5s:
                    c = surv[lm[surv] == lm[v]]
                    if len(c) == 0: continue
                    nm = nshared(np.repeat(h[[v]], len(c), 0), h[c]); c = c[nm >= 7]
                    for j in c:
                        nown = len(set(h[j].tolist()) & own_h)
                        fate = ("pT5-TC" if j in t5_in_pt5tc else "pof,no TC") if pof[j] else ("T5-TC" if j in tc_t5 else "killed later")
                        abrows.append(dict(evt=ie, isim=i, dr=r.dr, surv=int(j), nown=nown, fake=bool(fake[j]), fate=fate))
        lab[(ie, i)] = L
df["label"] = [lab.get((e, i), "found") for e, i in zip(df.evt, df.isim)]
df.to_pickle("b12_funnel.pkl"); pd.DataFrame(abrows).to_pickle("b12_absurv.pkl")
out = []
for reg, sel in [("core dR<0.02", df.dr < 0.02), ("mid 0.02-0.1", (df.dr >= 0.02) & (df.dr < 0.1)), ("noncore dR>0.1", df.dr >= 0.1)]:
    x = df[sel]; out.append(f"== {reg}: denominator {len(x)}, found {int((x.tcIdx>=0).sum())}, eff {(x.tcIdx>=0).mean():.4f}")
    for k, v in x[x.tcIdx < 0].label.value_counts().sort_index().items(): out.append(f"   {v:5d}  ({v/len(x)*100:5.2f}% of den)  {k}")
A = pd.DataFrame(abrows)
if len(A):
    for reg, sel in [("core", A.dr < 0.02), ("noncore", A.dr >= 0.1)]:
        a = A[sel]; g = a.groupby(["evt","isim"])
        out.append(f"-- AB-only-lost sims ({reg}): {g.ngroups}; their AB survivors: {a.surv.nunique()} ; own-hit count of survivors: {a.nown.value_counts().sort_index().to_dict()}")
        out.append(f"   survivor fate: {a.drop_duplicates(['evt','surv']).fate.value_counts().to_dict()}  survivor fake(<=7/10): {int(a.drop_duplicates(['evt','surv']).fake.sum())}")
        best = g.nown.max(); out.append(f"   per-sim best survivor own hits: {best.value_counts().sort_index().to_dict()}")
txt = "\n".join(out); print(txt); open("b12_funnel.txt", "w").write(txt + "\n")
