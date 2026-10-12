#!/usr/bin/env python3
"""b05: lower-stage funnel. For each denominator sim: OT layers with sim hits, OT layers with a sim-matched MD (both hits),
LS/T3 matched, T5 matched, pLS matched (+ pLS isQuad / isDup bits), plus pT3 / pLS-TC info. Output b05_lower.pkl"""
import numpy as np, uproot, pandas as pd, awkward as ak
from bcommon import F100
sims = pd.read_pickle("b01_simtable.pkl")
t = uproot.open(F100)["tree"]
br = ["sim_simHitLayer","sim_mdIdxAll","sim_mdIdxAllFrac","sim_lsIdxAll","sim_lsIdxAllFrac","sim_t3IdxAll","sim_t3IdxAllFrac",
      "sim_plsIdxAll","sim_plsIdxAllFrac","md_layer","ls_mdIdx0","ls_mdIdx1","t3_lsIdx0","t3_lsIdx1",
      "pLS_isQuad","pLS_isDup","pLS_pt","pLS_eta","sim_recoHitDetId","sim_simHitDetId"]
rows = []
for ie, d in enumerate(t.iterate(br, step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]; S = sims[sims.evt == ie]
    mdl = np.asarray(d.md_layer); l0 = np.asarray(d.ls_mdIdx0); l1 = np.asarray(d.ls_mdIdx1)
    t30 = np.asarray(d.t3_lsIdx0); t31 = np.asarray(d.t3_lsIdx1)
    for i in S.isim.astype(int):
        def m(k):
            a = np.asarray(d[f"sim_{k}IdxAll"][i]); f = np.asarray(d[f"sim_{k}IdxAllFrac"][i]); return a[f > 0.75]
        mds = m("md"); lss = m("ls"); t3s = m("t3"); pls = m("pls")
        simL = set(np.asarray(d.sim_simHitLayer[i]).tolist())
        recoDet = set(np.asarray(d.sim_recoHitDetId[i]).tolist())
        mdL = set(mdl[mds].tolist())
        lsL = set(mdl[l0[lss]].tolist())
        t3L = set(mdl[l0[t30[t3s]]].tolist()) if len(t3s) else set()
        q = np.asarray(d.pLS_isQuad)[pls] if len(pls) else np.zeros(0, bool)
        du = np.asarray(d.pLS_isDup)[pls] if len(pls) else np.zeros(0, int)
        rows.append(dict(evt=ie, isim=i, nSimL=len(simL), simL=sorted(simL), nRecoDet=len(recoDet), nMDL=len(mdL), mdL=sorted(mdL),
                         nLS=len(lss), lsL=sorted(lsL), nT3=len(t3s), t3L=sorted(t3L), nPLS=len(pls),
                         pls_anyQuad=bool(q.any()), pls_anyAlive=bool((du == 0).any()), pls_quadAlive=bool((q & (du == 0)).any()),
                         pls_bits=sorted(set(du.tolist()))))
pd.DataFrame(rows).to_pickle("b05_lower.pkl"); print(len(rows))
