#!/usr/bin/env python3
"""b07: pLS-level diagnosis. For every pLS matched (>0.75) to a denominator sim: isQuad, isDup bits, and the candidate
CheckHitspLS pass-1 killers (other pLS with |dEta|<=0.1 sharing >=3 of the 4 hit slots, kernel slot-counting incl. the
repeated hit of triplet seeds), with the killer's sim, quad flag and how many DISTINCT hits are shared.
Pass-1 loser rule: quad beats triplet; else higher pLS score loses (score not in ntuple -> only candidate list)."""
import numpy as np, uproot, pandas as pd, awkward as ak
from bcommon import *
sims = pd.read_pickle("b01_simtable.pkl")
t = uproot.open(F100)["tree"]
rows = []
for ie, d in enumerate(t.iterate(BR, step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]; o = build(d); ph = o["plsh"]
    eta = np.asarray(d.pLS_eta); q = np.asarray(d.pLS_isQuad).astype(bool); bits = np.asarray(d.pLS_isDup)
    psim = np.asarray(d.pLS_simIdx)
    S = sims[sims.evt == ie]
    for _, s in S.iterrows():
        i = int(s.isim)
        for p, f in zip(np.asarray(d.sim_plsIdxAll[i]), np.asarray(d.sim_plsIdxAllFrac[i])):
            if f <= 0.75: continue
            c = np.flatnonzero(np.abs(eta - eta[p]) <= 0.1); c = c[c != p]
            slots = (ph[p][:, None] == ph[c][:, None, :]).any(2).sum(1) if len(c) else np.zeros(0, int)
            k = slots >= 3
            kc = c[k]
            dist = [len(set(ph[p].tolist()) & set(ph[j].tolist())) for j in kc]
            any1 = c[(ph[p][:, None] == ph[c][:, None, :]).any(2).any(1)] if len(c) else c
            rows.append(dict(evt=ie, isim=i, dr=s.dr, found=s.tcIdx >= 0, pls=int(p), quad=bool(q[p]), bits=int(bits[p]),
                             nDistinct=len(set(ph[p].tolist())),
                             k3_n=len(kc), k3_sims=[int(psim[j]) for j in kc], k3_quad=[bool(q[j]) for j in kc],
                             k3_distinct=dist, k3_bits=[int(bits[j]) for j in kc],
                             share1_n=len(any1), share1_sims=[int(psim[j]) for j in any1]))
pd.DataFrame(rows).to_pickle("b07_pls.pkl"); print(len(rows))
