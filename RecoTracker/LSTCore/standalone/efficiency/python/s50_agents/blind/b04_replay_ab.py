#!/usr/bin/env python3
"""b04: exact offline replay of RemoveDupQuintupletsAfterBuild (Kernels.h; LST_AB defaults dEta<=0.1,dPhi<=0.1,nMatched>=7,
same lower module, remove the T5 with larger-or-equal score_rphisum (ix on ties, ix<jx)). Validates vs t5_isDupBits&1.
For T5s matched (>0.75) to denominator sims, records who removed them. Output b04_ab_kills.pkl"""
import sys, numpy as np, uproot, pandas as pd
from bcommon import *
sims = pd.read_pickle("b01_simtable.pkl")
t = uproot.open(F100)["tree"]
rows = []; val = []
for ie, d in enumerate(t.iterate(BR, step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]; o = build(d)
    assert (o["t3md"][np.asarray(d.t5_t3Idx1)][:, 0] == o["t3md"][np.asarray(d.t5_t3Idx0)][:, 2]).all()
    eta = np.asarray(d.t5_eta); phi = np.asarray(d.t5_phi); sc = np.asarray(d.t5_score); h = o["t5h"]
    lm = o["t5_lowmod"]; n = len(eta)
    rm = np.zeros(n, bool)
    killers = {}  # victim -> list of killers
    S = sims[sims.evt == ie]
    interest = set()
    for i in S.isim.astype(int):
        interest.update(np.asarray(d.sim_t5IdxAll[i]).tolist())
    order = np.argsort(lm, kind="stable")
    bounds = np.flatnonzero(np.diff(lm[order])) + 1
    for grp in np.split(order, bounds):
        grp = np.sort(grp)  # kernel index order within module
        m = len(grp)
        if m < 2: continue
        ii, jj = np.triu_indices(m, 1)
        for s0 in range(0, len(ii), 2_000_000):
            a = grp[ii[s0:s0+2_000_000]]; b = grp[jj[s0:s0+2_000_000]]
            de = np.abs(eta[a] - eta[b]); dp = np.abs((phi[a] - phi[b] + np.pi) % (2*np.pi) - np.pi)
            k = (de <= 0.1) & (dp <= 0.1)
            a, b = a[k], b[k]
            if len(a) == 0: continue
            nm = nshared(h[a], h[b])
            k = nm >= 7; a, b = a[k], b[k]
            vic = np.where(sc[a] >= sc[b], a, b); kil = np.where(sc[a] >= sc[b], b, a)
            rm[vic] = True
            for v, kk in zip(vic, kil):
                if v in interest: killers.setdefault(int(v), []).append(int(kk))
    reco = (np.asarray(d.t5_isDupBits) & 1).astype(bool)
    val.append((ie, n, int((rm != reco).sum()), int(reco.sum()), int(rm.sum())))
    t5sim = np.asarray(d.t5_simIdx); t5fake = np.asarray(d.t5_isFake)
    for v, ks in killers.items():
        for kk in ks:
            rows.append(dict(evt=ie, victim=v, killer=kk, killer_sim=int(t5sim[kk]), killer_fake=int(t5fake[kk]),
                             killer_ABdup=bool(rm[kk]), nshared=int(nshared(h[[v]], h[[kk]])[0]),
                             dscore=float(sc[v]-sc[kk])))
    print(val[-1], flush=True)
pd.DataFrame(rows).to_pickle("b04_ab_kills.pkl")
V = np.array(val); print("TOTAL T5", V[:,1].sum(), "mismatch", V[:,2].sum(), "reco AB-dup", V[:,3].sum(), "replay", V[:,4].sum())
