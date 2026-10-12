#!/usr/bin/env python3
"""b09: exact offline replay of RemoveDupPixelQuintupletsFromMap (defaults LST_PT5_*: |dEta|<=0.2,|dPhi|<=0.2 on the pT5's
(=T5's) half-precision eta/phi, >=7 of 14 hit slots shared (4 pLS slots + 10 T5 hits); ix removed if score_ix>score_jx or
tie and ix>jx; score = pT5 rPhiChiSquared w.r.t. the pixel-seed circle). Validated vs pT5_isDupReco.
For every pT5 whose pLS AND T5 are both matched to the same denominator sim ('own' pT5) that was removed, records winners."""
import numpy as np, uproot, pandas as pd, awkward as ak
from bcommon import *
sims = pd.read_pickle("b01_simtable.pkl")
t = uproot.open(F100)["tree"]
rows = []; agree = tot = 0
def dphi(a, b): return np.abs((a - b + np.pi) % (2*np.pi) - np.pi)
for ie, d in enumerate(t.iterate(BR, step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]; o = build(d)
    p5t5 = np.asarray(d.pT5_t5Idx); p5pls = np.asarray(d.pT5_plsIdx); n = len(p5t5)
    if n == 0: continue
    H = np.concatenate([o["plsh"][p5pls], o["t5h"][p5t5]], 1)
    eta = np.asarray(d.t5_eta)[p5t5]; phi = np.asarray(d.t5_phi)[p5t5]; sc = np.asarray(d.pT5_score)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); ii = ii.ravel(); jj = jj.ravel()
    k = (ii != jj) & (np.abs(eta[ii]-eta[jj]) <= 0.2) & (dphi(phi[ii], phi[jj]) <= 0.2); ii, jj = ii[k], jj[k]
    nm = nshared(H[ii], H[jj]); k = nm >= 7; ii, jj, nm = ii[k], jj[k], nm[k]
    kill = (sc[ii] > sc[jj]) | ((sc[ii] == sc[jj]) & (ii > jj))
    rm = np.zeros(n, bool); rm[ii[kill]] = True
    reco = np.asarray(d.pT5_isDupReco).astype(bool); agree += (rm == reco).sum(); tot += n
    t5sim = np.asarray(d.t5_simIdx); plssim = np.asarray(d.pLS_simIdx); p5sim = np.asarray(d.pT5_simIdx)
    tcp5 = set(np.asarray(d.tc_pt5Idx).tolist())
    S = sims[sims.evt == ie]
    for _, s in S.iterrows():
        i = int(s.isim)
        t5s = np.asarray(d.sim_t5IdxAll[i])[np.asarray(d.sim_t5IdxAllFrac[i]) > 0.75]
        pls = np.asarray(d.sim_plsIdxAll[i])[np.asarray(d.sim_plsIdxAllFrac[i]) > 0.75]
        own = np.flatnonzero(np.isin(p5t5, t5s) & np.isin(p5pls, pls))
        for v in own:
            if not rm[v]: continue
            w = jj[(ii == v) & kill]
            for j in w:
                rows.append(dict(evt=ie, isim=i, dr=s.dr, pt=s.pt, found=s.tcIdx >= 0, victim=int(v), winner=int(j),
                                 w_sim=int(p5sim[j]), w_t5sim=int(t5sim[p5t5[j]]), w_plssim=int(plssim[p5pls[j]]),
                                 same_pls=bool(p5pls[j] == p5pls[v]), same_t5=bool(p5t5[j] == p5t5[v]),
                                 w_inTC=int(j) in tcp5, w_removed=bool(rm[j]), sc_v=float(sc[v]), sc_w=float(sc[j]),
                                 nshared=int(nm[(ii == v) & (jj == j)][0]), v_pt=float(d.pLS_pt[p5pls[v]])))
pd.DataFrame(rows).to_pickle("b09_pt5kills.pkl")
msg = f"pT5 dedup replay agrees with pT5_isDupReco for {agree}/{tot} pT5s"
print(msg); open("b09_valid.txt", "w").write(msg + "\n")
