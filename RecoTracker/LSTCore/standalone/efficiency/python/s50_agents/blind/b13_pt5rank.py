#!/usr/bin/env python3
"""b13: counterfactual pT5 dedup (RemoveDupPixelQuintupletsFromMap) with alternative ranking keys, replayed exactly
offline (same pairing: |dEta|,|dPhi|<=0.2, >=7 shared hit slots). Surviving pT5s -> pT5 TCs. A denominator sim is
'found' if matched (>0.75) by a non-pT5 TC of the baseline ntuple or by a surviving pT5. pT5s feed nothing else that
changes (CrossCleanT5 / CrossCleanpT3 use ALL pT5s regardless of isDup); CrossCleanpLS/T4 changes are neglected.
Reports per key: found sims per region, delta vs baseline, number of pT5 TCs, fake pT5 TCs."""
import numpy as np, uproot, pandas as pd, awkward as ak
from bcommon import *
sims = pd.read_pickle("b01_simtable.pkl")
def dphi(a, b): return np.abs((a - b + np.pi) % (2*np.pi) - np.pi)
KEYS = ["baseline(pT5 rPhiChi2)", "T5 score_rphisum", "pT5chi2+T5score", "1-T5dnn", "pT5chi2/T5dnn",
        "|1/ptpLS-1/ptT5|", "pT5chi2*(1-T5dnn)", "ORACLE(matched first)"]
t = uproot.open(F100)["tree"]
tot = {k: dict(core=0, mid=0, non=0, nTC=0, nFake=0) for k in KEYS}
valid = [0, 0]
for ie, d in enumerate(t.iterate(BR, step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]; o = build(d)
    S = sims[sims.evt == ie]
    reg = {int(r.isim): ("core" if r.dr < 0.02 else ("non" if r.dr >= 0.1 else "mid")) for r in S.itertuples()}
    tct = np.asarray(d.tc_type); tcsim = np.asarray(d.tc_simIdx)
    other_found = set(tcsim[(tct != 7) & (tcsim >= 0)].tolist())
    p5t5 = np.asarray(d.pT5_t5Idx); p5pls = np.asarray(d.pT5_plsIdx); n = len(p5t5)
    p5sims = [set(np.asarray(x).tolist()) for x in d.pT5_simIdxAll]  # (>0.75 by construction)
    p5fake = np.array([len(x) == 0 for x in p5sims], bool)
    if n:
        H = np.concatenate([o["plsh"][p5pls], o["t5h"][p5t5]], 1)
        eta = np.asarray(d.t5_eta)[p5t5]; phi = np.asarray(d.t5_phi)[p5t5]
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); ii = ii.ravel(); jj = jj.ravel()
        k = (ii != jj) & (np.abs(eta[ii]-eta[jj]) <= 0.2) & (dphi(phi[ii], phi[jj]) <= 0.2); ii, jj = ii[k], jj[k]
        k = nshared(H[ii], H[jj]) >= 7; ii, jj = ii[k], jj[k]
        sc = np.asarray(d.pT5_score).astype(np.float64); t5sc = np.asarray(d.t5_score)[p5t5].astype(np.float64)
        dnn = np.asarray(d.t5_dnnScore)[p5t5].astype(np.float64)
        ptp = np.asarray(d.pLS_pt)[p5pls]; ptt = np.asarray(d.t5_pt)[p5t5]
        keyv = {"baseline(pT5 rPhiChi2)": sc, "T5 score_rphisum": t5sc, "pT5chi2+T5score": sc + t5sc,
                "1-T5dnn": 1 - dnn, "pT5chi2/T5dnn": sc / np.maximum(dnn, 1e-3),
                "|1/ptpLS-1/ptT5|": np.abs(1/ptp - 1/ptt), "pT5chi2*(1-T5dnn)": sc * (1 - dnn) + 1e-9 * sc,
                "ORACLE(matched first)": p5fake.astype(float) * 1e6 + sc}
    for K in KEYS:
        if n:
            s = keyv[K]; kill = (s[ii] > s[jj]) | ((s[ii] == s[jj]) & (ii > jj))
            rm = np.zeros(n, bool); rm[ii[kill]] = True
            if K.startswith("baseline"):
                valid[0] += (rm == np.asarray(d.pT5_isDupReco).astype(bool)).sum(); valid[1] += n
        else: rm = np.zeros(0, bool)
        found = set(other_found)
        for x in np.flatnonzero(~rm): found |= p5sims[x]
        T = tot[K]; T["nTC"] += int((~rm).sum()); T["nFake"] += int((p5fake & ~rm).sum())
        for i, r in reg.items():
            if i in found: T[r] += 1
out = [f"baseline pT5-dedup replay agrees for {valid[0]}/{valid[1]} pT5s",
       "found = matched by baseline non-pT5 TC or surviving pT5 (core den 1210, mid 1607, noncore 1633)"]
b = tot[KEYS[0]]
for K in KEYS:
    T = tot[K]
    out.append(f"{K:24s} core={T['core']:4d} ({T['core']-b['core']:+4d})  mid={T['mid']:4d} ({T['mid']-b['mid']:+4d})  non={T['non']:4d} ({T['non']-b['non']:+4d}) | pT5TCs={T['nTC']} ({T['nTC']-b['nTC']:+d}) fakepT5TCs={T['nFake']} ({T['nFake']-b['nFake']:+d})")
txt = "\n".join(out); print(txt); open("b13_pt5rank.txt", "w").write(txt + "\n")
