#!/usr/bin/env python3
"""b04: offline replay (per victim) of the two T5 dedup kernels for every T5 matched (>0.75) to a denominator sim.
 AfterBuild (Kernels.h RemoveDupQuintupletsAfterBuild, defaults LST_AB_*): same lower module, |dEta|<=0.1, |dPhi|<=0.1,
   nSharedHits>=7 ; the T5 with larger score_rphisum is removed (tie: lower index removed).
 BeforeTC (RemoveDupQuintupletsBeforeTC, defaults LST_BTC_*): any eligible module, both not AB-dup, not both partOfPT5,
   |dEta|<=0.1,|dPhi|<=0.1, and [((dR2<1e-3 or nShared>=5) and d2emb<1.0) or (dR2<0.02 and d2emb<0.1)];
   loser = non-pT5 one, else larger score (tie: lower index).
Validated against t5_isDupBits bit0 / bit1. Output: b04_kills.pkl (victim,killer rows) and b04_valid.txt"""
import numpy as np, uproot, pandas as pd, awkward as ak
from bcommon import *
sims = pd.read_pickle("b01_simtable.pkl")
t = uproot.open(F100)["tree"]
rows = []; vstat = np.zeros(8, int)
def dphi(a, b): return np.abs((a - b + np.pi) % (2*np.pi) - np.pi)
for ie, d in enumerate(t.iterate(BR, step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]; o = build(d)
    eta = np.asarray(d.t5_eta); phi = np.asarray(d.t5_phi); sc = np.asarray(d.t5_score); h = o["t5h"]; lm = o["t5_lowmod"]
    bits = np.asarray(d.t5_isDupBits); pof = np.asarray(d.t5_partOfPT5).astype(bool)
    emb = ak.to_numpy(d.t5_embed).astype(np.float32) if len(eta) else np.zeros((0, 6), np.float32)
    t5sim = np.asarray(d.t5_simIdx); t5fake = np.asarray(d.t5_isFake); tight = np.asarray(d.t5_tightCutFlag)
    ab_reco = (bits & 1).astype(bool)
    oe = np.argsort(eta, kind="stable"); es = eta[oe]
    S = sims[sims.evt == ie]
    vict = {}
    for i in S.isim.astype(int):
        for v, f in zip(np.asarray(d.sim_t5IdxAll[i]), np.asarray(d.sim_t5IdxAllFrac[i])):
            if f > 0.75: vict.setdefault(int(v), []).append(i)
    for v, simlist in vict.items():
        lo, hi = np.searchsorted(es, eta[v] - 0.1 - 1e-6), np.searchsorted(es, eta[v] + 0.1 + 1e-6, "right")
        c = oe[lo:hi]; c = c[c != v]
        c = c[(np.abs(eta[c] - eta[v]) <= 0.1) & (dphi(phi[c], phi[v]) <= 0.1)]
        nm = nshared(np.repeat(h[[v]], len(c), 0), h[c]) if len(c) else np.zeros(0, int)
        # ---- AfterBuild
        same = lm[c] == lm[v]
        abk = same & (nm >= 7) & ((sc[v] > sc[c]) | ((sc[v] == sc[c]) & (v < c)))
        ab_rep = abk.any()
        vstat[0] += 1; vstat[1] += ab_rep == ab_reco[v]
        # ---- BeforeTC (only if v not AB-dup in reco)
        btc_rep = False; btck = np.zeros(len(c), bool)
        if not ab_reco[v]:
            de = np.abs(eta[c] - eta[v]); dp = dphi(phi[c], phi[v]); dr2 = de*de + dp*dp
            d2 = ((emb[c] - emb[v])**2).sum(1)
            ok = (~ab_reco[c]) & ~(pof[c] & pof[v])
            cond = ok & ((((dr2 < 1e-3) | (nm >= 5)) & (d2 < 1.0)) | ((dr2 < 0.02) & (d2 < 0.1)))
            lose = np.where(pof[c] & ~pof[v], True, np.where(pof[v] & ~pof[c], False,
                     (sc[v] > sc[c]) | ((sc[v] == sc[c]) & (v < c))))
            btck = cond & lose
            btc_rep = btck.any()
            if not pof[v]:
                vstat[2] += 1; vstat[3] += btc_rep == bool(bits[v] & 2)
                vstat[4] += btc_rep and not (bits[v] & 2); vstat[5] += (not btc_rep) and bool(bits[v] & 2)
        for kind, km in (("AB", abk), ("BTC", btck)):
            for j in c[km]:
                k = int(np.flatnonzero(c == j)[0])
                de = abs(eta[j]-eta[v]); dp = float(dphi(phi[j], phi[v]))
                for i in simlist:
                    rows.append(dict(evt=ie, isim=i, victim=v, kind=kind, killer=int(j), killer_sim=int(t5sim[j]),
                                     killer_fake=int(t5fake[j]), killer_bits=int(bits[j]), killer_pof=int(pof[j]),
                                     killer_tight=int(tight[j]), nshared=int(nm[k]), dR2=de*de+dp*dp,
                                     d2emb=float(((emb[j]-emb[v])**2).sum()), sc_v=float(sc[v]), sc_k=float(sc[j]),
                                     v_bits=int(bits[v]), v_pof=int(pof[v]), v_tight=int(tight[v])))
    print(ie, len(vict), vstat, flush=True)
pd.DataFrame(rows).to_pickle("b04_kills.pkl")
msg = f"AfterBuild replay agrees with reco bit0 for {vstat[1]}/{vstat[0]} sim-matched T5s; BeforeTC replay agrees with bit1 for {vstat[3]}/{vstat[2]} (non-AB-dup, non-partOfPT5) sim-matched T5s (fp {vstat[4]}, fn {vstat[5]}). partOfPT5 T5s can carry bit1 from the asymmetric pT5 branch but that does not affect TCs."
print(msg); open("b04_valid.txt", "w").write(msg + "\n")
