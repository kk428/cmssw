#!/usr/bin/env python3
"""b11: full offline replay of the T5 -> T5-TC selection after pT5 building:
   RemoveDupQuintupletsBeforeTC (on non-AfterBuild-dup T5s) -> CrossCleanT5 (vs ALL pT5s using the pT5's T5 embedding, and
   vs ALL pT3s with dR2<1e-3) -> AddT5asTrackCandidate (!isDup && !partOfPT5 && tightCutFlag).
 Baseline validated against the T5 TCs in the ntuple (tc_t5Idx). Then counterfactual variants that add a minimum
 number of shared hits to the angular/embedding dedup conditions, or change embedding thresholds.
 Per variant: lost denominator sims recovered (core / noncore), new T5 TCs, of which fake (t5_isFake) and duplicate
 (matched sim already has a TC or gets >1 new). Caveat: CrossCleanpLS/T4 interplay of new T5 TCs is not replayed."""
import numpy as np, uproot, pandas as pd, awkward as ak, sys
from bcommon import *
sims = pd.read_pickle("b01_simtable.pkl")
VARIANTS = {
  "baseline": dict(nmin=0, d2a=1.0, d2b=0.1),
  "nShared>=1": dict(nmin=1, d2a=1.0, d2b=0.1),
  "nShared>=2": dict(nmin=2, d2a=1.0, d2b=0.1),
  "nShared>=3": dict(nmin=3, d2a=1.0, d2b=0.1),
  "nShared>=5": dict(nmin=5, d2a=1.0, d2b=0.1),
  "emb d2<0.3/0.05": dict(nmin=0, d2a=0.3, d2b=0.05),
  "emb d2<0.1/0.03": dict(nmin=0, d2a=0.1, d2b=0.03),
  "nShared>=1 OR d2<0.05": dict(nmin=1, d2a=1.0, d2b=0.1, orclose=0.05),
  "no BTC/xclean at all": dict(nmin=99, d2a=1.0, d2b=0.1),
  "tight for pT>100 (fix flag)": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True),
  "tight pT>100 + nShared>=1": dict(nmin=1, d2a=1.0, d2b=0.1, tight100=True),
  "tight pT>100 + no BTC/xclean": dict(nmin=99, d2a=1.0, d2b=0.1, tight100=True, dump=True),
  "all T5 tight (no tight gate)": dict(nmin=0, d2a=1.0, d2b=0.1, tightall=True),
  "tight100 + protect dnn>0.9": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, protect=0.9),
  "tight100 + protect dnn>0.95": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, protect=0.95),
  "tight100 + protect dnn>0.98": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, protect=0.98),
  "tight100 + protect dnn>0.99": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, protect=0.99),
  "tight100 + protect dnn>0.995": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, protect=0.995),
  "tight100 + protect chi2<1": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, pchi=1.0),
  "tight100 + protect chi2<3": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, pchi=3.0),
  "tight100 + protect chi2<10": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, pchi=10.0),
  "tight100 + protect chi2<3 & pt>20": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, pchi=3.0, ppt=20.0),
  "tight100 + protect chi2<10 & pt>50": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, pchi=10.0, ppt=50.0),
  "tight100 + protect chi2<1 & dnn>0.9": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, pchi=1.0, pdnn=0.9),
  "protect chi2<3 & pt>20 (no tight fix)": dict(nmin=0, d2a=1.0, d2b=0.1, pchi=3.0, ppt=20.0),
  "tight100 + protect chi2<3 & pt>50": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, pchi=3.0, ppt=50.0),
  "tight100 + protect chi2<1 & pt>20": dict(nmin=0, d2a=1.0, d2b=0.1, tight100=True, pchi=1.0, ppt=20.0),
}
import sys as _s
if len(_s.argv) > 1: VARIANTS = {k: v for k, v in VARIANTS.items() if k == "baseline" or k in _s.argv[1:]}
def protmask(V, idx, dnn, sc, pt):
    m = np.ones(len(idx), bool); used = False
    if "protect" in V: m &= dnn[idx] > V["protect"]; used = True
    if "pchi" in V: m &= sc[idx] < V["pchi"]; used = True
    if "ppt" in V: m &= pt[idx] > V["ppt"]
    if "pdnn" in V: m &= dnn[idx] > V["pdnn"]
    return m if used else np.zeros(len(idx), bool)
def dphi(a, b): return np.abs((a - b + np.pi) % (2*np.pi) - np.pi)
def pairs_window(eta, idx, w):
    """all unordered pairs (a,b) of idx with |eta_a-eta_b|<=w"""
    o = idx[np.argsort(eta[idx], kind="stable")]; e = eta[o]
    A = []; B = []
    for k in range(1, len(o)):
        m = (e[k:] - e[:-k]) <= w + 1e-7
        if not m.any(): break
        A.append(o[:-k][m]); B.append(o[k:][m])
    if not A: return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(A), np.concatenate(B)
t = uproot.open(F100)["tree"]
recov = {v: [] for v in VARIANTS}; dumprows = []
res = {v: dict(core_rec=0, non_rec=0, mid_rec=0, newTC=0, newFake=0, newDup=0, lostTC=0, lostTC_true=0) for v in VARIANTS}
val = [0, 0, 0]
for ie, d in enumerate(t.iterate(BR + ["pT3_t3_eta","pT3_t3_phi"], step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]; o = build(d); h = o["t5h"]
    eta = np.asarray(d.t5_eta); phi = np.asarray(d.t5_phi); sc = np.asarray(d.t5_score)
    bits = np.asarray(d.t5_isDupBits); pof = np.asarray(d.t5_partOfPT5).astype(bool); tight = np.asarray(d.t5_tightCutFlag).astype(bool)
    emb = ak.to_numpy(d.t5_embed).astype(np.float32)
    fake = np.asarray(d.t5_isFake).astype(bool); dnn = np.asarray(d.t5_dnnScore); t5pt = np.asarray(d.t5_pt)
    alive = np.flatnonzero((bits & 1) == 0)
    a, b = pairs_window(eta, alive, 0.1)
    k = dphi(phi[a], phi[b]) <= 0.1; a, b = a[k], b[k]
    k = ~(pof[a] & pof[b]); a, b = a[k], b[k]
    de = eta[a]-eta[b]; dp = dphi(phi[a], phi[b]); dr2 = de*de+dp*dp
    d2 = ((emb[a]-emb[b])**2).sum(1); nm = nshared(h[a], h[b])
    loser = np.where(pof[a] & ~pof[b], b, np.where(pof[b] & ~pof[a], a, np.where((sc[a] > sc[b]) | ((sc[a] == sc[b]) & (a < b)), a, b)))
    # cross-clean partners
    p5t5 = np.asarray(d.pT5_t5Idx); p3t3 = np.asarray(d.pT3_t3Idx)
    p3eta = np.asarray(d.pT3_t3_eta); p3phi = np.asarray(d.pT3_t3_phi)
    p3h = o["mdh"][o["t3md"][p3t3]].reshape(len(p3t3), 6) if len(p3t3) else np.zeros((0, 6), int)
    cand = np.flatnonzero(((bits & 1) == 0) & ~pof)   # only these can become T5 TCs
    # T5 TC sim matching
    t5simAll = d.t5_simIdxAll; t5fracAll = d.t5_simIdxAllFrac
    tcsim = np.asarray(d.tc_simIdx); tc_t5 = np.asarray(d.tc_t5Idx); tctype = np.asarray(d.tc_type)
    reco_t5tc = set(tc_t5[tctype == 4].tolist())
    S = sims[sims.evt == ie]
    found = {int(r.isim): r.tcIdx >= 0 for r in S.itertuples()}
    region = {int(r.isim): ("core" if r.dr < 0.02 else ("non" if r.dr >= 0.1 else "mid")) for r in S.itertuples()}
    # sims already matched by some (non-T5-baseline) TC: use reco TC list
    tc_matched_sims = set(tcsim[tcsim >= 0].tolist())
    base_set = None
    for vname, V in VARIANTS.items():
        cond = (((dr2 < 1e-3) | (nm >= 5)) & (d2 < V["d2a"])) | ((dr2 < 0.02) & (d2 < V["d2b"]))
        if "orclose" in V: cond &= (nm >= V["nmin"]) | (d2 < V["orclose"])
        else: cond &= nm >= V["nmin"]
        cond &= (nm >= 5) | ~protmask(V, loser, dnn, sc, t5pt)
        dup = np.zeros(len(eta), bool); dup[loser[cond]] = True
        surv = cand[~dup[cand]]
        # CrossCleanT5 vs pT5
        if len(p5t5) and len(surv):
            ok = np.ones(len(surv), bool)
            for s0 in range(0, len(surv), 2000):
                sv = surv[s0:s0+2000]
                E = np.abs(eta[sv][:, None]-eta[p5t5][None, :]); P = dphi(phi[sv][:, None], phi[p5t5][None, :]); R2 = E*E+P*P
                D2 = ((emb[sv][:, None, :]-emb[p5t5][None, :, :])**2).sum(2)
                c = ((R2 < 0.02) & (D2 < 0.1)) | ((R2 < 1e-3) & (D2 < 1.0))
                if V["nmin"] > 0 and c.any():
                    r_, c_ = np.nonzero(c)
                    nmx = nshared(h[sv[r_]], h[p5t5[c_]])
                    keep = (nmx >= V["nmin"]) | ((D2[r_, c_] < V["orclose"]) if "orclose" in V else False)
                    c[r_[~keep], c_[~keep]] = False
                if c.any():
                    r_, c_ = np.nonzero(c)
                    keep = (nshared(h[sv[r_]], h[p5t5[c_]]) >= 5) | ~protmask(V, sv[r_], dnn, sc, t5pt)
                    c[r_[~keep], c_[~keep]] = False
                ok[s0:s0+2000] &= ~c.any(1)
            surv = surv[ok]
        if len(p3t3) and len(surv):
            E = np.abs(eta[surv][:, None]-p3eta[None, :]); P = dphi(phi[surv][:, None], p3phi[None, :])
            c = (E*E+P*P) < 1e-3
            if c.any():
                r_, c_ = np.nonzero(c); nmx = (h[surv[r_]][:, :, None] == p3h[c_][:, None, :]).any(2).sum(1)
                keep = (nmx >= 3) | ~protmask(V, surv[r_], dnn, sc, t5pt); c[r_[~keep], c_[~keep]] = False
            if V["nmin"] > 0 and c.any():
                r_, c_ = np.nonzero(c); nmx = (h[surv[r_]][:, :, None] == p3h[c_][:, None, :]).any(2).sum(1)
                keep = nmx >= min(V["nmin"], 6); c[r_[~keep], c_[~keep]] = False
            surv = surv[~c.any(1)]
        tg = tight.copy()
        if V.get("tight100"): tg |= np.asarray(d.t5_pt) > 100
        if V.get("tightall"): tg[:] = True
        tcset = set(surv[tg[surv]].tolist())
        if vname == "baseline":
            base_set = tcset
            val[0] += len(tcset & reco_t5tc); val[1] += len(tcset - reco_t5tc); val[2] += len(reco_t5tc - tcset)
            continue
        new = tcset - base_set; gone = base_set - tcset
        if V.get("dump"):
            for x in new:
                ms = [int(s_) for s_, f_ in zip(np.asarray(t5simAll[x]), np.asarray(t5fracAll[x])) if f_ > 0.75]
                dumprows.append(dict(evt=ie, t5=int(x), fake=bool(fake[x]), dnn=float(dnn[x]), score=float(sc[x]), pt=float(d.t5_pt[x]),
                                     eta=float(eta[x]), sims=ms, rec=any((s_ in found and not found[s_]) for s_ in ms),
                                     reg=[region.get(s_, "x") for s_ in ms]))
        R = res[vname]; R["newTC"] += len(new); R["lostTC"] += len(gone)
        R["newFake"] += int(sum(fake[x] for x in new))
        newsims = {}
        for x in new:
            for s_, f_ in zip(np.asarray(t5simAll[x]), np.asarray(t5fracAll[x])):
                if f_ > 0.75: newsims.setdefault(int(s_), 0); newsims[int(s_)] += 1
        for s_, n_ in newsims.items():
            if s_ in found and not found[s_]:
                R[region[s_] + "_rec"] += 1; R["newDup"] += n_ - 1; recov[vname].append((ie, s_))
            else:
                R["newDup"] += n_ if s_ in tc_matched_sims else n_ - 1
        for x in gone:
            for s_, f_ in zip(np.asarray(t5simAll[x]), np.asarray(t5fracAll[x])):
                if f_ > 0.75: R["lostTC_true"] += 1
    if ie % 10 == 0: print(ie, val, {k: (v['core_rec'], v['newTC'], v['newFake']) for k, v in res.items() if k != 'baseline'}, flush=True)
out = [f"baseline replay of T5 TCs: agree {val[0]}, replay-only {val[1]}, reco-only {val[2]}"]
for vname, R in res.items():
    if vname == "baseline": continue
    out.append(f"{vname:28s} core_recovered={R['core_rec']:4d} mid(0.02-0.1)_rec={R['mid_rec']:4d} noncore_rec={R['non_rec']:4d} | newT5TC={R['newTC']:5d} newFake={R['newFake']:5d} newDup={R['newDup']:5d} | baselineT5TC_lost={R['lostTC']} (true-matched {R['lostTC_true']})")
print("\n".join(out)); open("b11_btc_counterfactual.txt", "w").write("\n".join(out) + "\n")
import pickle; pickle.dump(recov, open("b11_recovered.pkl", "wb"))
if dumprows: pd.DataFrame(dumprows).to_pickle("b11_dump.pkl")
