#!/usr/bin/env python3
"""b15: combined offline replay of the TC-level selection downstream of pT5 building, to estimate (i) ceilings and
(ii) combined realistic fixes on one consistent footing. Chain replayed:
  [pT5 set (optionally oracle-remove unmatched pT5s)] -> pT5 dedup (ranking key) -> partOfPT5 of T5s = used by any
  built pT5 (as in the kernel, set at creation) -> BeforeTC dedup on non-AB-dup T5s -> CrossCleanT5 vs all pT5s/pT3s
  -> T5 TCs (tightCutFlag, optional pT>100 fix).
'found' = object-level match (>0.75) of: baseline pT3/pLS/T4 TCs (unchanged) + surviving pT5s + T5 TCs.
Deltas are quoted w.r.t. this replay's own baseline (object-level; the TC hit-extension loss of b14 is NOT modelled)."""
import numpy as np, uproot, pandas as pd, awkward as ak, sys
from bcommon import *
sims = pd.read_pickle("b01_simtable.pkl")
def dphi(a, b): return np.abs((a - b + np.pi) % (2*np.pi) - np.pi)
CONF = {
 "baseline": dict(),
 "A pT5 key = pT5chi2*(1-T5dnn)": dict(key="mix"),
 "B tight100 + protect(chi2<3,pt>50)": dict(tight100=True, pchi=3.0, ppt=50.0),
 "C tight100 + protect(chi2<3,pt>20)": dict(tight100=True, pchi=3.0, ppt=20.0),
 "A+B": dict(key="mix", tight100=True, pchi=3.0, ppt=50.0),
 "A+C": dict(key="mix", tight100=True, pchi=3.0, ppt=20.0),
 "CEIL pT5 dedup oracle": dict(key="oracle"),
 "CEIL no fake pT5 built (oracle)": dict(nofakept5=True),
 "CEIL no fake pT5 + pT5 oracle + tight100 + no BTC/xclean": dict(nofakept5=True, key="oracle", tight100=True, nobtc=True),
}
t = uproot.open(F100)["tree"]
res = {c: dict(core=0, mid=0, non=0, nTC=0, nFake=0) for c in CONF}
perevt = []
for ie, d in enumerate(t.iterate(BR + ["pT3_t3_eta","pT3_t3_phi","pT3_simIdxAll","tc_simIdxAll"], step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]; o = build(d); h = o["t5h"]
    S = sims[sims.evt == ie]; reg = {int(r.isim): ("core" if r.dr < 0.02 else ("non" if r.dr >= 0.1 else "mid")) for r in S.itertuples()}
    tct = np.asarray(d.tc_type)
    other = set()
    for k in np.flatnonzero((tct == 5) | (tct == 8) | (tct == 9)): other |= set(np.asarray(d.tc_simIdxAll[k]).tolist())
    nOtherTC = int(((tct == 5) | (tct == 8) | (tct == 9)).sum()); nOtherFake = int(np.asarray(d.tc_isFake)[(tct == 5) | (tct == 8) | (tct == 9)].sum())
    eta = np.asarray(d.t5_eta); phi = np.asarray(d.t5_phi); sc = np.asarray(d.t5_score); bits = np.asarray(d.t5_isDupBits)
    tight0 = np.asarray(d.t5_tightCutFlag).astype(bool); emb = ak.to_numpy(d.t5_embed).astype(np.float32)
    fake = np.asarray(d.t5_isFake).astype(bool); dnn = np.asarray(d.t5_dnnScore); t5pt = np.asarray(d.t5_pt)
    t5sims = [set(np.asarray(x).tolist()) for x in d.t5_simIdxAll]
    p5t5 = np.asarray(d.pT5_t5Idx); p5pls = np.asarray(d.pT5_plsIdx); n5 = len(p5t5)
    p5sims = [set(np.asarray(x).tolist()) for x in d.pT5_simIdxAll]; p5fake = np.array([len(x) == 0 for x in p5sims], bool)
    H5 = np.concatenate([o["plsh"][p5pls], h[p5t5]], 1) if n5 else np.zeros((0, 14), int)
    p3t3 = np.asarray(d.pT3_t3Idx); p3eta = np.asarray(d.pT3_t3_eta); p3phi = np.asarray(d.pT3_t3_phi)
    p3h = o["mdh"][o["t3md"][p3t3]].reshape(len(p3t3), 6) if len(p3t3) else np.zeros((0, 6), int)
    abalive = np.flatnonzero((bits & 1) == 0)
    # BTC candidate pairs (independent of config except pof)
    oo = abalive[np.argsort(eta[abalive], kind="stable")]; e = eta[oo]; A = []; B = []
    for kk in range(1, len(oo)):
        m = (e[kk:] - e[:-kk]) <= 0.1 + 1e-7
        if not m.any(): break
        A.append(oo[:-kk][m]); B.append(oo[kk:][m])
    pa = np.concatenate(A) if A else np.zeros(0, int); pb = np.concatenate(B) if B else np.zeros(0, int)
    k = dphi(phi[pa], phi[pb]) <= 0.1; pa, pb = pa[k], pb[k]
    de = eta[pa]-eta[pb]; dp = dphi(phi[pa], phi[pb]); dr2 = de*de+dp*dp; d2 = ((emb[pa]-emb[pb])**2).sum(1); nm = nshared(h[pa], h[pb])
    basecond = (((dr2 < 1e-3) | (nm >= 5)) & (d2 < 1.0)) | ((dr2 < 0.02) & (d2 < 0.1))
    for cname, C in CONF.items():
        keep5 = ~p5fake if C.get("nofakept5") else np.ones(n5, bool)
        idx5 = np.flatnonzero(keep5)
        # pT5 dedup among existing
        rm = np.zeros(n5, bool)
        if len(idx5):
            ii, jj = np.meshgrid(idx5, idx5, indexing="ij"); ii = ii.ravel(); jj = jj.ravel()
            e5 = eta[p5t5]; f5 = phi[p5t5]
            kq = (ii != jj) & (np.abs(e5[ii]-e5[jj]) <= 0.2) & (dphi(f5[ii], f5[jj]) <= 0.2); ii, jj = ii[kq], jj[kq]
            kq = nshared(H5[ii], H5[jj]) >= 7; ii, jj = ii[kq], jj[kq]
            s5 = np.asarray(d.pT5_score).astype(np.float64)
            if C.get("key") == "mix": s5 = s5 * (1 - dnn[p5t5]) + 1e-9 * s5
            if C.get("key") == "oracle": s5 = p5fake * 1e6 + s5
            kill = (s5[ii] > s5[jj]) | ((s5[ii] == s5[jj]) & (ii > jj)); rm[ii[kill]] = True
        surv5 = idx5[~rm[idx5]]
        pof = np.zeros(len(eta), bool); pof[p5t5[idx5]] = True
        # BTC
        tg = tight0.copy()
        if C.get("tight100"): tg |= t5pt > 100
        cand = np.flatnonzero(((bits & 1) == 0) & ~pof)
        if C.get("nobtc"):
            surv = cand
        else:
            kp = ~(pof[pa] & pof[pb])
            loser = np.where(pof[pa] & ~pof[pb], pb, np.where(pof[pb] & ~pof[pa], pa, np.where((sc[pa] > sc[pb]) | ((sc[pa] == sc[pb]) & (pa < pb)), pa, pb)))
            cond = basecond & kp
            if "pchi" in C:
                prot = (sc[loser] < C["pchi"]) & (t5pt[loser] > C["ppt"]); cond &= (nm >= 5) | ~prot
            dup = np.zeros(len(eta), bool); dup[loser[cond]] = True
            surv = cand[~dup[cand]]
            if len(idx5) and len(surv):  # CrossCleanT5 vs all (built) pT5s
                t5of = p5t5[idx5]; ok = np.ones(len(surv), bool)
                for s0 in range(0, len(surv), 2000):
                    sv = surv[s0:s0+2000]
                    E = np.abs(eta[sv][:, None]-eta[t5of][None, :]); P = dphi(phi[sv][:, None], phi[t5of][None, :]); R2 = E*E+P*P
                    D2 = ((emb[sv][:, None, :]-emb[t5of][None, :, :])**2).sum(2)
                    c = ((R2 < 0.02) & (D2 < 0.1)) | ((R2 < 1e-3) & (D2 < 1.0))
                    if "pchi" in C and c.any():
                        r_, c_ = np.nonzero(c)
                        prot = (sc[sv[r_]] < C["pchi"]) & (t5pt[sv[r_]] > C["ppt"])
                        kp2 = (nshared(h[sv[r_]], h[t5of[c_]]) >= 5) | ~prot; c[r_[~kp2], c_[~kp2]] = False
                    ok[s0:s0+2000] &= ~c.any(1)
                surv = surv[ok]
            if len(p3t3) and len(surv):
                E = np.abs(eta[surv][:, None]-p3eta[None, :]); P = dphi(phi[surv][:, None], p3phi[None, :]); c = (E*E+P*P) < 1e-3
                if "pchi" in C and c.any():
                    r_, c_ = np.nonzero(c); nmx = (h[surv[r_]][:, :, None] == p3h[c_][:, None, :]).any(2).sum(1)
                    prot = (sc[surv[r_]] < C["pchi"]) & (t5pt[surv[r_]] > C["ppt"]); kp3 = (nmx >= 3) | ~prot; c[r_[~kp3], c_[~kp3]] = False
                surv = surv[~c.any(1)]
        t5tc = surv[tg[surv]]
        found = set(other)
        for x in surv5: found |= p5sims[x]
        for x in t5tc: found |= t5sims[x]
        R = res[cname]
        pe = dict(evt=ie, conf=cname, core=0, mid=0, non=0, nFake=nOtherFake + int(p5fake[surv5].sum()) + int(fake[t5tc].sum()))
        for i, r in reg.items():
            if i in found: R[r] += 1; pe[r] += 1
        perevt.append(pe)
        R["nTC"] += nOtherTC + len(surv5) + len(t5tc); R["nFake"] += nOtherFake + int(p5fake[surv5].sum()) + int(fake[t5tc].sum())
    if ie % 20 == 0: print(ie, {c: (v["core"], v["nTC"]) for c, v in res.items()}, flush=True)
b = res["baseline"]
out = ["object-level replay; core den 1210, mid 1607, noncore 1633 (99 events). Deltas vs replay baseline."]
for c, R in res.items():
    out.append(f"{c:58s} core={R['core']:4d} ({R['core']-b['core']:+4d}) mid={R['mid']:4d} ({R['mid']-b['mid']:+4d}) non={R['non']:4d} ({R['non']-b['non']:+4d}) | TCs={R['nTC']} fakeTCs={R['nFake']} ({R['nFake']-b['nFake']:+d}) fakerate={R['nFake']/R['nTC']:.4f}")
P = pd.DataFrame(perevt); P.to_pickle("b15_perevt.pkl")
P["half"] = np.where(P.evt % 2 == 0, "even", "odd")
g = P.groupby(["conf","half"])[["core","mid","non","nFake"]].sum()
base = g.xs("baseline", level=0)
out.append("even/odd event split (delta vs baseline in each half):")
for c in CONF:
    if c == "baseline": continue
    dd = g.xs(c, level=0) - base
    out.append(f"  {c:58s} even: core {dd.loc['even','core']:+d} mid {dd.loc['even','mid']:+d} fake {dd.loc['even','nFake']:+d} | odd: core {dd.loc['odd','core']:+d} mid {dd.loc['odd','mid']:+d} fake {dd.loc['odd','nFake']:+d}")
txt = "\n".join(out); print(txt); open("b15_combined.txt", "w").write(txt + "\n")
