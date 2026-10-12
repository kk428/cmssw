#!/usr/bin/env python3
"""b01: build per-sim table (denominator sims only) with per-stage max match fractions.
Core definition copied from s44_eval_fixes.py. Event 89 skipped. Output: b01_simtable.pkl (pandas)."""
import sys, numpy as np, uproot, awkward as ak, pandas as pd
F = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s44_base_100evt.root"
OUT = sys.argv[2] if len(sys.argv) > 2 else "b01_simtable.pkl"
SKIP = {89}
PT_CUT, ETA_CUT, VTX_Z_MAX, VTX_R_MAX, GJ_PT_MIN, GJ_ETA_MAX = 0.8, 4.5, 30.0, 2.5, 1000.0, 2.5
t = uproot.open(F)["tree"]
keys = set(t.keys())
simb = ["sim_pt","sim_eta","sim_phi","sim_q","sim_vx","sim_vy","sim_vz","sim_pca_dxy","sim_pca_dz","sim_pdgId",
        "sim_genjet_deltaR","sim_genjet_idx","genjet_pt","genjet_eta","sim_tcIdx","sim_tcIdxAllFrac","sim_tcIdxAll"]
objs = ["md","ls","t3","t5","pls","pt3","pt5","t4"]
for o in objs:
    if f"sim_{o}IdxAllFrac" in keys: simb += [f"sim_{o}IdxAll", f"sim_{o}IdxAllFrac"]
for b in ["sim_simHitLayer","sim_simHitDetId","sim_recoHitDetId"]:
    if b in keys: simb.append(b)
rows = []
for ie, d in enumerate(t.iterate(simb, step_size=1, library="ak")):
    if ie in SKIP: continue
    d = d[0]
    pt = np.asarray(d.sim_pt, float); n = len(pt)
    gj = np.asarray(d.sim_genjet_idx, np.int32); gjpt = np.asarray(d.genjet_pt, float); gjeta = np.asarray(d.genjet_eta, float)
    c = np.clip(gj, 0, max(len(gjpt)-1, 0))
    gjpt_s = gjpt[c] if len(gjpt) else np.zeros(n); gjeta_s = gjeta[c] if len(gjeta) else np.zeros(n)
    vx = np.asarray(d.sim_vx, float); vy = np.asarray(d.sim_vy, float)
    den = ((np.asarray(d.sim_q) != 0) & (pt > PT_CUT) & (np.abs(np.asarray(d.sim_eta)) < ETA_CUT) &
           (np.abs(np.asarray(d.sim_vz)) < VTX_Z_MAX) & (np.sqrt(vx**2+vy**2) < VTX_R_MAX) &
           (gj >= 0) & (gjpt_s > GJ_PT_MIN) & (np.abs(gjeta_s) < GJ_ETA_MAX))
    dr = np.asarray(d.sim_genjet_deltaR, float)
    for i in np.nonzero(den)[0]:
        r = dict(evt=ie, isim=int(i), pt=pt[i], eta=float(d.sim_eta[i]), phi=float(d.sim_phi[i]), dr=dr[i],
                 dxy=float(d.sim_pca_dxy[i]), dz=float(d.sim_pca_dz[i]), pdg=int(d.sim_pdgId[i]), q=int(d.sim_q[i]),
                 vxy=float(np.hypot(vx[i], vy[i])), gjpt=gjpt_s[i], tcIdx=int(d.sim_tcIdx[i]))
        fr = np.asarray(d.sim_tcIdxAllFrac[i], float); r["tc_maxfrac"] = fr.max() if len(fr) else 0.0
        for o in objs:
            k = f"sim_{o}IdxAllFrac"
            if k in d.fields:
                fr = np.asarray(d[k][i], float)
                r[f"{o}_maxfrac"] = fr.max() if len(fr) else 0.0
                r[f"{o}_n75"] = int((fr > 0.75).sum())
        if "sim_simHitLayer" in d.fields:
            lay = np.asarray(d.sim_simHitLayer[i]); det = np.asarray(d.sim_simHitDetId[i])
            r["nSimHit"] = len(lay)
            r["nSimLayers"] = len(set(lay.tolist()))
        if "sim_recoHitDetId" in d.fields:
            r["nRecoHit"] = len(d.sim_recoHitDetId[i])
        rows.append(r)
    if ie % 10 == 0: print("evt", ie, len(rows), flush=True)
df = pd.DataFrame(rows); df.to_pickle(OUT); print(df.shape)
