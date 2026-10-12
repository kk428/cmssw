#!/usr/bin/env python3
"""b06: pT5-level diagnosis for lost sims whose matched T5s were absorbed into pT5s (partOfPT5).
For each such sim: its own pLSs (quad?, isDup bits), pT5s built from its pLS (any T5), pT5s built from its T5s (any pLS),
and their dedup / TC fate; the sim matched by the pT5 TC that absorbed its T5."""
import numpy as np, uproot, pandas as pd, awkward as ak
from bcommon import F100
sims = pd.read_pickle("b01_simtable.pkl"); fate = pd.read_pickle("b02_t5fate.pkl")
t = uproot.open(F100)["tree"]
br = ["sim_t5IdxAll","sim_t5IdxAllFrac","sim_plsIdxAll","sim_plsIdxAllFrac","sim_pt5IdxAll","sim_pt5IdxAllFrac",
      "pT5_t5Idx","pT5_plsIdx","pT5_isDupReco","pT5_score","pT5_simIdx","t5_simIdx","t5_isDupBits",
      "pLS_isQuad","pLS_isDup","pLS_simIdx","pLS_pt","pLS_eta","pLS_phi","tc_pt5Idx","tc_simIdx","tc_type","sim_tcIdx","pT3_plsIdx"]
rows = []
for ie, d in enumerate(t.iterate(br, step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]; S = sims[sims.evt == ie]
    p5t5 = np.asarray(d.pT5_t5Idx); p5pls = np.asarray(d.pT5_plsIdx); p5dup = np.asarray(d.pT5_isDupReco)
    tcp5 = np.asarray(d.tc_pt5Idx); tcsim = np.asarray(d.tc_simIdx)
    p5tc = {int(p): k for k, p in enumerate(tcp5) if p >= 0}
    plsq = np.asarray(d.pLS_isQuad); plsd = np.asarray(d.pLS_isDup); plssim = np.asarray(d.pLS_simIdx)
    pt3pls = set(np.asarray(d.pT3_plsIdx).tolist())
    for _, s in S.iterrows():
        i = int(s.isim)
        t5s = np.asarray(d.sim_t5IdxAll[i])[np.asarray(d.sim_t5IdxAllFrac[i]) > 0.75]
        pls = np.asarray(d.sim_plsIdxAll[i])[np.asarray(d.sim_plsIdxAllFrac[i]) > 0.75]
        mT5 = np.isin(p5t5, t5s); mP = np.isin(p5pls, pls)
        r = dict(evt=ie, isim=i, dr=s.dr, pt=s.pt, found=s.tcIdx >= 0,
                 nT5=len(t5s), nT5_alive_AB=int(((np.asarray(d.t5_isDupBits)[t5s] & 1) == 0).sum()) if len(t5s) else 0,
                 nPLS=len(pls), nPLSquad=int(plsq[pls].sum()) if len(pls) else 0,
                 nPLS_pass1alive=int(((plsd[pls] & 1) == 0).sum()) if len(pls) else 0,
                 pls_in_pT3=int(any(p in pt3pls for p in pls)),
                 n_pT5_ownT5=int(mT5.sum()), n_pT5_ownPLS=int(mP.sum()), n_pT5_both=int((mT5 & mP).sum()),
                 n_pT5_both_alive=int((mT5 & mP & (p5dup == 0)).sum()),
                 n_pT5_ownT5_alive=int((mT5 & (p5dup == 0)).sum()), n_pT5_ownPLS_alive=int((mP & (p5dup == 0)).sum()))
        # TCs that absorbed own T5 (pT5 TCs containing own T5): matched sim
        absorbers = [p5tc[k] for k in np.flatnonzero(mT5) if k in p5tc]
        r["absorber_tc_sims"] = [int(tcsim[k]) for k in absorbers]
        r["absorber_pls_sims"] = [int(plssim[p5pls[k]]) for k in np.flatnonzero(mT5) if k in p5tc]
        # own-pLS pT5 TCs: which sim do they match (own pLS + someone else's T5)
        r["ownpls_tc_sims"] = [int(tcsim[p5tc[k]]) for k in np.flatnonzero(mP) if k in p5tc]
        rows.append(r)
pd.DataFrame(rows).to_pickle("b06_pt5.pkl"); print(len(rows))
