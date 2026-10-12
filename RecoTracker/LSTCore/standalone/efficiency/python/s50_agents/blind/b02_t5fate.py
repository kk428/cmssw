#!/usr/bin/env python3
"""b02: for each denominator sim, list the fates of its matched (frac>0.75) T5s, pT5s, pLSs.
T5 isDup bits (from Kernels.h/TrackCandidate.h): 1=RemoveDupQuintupletsAfterBuild, 2=RemoveDupQuintupletsBeforeTC,
0x10=CrossCleanT5. TC requires !isDup && !partOfPT5 && tightCutFlag (AddT5asTrackCandidate).
Output b02_t5fate.pkl : one row per (sim, matched T5)."""
import numpy as np, uproot, awkward as ak, pandas as pd
F = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s44_base_100evt.root"
sims = pd.read_pickle("b01_simtable.pkl")
t = uproot.open(F)["tree"]
br = ["sim_t5IdxAll","sim_t5IdxAllFrac","sim_pt5IdxAll","sim_plsIdxAll","sim_plsIdxAllFrac",
      "t5_isDupBits","t5_partOfPT5","t5_triedInPT5","t5_tightCutFlag","t5_dnnScore","t5_score","t5_pt","t5_eta","t5_phi",
      "t5_simIdxAll","t5_simIdxAllFrac","t5_embed","t5_t3Idx0","t5_t3Idx1",
      "pT5_t5Idx","pT5_plsIdx","pT5_isDupReco","pT5_score","pT5_simIdxAll","pT5_simIdxAllFrac",
      "pLS_simIdxAll","pLS_isQuad","pLS_isDup","pLS_pt","pLS_eta","pLS_phi",
      "tc_type","tc_t5Idx","tc_pt5Idx","tc_simIdxAll","tc_simIdxAllFrac"]
rows = []
for ie, d in enumerate(t.iterate(br, step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]
    S = sims[sims.evt == ie]
    if len(S) == 0: continue
    pT5_t5 = np.asarray(d.pT5_t5Idx); pT5_pls = np.asarray(d.pT5_plsIdx); pT5_dup = np.asarray(d.pT5_isDupReco)
    pT5_score = np.asarray(d.pT5_score)
    tc_type = np.asarray(d.tc_type); tc_pt5 = np.asarray(d.tc_pt5Idx); tc_t5 = np.asarray(d.tc_t5Idx)
    pt5_in_tc = set(tc_pt5[tc_pt5 >= 0].tolist()); t5_in_tc = set(tc_t5[(tc_t5 >= 0)].tolist())
    # map T5 -> list of pT5 indices
    t52pt5 = {}
    for k, t5i in enumerate(pT5_t5): t52pt5.setdefault(int(t5i), []).append(k)
    bits = np.asarray(d.t5_isDupBits); pof = np.asarray(d.t5_partOfPT5); tried = np.asarray(d.t5_triedInPT5)
    tight = np.asarray(d.t5_tightCutFlag); dnn = np.asarray(d.t5_dnnScore); sc = np.asarray(d.t5_score)
    pls_sim = d.pLS_simIdxAll
    for _, s in S.iterrows():
        i = int(s.isim)
        t5s = np.asarray(d.sim_t5IdxAll[i]); fr = np.asarray(d.sim_t5IdxAllFrac[i])
        for t5i, f in zip(t5s, fr):
            if f <= 0.75: continue
            pt5s = t52pt5.get(int(t5i), [])
            # pT5s using this T5: how many, survived?, in TC?, pLS belongs to this sim?
            pt5_info = []
            for k in pt5s:
                ps = list(pls_sim[pT5_pls[k]]) if pT5_pls[k] >= 0 else []
                pt5_info.append((k, int(pT5_dup[k]), k in pt5_in_tc, i in ps, float(pT5_score[k])))
            rows.append(dict(evt=ie, isim=i, dr=s.dr, pt=s.pt, eta=s.eta, found=s.tcIdx >= 0, t5=int(t5i), frac=float(f),
                             bits=int(bits[t5i]), partOfPT5=int(pof[t5i]), tried=int(tried[t5i]), tight=int(tight[t5i]),
                             dnn=float(dnn[t5i]), score=float(sc[t5i]), t5pt=float(d.t5_pt[t5i]),
                             t5InTC=int(t5i) in t5_in_tc,
                             n_pt5=len(pt5s), n_pt5_alive=sum(1 for x in pt5_info if x[1] == 0),
                             n_pt5_inTC=sum(1 for x in pt5_info if x[2]),
                             n_pt5_rightpLS=sum(1 for x in pt5_info if x[3]),
                             n_pt5_rightpLS_alive=sum(1 for x in pt5_info if x[3] and x[1] == 0),
                             n_pt5_wrongpLS_inTC=sum(1 for x in pt5_info if (not x[3]) and x[2])))
    if ie % 10 == 0: print(ie, len(rows), flush=True)
df = pd.DataFrame(rows); df.to_pickle("b02_t5fate.pkl"); print(df.shape)
