#!/usr/bin/env python3
"""b14: does the TC-stage hit extension (ExtendTrackCandidatesFromDupT5) or anything after object building change the
sim match? Compare each TC's sims (tc_simIdxAll, >0.75) with the sims of the object it was made from."""
import numpy as np, uproot, pandas as pd
from bcommon import F100
sims = pd.read_pickle("b01_simtable.pkl")
t = uproot.open(F100)["tree"]
br = ["tc_type","tc_pt5Idx","tc_t5Idx","tc_pt3Idx","tc_plsIdx","tc_simIdxAll","tc_nhitOT","tc_nhits","pT5_simIdxAll","t5_simIdxAll","pT3_simIdxAll","pLS_simIdxAll","sim_tcIdx"]
rows = []
for ie, d in enumerate(t.iterate(br, step_size=1, library="ak")):
    if ie == 89: continue
    d = d[0]
    den = {int(r.isim): r.dr for r in sims[sims.evt == ie].itertuples()}
    for k, ty in enumerate(np.asarray(d.tc_type)):
        if ty == 7: obj = set(np.asarray(d.pT5_simIdxAll[d.tc_pt5Idx[k]]).tolist())
        elif ty == 4: obj = set(np.asarray(d.t5_simIdxAll[d.tc_t5Idx[k]]).tolist())
        elif ty == 5: obj = set(np.asarray(d.pT3_simIdxAll[d.tc_pt3Idx[k]]).tolist())
        elif ty == 8: obj = set(np.asarray(d.pLS_simIdxAll[d.tc_plsIdx[k]]).tolist())
        else: continue
        tcs = set(np.asarray(d.tc_simIdxAll[k]).tolist())
        for s in obj - tcs:
            rows.append(dict(evt=ie, tc=k, type=int(ty), sim=s, lost_match=True, den=s in den, dr=den.get(s, -1),
                             simfound=int(d.sim_tcIdx[s]) >= 0 if s < len(d.sim_tcIdx) else None, nhits=int(d.tc_nhits[k])))
df = pd.DataFrame(rows); df.to_pickle("b14_tcext.pkl")
print("object-matched sims that the final TC no longer matches:", len(df))
print(df.groupby("type").size().to_dict())
x = df[df.den]
for name, sel in [("core", x.dr < 0.02), ("mid", (x.dr >= 0.02) & (x.dr < 0.1)), ("noncore", x.dr >= 0.1)]:
    y = x[sel]; print(name, "denominator sims affected:", len(y), " of which ended up NOT found:", int((~y.simfound.astype(bool)).sum()), " by type", y.groupby("type").size().to_dict())
