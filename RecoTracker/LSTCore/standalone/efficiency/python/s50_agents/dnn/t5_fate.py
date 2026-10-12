"""Fate of the sim-matched T5s (>0.75) of failing denominator sims (S44 base, 99 evts)."""
import numpy as np, uproot
from collections import Counter
F = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s44_base_100evt.root"
z = np.load("t5_replay.npz", allow_pickle=True); R = z["rows"]; c = {k: i for i, k in enumerate(z["cols"])}
t = uproot.open(F)["tree"]
B = ["sim_t5IdxAll", "sim_t5IdxAllFrac", "t5_isDupBits", "t5_tightCutFlag", "t5_partOfPT5", "t5_partOfTC", "t5_tc_idx", "t5_dnnScore",
     "tc_type", "tc_simIdxAll", "tc_simIdxAllFrac", "sim_tcIdxAll", "sim_tcIdxAllFrac", "tc_isFake", "tc_pt5Idx", "tc_t5Idx",
     "pT5_t5Idx", "pT5_isDupReco" ,"t5_pt"]
out = {0: Counter(), 1: Counter()}
dnnfate = {0: [], 1: []}
for ie in np.unique(R[:, c["ev"]]).astype(int):
    rr = R[(R[:, c["ev"]] == ie) & (R[:, c["matched"]] == 0) & (R[:, c["n_t5m"]] > 0) & np.isin(R[:, c["region"]], [0, 1])]
    if len(rr) == 0:
        continue
    a = t.arrays(B, entry_start=ie, entry_stop=ie + 1)[0]
    bits = np.asarray(a.t5_isDupBits); tight = np.asarray(a.t5_tightCutFlag); pt5 = np.asarray(a.t5_partOfPT5); intc = np.asarray(a.t5_partOfTC)
    tctype = np.asarray(a.tc_type)
    for row in rr:
        s = int(row[c["sim"]]); reg = int(row[c["region"]])
        idx = np.asarray(a.sim_t5IdxAll[s]); fr = np.asarray(a.sim_t5IdxAllFrac[s]); m = idx[fr > 0.75]
        # best TC frac for this sim and its type
        tcs = np.asarray(a.sim_tcIdxAll[s]); tcf = np.asarray(a.sim_tcIdxAllFrac[s])
        bestf = tcf.max() if len(tcf) else 0.0
        btype = int(tctype[tcs[np.argmax(tcf)]]) if len(tcf) else -1
        if np.any(intc[m]):
            cat = f"T5 in a TC but TC frac<=0.75 (best TC type {btype})"
        elif np.any(pt5[m]):
            cat = "T5 used in pT5 but that pT5 not a TC / dup-killed"
        elif np.any((bits[m] == 0) & tight[m]):
            cat = "clean tight T5 not in TC (other/cross-clean?)"
        elif np.all(bits[m] != 0):
            b = np.bitwise_or.reduce(bits[m])
            cat = "all matched T5 dup-killed: " + "+".join(n for bb, n in [(1, "AfterBuild"), (2, "BeforeTC(embed)"), (16, "CrossCleanT5")] if b & bb)
            # was any killed only by BeforeTC (embedding)?
        else:
            cat = "surviving T5s all fail tightCutFlag (rz chi2)"
        out[reg][cat] += 1
for reg, name in [(0, "core dR<0.02"), (1, "jet dR>=0.1")]:
    print(f"== {name}: failing sims WITH matched T5 = {sum(out[reg].values())}")
    for k, v in out[reg].most_common():
        print(f"   {v:4d}  {k}")
