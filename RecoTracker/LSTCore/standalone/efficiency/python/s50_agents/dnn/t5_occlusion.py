"""Which T5-DNN inputs drive the low scores of PURE (10/10 hits) sim-matched core candidate pairs?
Occlusion test: for core pure pairs that FAIL the WP (sim pT>=20), replace one feature group by the value
of a randomly drawn dR>=0.1 pure pair in the same reco-pT bin and same |eta1| bin (0.25), re-score, and
report the fraction that flips to pass (10 random draws averaged). Also the same for passing non-core
pairs (control: how many flip to fail when given core values)."""
import numpy as np
import dnnlib as d
rng = np.random.default_rng(1)
z = np.load("t5_replay.npz", allow_pickle=True)
P = z["pairs"]; pc = {k: i for i, k in enumerate(z["pcols"])}; X = z["pfeat"]
reg = P[:, pc["region"]]; fr = P[:, pc["frac"]]; pt = P[:, pc["simpt"]]; t5pt = P[:, pc["t5pt"]]
e1 = X[:, 0] * d.T5_ETA_NORM; rin = 10 ** X[:, 20]
pure = fr > 0.999
ptbins = np.digitize(t5pt, [10, 20, 50, 100, 300]); ebins = d.etabin(e1)
groups = {"hit1 position (eta1,|phi1|,z1,r1)": [0, 1, 2, 3], "deta(all)": [4, 8, 12, 16], "dphi(all)": [5, 9, 13, 17],
          "dz(all)": [6, 10, 14, 18], "dr(all)": [7, 11, 15, 19], "radii (Rin,Rbr,Rout)": [20, 21, 22],
          "Rbridge only": [21], "Rout only": [22], "inner-T3 diffs (21,32)": [4, 5, 6, 7, 8, 9, 10, 11], "outer diffs (43,54)": [12, 13, 14, 15, 16, 17, 18, 19]}
def flip(src_mask, ref_mask, want_pass):
    idx = np.nonzero(src_mask)[0]
    res = {}
    for g, cols in groups.items():
        fl = []
        for _ in range(10):
            Xs = X[idx].copy(); ok = np.ones(len(idx), bool)
            for k, i in enumerate(idx):
                cand = np.nonzero(ref_mask & (ptbins == ptbins[i]) & (ebins == ebins[i]))[0]
                if len(cand) == 0:
                    ok[k] = False; continue
                Xs[k, cols] = X[rng.choice(cand), cols]
            s = d.t5_score(Xs)
            p = d.t5_pass(s, Xs[:, 0] * d.T5_ETA_NORM, 10 ** Xs[:, 20])
            fl.append(np.mean((p == want_pass)[ok]))
        res[g] = np.mean(fl)
    return res
core_fail = (reg == 0) & pure & (pt >= 20) & (P[:, pc["pass"]] == 0)
nonc_pure = (reg == 1) & pure
nonc_pass = (reg == 1) & pure & (pt >= 20) & (P[:, pc["pass"]] > 0)
core_pure = (reg == 0) & pure
print(f"core pure failing pairs (simpt>=20): N={core_fail.sum()}; donor: dR>=0.1 pure pairs N={nonc_pure.sum()}")
r1 = flip(core_fail, nonc_pure, True)
for g, v in sorted(r1.items(), key=lambda x: -x[1]):
    print(f"   swap {g:38s}: {v:.3f} flip to PASS")
print(f"\ncontrol: dR>=0.1 pure passing pairs (simpt>=20) N={nonc_pass.sum()} given CORE donor values -> fraction flipping to FAIL")
r2 = flip(nonc_pass, core_pure, False)
for g, v in sorted(r2.items(), key=lambda x: -x[1]):
    print(f"   swap {g:38s}: {v:.3f} flip to FAIL")
# radius consistency
lr = X[:, 20] - X[:, 22]; lb = X[:, 21] - X[:, 20]
for name, m in [("core", (reg == 0) & pure & (pt >= 20)), ("dR>=0.1", (reg == 1) & pure & (pt >= 20))]:
    print(f"{name}: |log10(Rin/Rout)| median {np.median(np.abs(lr[m])):.3f} 90% {np.percentile(np.abs(lr[m]),90):.3f};  |log10(Rbr/Rin)| median {np.median(np.abs(lb[m])):.3f}")
