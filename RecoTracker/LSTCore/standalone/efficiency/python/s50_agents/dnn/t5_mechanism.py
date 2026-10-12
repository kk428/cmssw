import numpy as np
import dnnlib as d
z = np.load("t5_replay.npz", allow_pickle=True)
P = z["pairs"]; pc = {k: i for i, k in enumerate(z["pcols"])}; X = z["pfeat"]
reg = P[:, pc["region"]]; pt = P[:, pc["simpt"]]; fr = P[:, pc["frac"]]; ps = P[:, pc["pass"]] > 0
sc = P[:, pc["score"]]; wp = P[:, pc["wp"]]
print("Per-pair T5 DNN pass rate by region x sim-pT x hit purity (frac=1.0 vs <1.0)")
for lo, hi in [(5, 20), (20, 100), (100, 1e9)]:
    for name, rm in [("core", reg == 0), ("dR>=0.1", reg == 1)]:
        b = rm & (pt >= lo) & (pt < hi)
        pure = b & (fr > 0.999); imp = b & (fr < 0.999)
        print(f" pt[{lo},{hi}) {name:8s}: all {ps[b].mean():.3f} (N={b.sum():5d}) | pure {ps[pure].mean() if pure.sum() else np.nan:.3f} (N={pure.sum():5d}) | 8-9/10 hits {ps[imp].mean() if imp.sum() else np.nan:.3f} (N={imp.sum():5d}) | frac-impure {imp.sum()/max(b.sum(),1):.2f}")
print("\nMedian (score - WP) for pure pairs:")
for lo, hi in [(20, 100), (100, 1e9)]:
    for name, rm in [("core", reg == 0), ("dR>=0.1", reg == 1)]:
        b = rm & (pt >= lo) & (pt < hi) & (fr > 0.999)
        print(f" pt[{lo},{hi}) {name}: median margin {np.median(sc[b]-wp[b]):+.3f}, pass {ps[b].mean():.3f}, N={b.sum()}")
# Feature comparison: pure, pt>20, core vs dR>=0.1; standardized mean difference, and per-feature
# 'swap' test: replace one feature in core pairs by matched dR>=0.1 pairs' median (crude attribution)
b0 = (reg == 0) & (pt >= 20) & (fr > 0.999); b1 = (reg == 1) & (pt >= 20) & (fr > 0.999)
print(f"\nFeature shift core vs dR>=0.1 (pure pairs, simpt>=20; N={b0.sum()}, {b1.sum()}):")
sd = X[b0 | b1].std(0) + 1e-12
smd = (X[b0].mean(0) - X[b1].mean(0)) / sd
for i in np.argsort(-np.abs(smd))[:8]:
    print(f"   {d.T5_FEATURE_NAMES[i]:12s} SMD={smd[i]:+.2f}  core mean {X[b0,i].mean():+.4f}  noncore mean {X[b1,i].mean():+.4f}")
# gradient-free attribution: for failing core pure pairs, set feature i to its value after scaling to make it 'non-core like'?
# Instead: within pure pairs, pass rate vs |eta1| and vs sim pT for both regions (eta/pt composition check)
print("\nPure pairs pass rate by |eta| bin (simpt>=20):")
for elo in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
    e = np.abs(P[:, pc["simeta"]])
    for name, rm in [("core", b0), ("dR>=0.1", b1)]:
        b = rm & (e >= elo) & (e < elo + 0.5)
        print(f"   eta[{elo},{elo+0.5}) {name:8s} pass {ps[b].mean() if b.sum() else np.nan:.3f} N={b.sum()}")
# Score vs T5-pt (reco) for pure pairs: is it a high-pT (low curvature) problem?
print("\nPure pairs, all regions: pass rate vs reco T5 pt (inner radius):")
t5pt = P[:, pc["t5pt"]]
for lo, hi in [(5, 10), (10, 20), (20, 50), (50, 100), (100, 300), (300, 1e9)]:
    b = (fr > 0.999) & (t5pt >= lo) & (t5pt < hi)
    for name, rm in [("core", reg == 0), ("dR>=0.1", reg == 1)]:
        bb = b & rm
        print(f"   recoPt[{lo},{hi}) {name:8s} pass {ps[bb].mean() if bb.sum() else np.nan:.3f} N={bb.sum()}")
