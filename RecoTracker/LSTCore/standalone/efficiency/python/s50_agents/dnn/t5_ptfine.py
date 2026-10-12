import numpy as np
z = np.load("t5_replay.npz", allow_pickle=True)
P = z["pairs"]; pc = {k: i for i, k in enumerate(z["pcols"])}; X = z["pfeat"]
reg = P[:, pc["region"]]; fr = P[:, pc["frac"]]; pt = P[:, pc["simpt"]]; ps = P[:, pc["pass"]] > 0
pure = fr > 0.999; lr = np.abs(X[:, 20] - X[:, 22]); lb = np.abs(X[:, 21] - X[:, 20])
print("PURE pairs, fine sim-pT bins: pass rate and radius-consistency (median |log10 Rin/Rout|, |log10 Rbr/Rin|)")
for lo, hi in [(5, 10), (10, 20), (20, 35), (35, 50), (50, 75), (75, 100), (100, 200), (200, 500), (500, 1e9)]:
    s = f" simpt[{lo},{hi}):"
    for name, rm in [("core", reg == 0), ("dR>=0.1", reg == 1), ("0.02-0.1", reg == 2)]:
        b = rm & pure & (pt >= lo) & (pt < hi)
        if b.sum() >= 10:
            s += f"  {name} pass {ps[b].mean():.3f} N={b.sum():4d} lr {np.median(lr[b]):.3f} lb {np.median(lb[b]):.3f} |"
        else:
            s += f"  {name} N={b.sum()} |"
    print(s)
