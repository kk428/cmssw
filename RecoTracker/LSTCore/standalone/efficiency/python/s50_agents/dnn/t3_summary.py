import numpy as np
z = np.load("t5_replay.npz", allow_pickle=True); R = z["rows"]; c = {k: i for i, k in enumerate(z["cols"])}
y = np.load("t3_replay.npz", allow_pickle=True); Q = y["rows"]; qc = {k: i for i, k in enumerate(y["cols"])}
P = y["pairs"]; pc = {k: i for i, k in enumerate(y["pcols"])}
assert np.all(Q[:, 0] == R[:, c["ev"]]) and np.all(Q[:, 1] == R[:, c["sim"]])
ps = P[:, pc["pass"]] > 0; ex = P[:, pc["exists"]] > 0
print("T3 pairs:", len(P), " exists but fails recomputed DNN (should be 0):", int(np.sum(ex & ~ps)))
col = lambda k: R[:, c[k]]; q = lambda k: Q[:, qc[k]]
reg = col("region")
for name, m, rl in [("core dR<0.02", reg == 0, [0]), ("jet dR>=0.1", reg == 1, [1]), ("inclusive", reg >= 0, [0, 1, 2, 3])]:
    f = m & (col("matched") == 0)
    noT3 = col("n_good_t3") == 0
    nT5 = col("n_t5m") == 0
    hasp = q("n_pair") > 0
    allfail = hasp & (q("n_pass") == 0) & (q("n_exist") == 0)
    print(f"\n=== {name}: failures={f.sum()}")
    print(f"  fail & no T5 & no good T3 : {np.sum(f & nT5 & noT3)}")
    print(f"     with good LS pair(s), ALL fail T3 DNN (upper bound on T3-DNN kill): {np.sum(f & nT5 & noT3 & allfail)}")
    print(f"     with good LS pair(s), some pass T3 DNN but no T3 (pointing/rz/occupancy): {np.sum(f & nT5 & noT3 & hasp & ~allfail)}")
    print(f"     no good LS pair at all: {np.sum(f & nT5 & noT3 & ~hasp)}")
    print(f"  [all failures] with >=1 good-LS-pair whose would-be T3 is missing & fails DNN: {np.sum(f & (q('n_fail_noexist')>0))}")
    pm = np.isin(P[:, pc["region"]], rl)
    print(f"  per-pair T3 DNN pass: {ps[pm].mean():.4f} (N={pm.sum()}), exists: {ex[pm].mean():.4f}, exists|pass: {ex[pm&ps].mean():.4f}")
    for lo, hi in [(0.8, 5), (5, 20), (20, 100), (100, 1e9)]:
        b = pm & (P[:, pc["pt"]] >= lo) & (P[:, pc["pt"]] < hi)
        print(f"     pt[{lo},{hi}): pass={ps[b].mean() if b.sum() else np.nan:.4f} N={b.sum()}  median prompt={np.median(P[b, pc['prompt']]) if b.sum() else np.nan:.3f}")
