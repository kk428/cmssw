import numpy as np
z = np.load("t5_replay.npz", allow_pickle=True)
R = z["rows"]; c = {k: i for i, k in enumerate(z["cols"])}
P = z["pairs"]; pc = {k: i for i, k in enumerate(z["pcols"])}
col = lambda k: R[:, c[k]]
reg = col("region")
regions = {"core dR<0.02": reg == 0, "jet dR>=0.1": reg == 1, "jet all": reg <= 2, "inclusive (all denom)": reg >= 0}
# sanity: P(pass) && exists consistency
ex = P[:, pc["exists"]] > 0; ps = P[:, pc["pass"]] > 0
print("pairs total", len(P), " exist&fail-DNN (should be ~0):", int(np.sum(ex & ~ps)), " exist:", int(ex.sum()))
for name, m in regions.items():
    den = m.sum(); num = (m & (col("matched") > 0)).sum()
    f = m & (col("matched") == 0)
    nf = f.sum()
    has_t5 = col("n_t5m") > 0
    cand = col("n_cand") > 0
    allfail = cand & (col("n_cand_pass") == 0)
    passnoex = cand & (col("n_cand_pass") > 0) & ~has_t5
    print(f"\n=== {name}: den={den} eff={num/den:.4f} failures={nf}")
    print(f"  fail & has matched T5 in ntuple            : {np.sum(f & has_t5)}")
    print(f"  fail & no T5, cand pairs, ALL fail T5 DNN   : {np.sum(f & ~has_t5 & allfail)}")
    print(f"  fail & no T5, cand pair passes DNN (killed later): {np.sum(f & ~has_t5 & passnoex)}")
    print(f"  fail & no T5, no good cand pair             : {np.sum(f & ~has_t5 & ~cand)}")
    print(f"     of which n_good_t3(>0.75)==0            : {np.sum(f & ~has_t5 & ~cand & (col('n_good_t3')==0))}; n_good_ls==0: {np.sum(f & ~has_t5 & ~cand & (col('n_good_ls')==0))}")
    # same for successes, to see if DNN-killed T5s matter
    s = m & (col("matched") > 0)
    print(f"  [success sims] no T5 & all cand fail DNN    : {np.sum(s & ~has_t5 & allfail)} / {s.sum()}")
    # among ALL sims with candidates: fraction all-fail
    print(f"  sims w/ cand: {np.sum(m & cand)};  all-cand-fail-DNN frac = {np.sum(m&allfail)/max(1,np.sum(m&cand)):.4f}")
    # per pair pass rate
    pm = np.isin(P[:, pc["region"]], {"core dR<0.02": [0], "jet dR>=0.1": [1], "jet all": [0, 1, 2], "inclusive (all denom)": [0, 1, 2, 3]}[name])
    print(f"  per-pair DNN pass rate: {ps[pm].mean():.4f} (N={pm.sum()});  exists|pass: {ex[pm & ps].mean():.4f}")
    # per sim pt bins
    for lo, hi in [(0.8, 2), (2, 5), (5, 20), (20, 100), (100, 1e9)]:
        b = m & (col("pt") >= lo) & (col("pt") < hi)
        bp = pm & (P[:, pc["simpt"]] >= lo) & (P[:, pc["simpt"]] < hi)
        print(f"    pt[{lo},{hi}): den={b.sum():5d} eff={np.mean(col('matched')[b]) if b.sum() else np.nan:.3f}  sims-with-cand all-fail frac={np.sum(b&allfail)/max(1,np.sum(b&cand)):.4f} (n={np.sum(b&cand)})  pair pass={ps[bp].mean() if bp.sum() else np.nan:.4f} (N={bp.sum()})")
