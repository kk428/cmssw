import numpy as np
z = np.load("t5_replay.npz", allow_pickle=True)
R = z["rows"]; c = {k: i for i, k in enumerate(z["cols"])}
P = z["pairs"]; pc = {k: i for i, k in enumerate(z["pcols"])}
col = lambda k: R[:, c[k]]
key = R[:, c["ev"]] * 100000 + R[:, c["sim"]]
pkey = P[:, pc["ev"]] * 100000 + P[:, pc["sim"]]
rej = P[:, pc["pass"]] == 0
pure = P[:, pc["frac"]] > 0.999
nrej = np.array([np.sum(rej[pkey == k]) for k in key]); nrejpure = np.array([np.sum(rej[(pkey == k) & pure]) for k in key])
nexist = col("n_cand_exist")
for name, m in [("core", col("region") == 0), ("dR>=0.1", col("region") == 1)]:
    f = m & (col("matched") == 0) & (col("n_t5m") > 0)
    s = m & (col("matched") > 0) & (col("n_t5m") > 0)
    print(f"{name}: failing sims WITH matched T5: {f.sum()}; of those with >=1 DNN-rejected good pair: {np.sum(f & (nrej>0))}; with >=1 rejected PURE pair: {np.sum(f & (nrejpure>0))}")
    print(f"   mean #existing good T5 pairs: fail {nexist[f].mean():.1f}  success {nexist[s].mean():.1f};  mean #rejected: fail {nrej[f].mean():.1f}  success {nrej[s].mean():.1f}")
    print(f"   rejected fraction of good pairs: fail {nrej[f].sum()/max(1,(nrej[f]+nexist[f]).sum()):.3f}  success {nrej[s].sum()/max(1,(nrej[s]+nexist[s]).sum()):.3f}")
