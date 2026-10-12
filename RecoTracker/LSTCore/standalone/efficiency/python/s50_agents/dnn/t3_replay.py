"""T3-DNN counterfactual replay (+ T4-candidate count) on S44 base 100-evt ntuple, event 89 skipped.
For each denominator sim: sim-matched LS pairs sharing the middle MD (CreateTriplets pairing), would-be T3
sim-matched (>0.75 of 6 hits from MD-level fractions), T3 DNN evaluated offline with betaIn rebuilt from the
segment dPhiChange (validated vs t3_betaIn), and existence as a stored T3. The pointing and r-z constraints that
run BEFORE the T3 DNN are NOT replayed -> 'fails DNN' is an upper bound on T3-DNN kills."""
import numpy as np, uproot, sys
import dnnlib as d
F = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s44_base_100evt.root"
z = np.load("t5_replay.npz", allow_pickle=True); R = z["rows"]; c = {k: i for i, k in enumerate(z["cols"])}
t = uproot.open(F)["tree"]
B = ["md_anchor_x", "md_anchor_y", "md_anchor_z", "md_layer", "md_simIdxAll", "md_simIdxAllFrac", "ls_mdIdx0", "ls_mdIdx1",
     "t3_lsIdx0", "t3_lsIdx1", "t3_radius", "t3_betaIn", "sim_lsIdxAll", "sim_lsIdxAllFrac", "sim_t3IdxAll", "sim_t3IdxAllFrac",
     "sim_t4IdxAllFrac"]


def red(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def dphichange(ax, ay, az, phi, layer, mi, mo):
    bar = (layer[mi] <= 6) & (layer[mo] <= 6)
    b = red(np.arctan2(ay[mo] - ay[mi], ax[mo] - ax[mi]) - phi[mi])
    dp = red(phi[mo] - phi[mi]); dzf = (az[mo] - az[mi]) / az[mi]
    e = dp / dzf * (1 + dzf)
    return np.where(bar, b, e)


out = []
val = []
for ie in np.unique(R[:, c["ev"]]).astype(int):
    rows = R[R[:, c["ev"]] == ie]
    a = t.arrays(B, entry_start=ie, entry_stop=ie + 1)[0]
    ax, ay, az = (np.asarray(a[k], np.float64) for k in ("md_anchor_x", "md_anchor_y", "md_anchor_z"))
    r = np.hypot(ax, ay); eta = np.abs(np.arcsinh(az / r)); phi = np.arctan2(ay, ax); zz = np.abs(az)
    layer = np.asarray(a.md_layer)
    l0, l1 = np.asarray(a.ls_mdIdx0), np.asarray(a.ls_mdIdx1)
    t0, t1 = np.asarray(a.t3_lsIdx0), np.asarray(a.t3_lsIdx1)
    nls = len(l0)
    t3key = set((t0.astype(np.int64) * nls + t1).tolist())

    def betain(s_in, s_out):
        m1, m2, m3 = l0[s_in], l1[s_in], l1[s_out]
        dpc = dphichange(ax, ay, az, phi, layer, m1, m2)
        return dpc - red(np.arctan2(ay[m3] - ay[m1], ax[m3] - ax[m1]) - phi[m1]), np.stack([m1, m2, m3], 1)

    if len(val) < 5:  # validate on stored T3s
        bi, _ = betain(t0, t1)
        val.append(np.percentile(np.abs(bi - np.asarray(a.t3_betaIn)), [50, 99, 99.9]))
    md_s = a.md_simIdxAll; md_f = a.md_simIdxAllFrac
    for row in rows:
        s = int(row[c["sim"]])
        ls = np.asarray(a.sim_lsIdxAll[s], np.int64); lf = np.asarray(a.sim_lsIdxAllFrac[s])
        ls = ls[lf >= 0.5]
        n_pair = n_exist = n_pass = n_fail_noexist = 0
        if len(ls) >= 2:
            byin = {}
            for j in ls:
                byin.setdefault(l0[j], []).append(j)
            P = [(i, j) for i in ls for j in byin.get(l1[i], [])]
            if P:
                P = np.array(P, np.int64)
                bi, m = betain(P[:, 0], P[:, 1])
                frac = np.zeros(len(P))
                for k in range(3):
                    for q, mdi in enumerate(m[:, k]):
                        ss = np.asarray(md_s[mdi]); ff = np.asarray(md_f[mdi])
                        frac[q] += ff[ss == s].sum()
                frac /= 3
                g = frac > 0.75
                if g.any():
                    P, bi, m = P[g], bi[g], m[g]
                    rad = d.radius3(ax[m[:, 0]], ay[m[:, 0]], ax[m[:, 1]], ay[m[:, 1]], ax[m[:, 2]], ay[m[:, 2]])
                    sc = d.t3_scores(d.t3_features(eta[m], phi[m], zz[m], r[m], rad, bi))
                    ps = d.t3_pass(sc, eta[m[:, 0]], rad)
                    ex = np.array([(p * nls + q) in t3key for p, q in P])
                    n_pair, n_exist, n_pass = len(P), int(ex.sum()), int(ps.sum())
                    n_fail_noexist = int((~ps & ~ex).sum())
                    for q in range(len(P)):
                        val.append(None) if False else None
                        out.append((ie, s, row[c["region"]], row[c["pt"]], ps[q], ex[q], sc[q, 0], sc[q, 1], sc[q, 2]))
        row_ext = (ie, s, n_pair, n_exist, n_pass, n_fail_noexist)
        out_rows = globals().setdefault("OR", [])
        out_rows.append(row_ext)
    print(ie, file=sys.stderr, flush=True)
print("betaIn validation |recomputed - stored| percentiles (50,99,99.9):", [v for v in val if v is not None][:5])
np.savez("t3_replay.npz", pairs=np.array(out, float), pcols=np.array("ev sim region pt pass exists fake prompt disp".split()),
         rows=np.array(OR, float), cols=np.array("ev sim n_pair n_exist n_pass n_fail_noexist".split()))
