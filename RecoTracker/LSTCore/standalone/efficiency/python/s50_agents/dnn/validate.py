import sys, numpy as np, uproot
import dnnlib as d
F = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s44_base_100evt.root"
t = uproot.open(F)["tree"]
br = ["md_anchor_x", "md_anchor_y", "md_anchor_z", "ls_mdIdx0", "ls_mdIdx1", "t3_lsIdx0", "t3_lsIdx1", "t3_radius", "t3_betaIn",
      "t5_t3Idx0", "t5_t3Idx1", "t5_dnnScore", "t5_innerRadius", "t5_outerRadius", "t5_bridgeRadius",
      "t5_t3_fakeScore1", "t5_t3_promptScore1", "t5_t3_displacedScore1"]
a = t.arrays(br, entry_start=0, entry_stop=3, library="np")
for ie in range(3):
    ax, ay, az = [a[k][ie].astype(np.float64) for k in ("md_anchor_x", "md_anchor_y", "md_anchor_z")]
    r = np.hypot(ax, ay); eta = np.abs(np.arcsinh(az / r)); phi = np.arctan2(ay, ax); z = np.abs(az)
    l0, l1 = a["ls_mdIdx0"][ie], a["ls_mdIdx1"][ie]
    t0, t1 = a["t3_lsIdx0"][ie], a["t3_lsIdx1"][ie]
    T3md = np.stack([l0[t0], l0[t1], l1[t1]], 1)
    assert np.all(l1[t0] == l0[t1])
    i0, i1 = a["t5_t3Idx0"][ie], a["t5_t3Idx1"][ie]
    assert np.all(T3md[i0, 2] == T3md[i1, 0])
    m = np.concatenate([T3md[i0], T3md[i1, 1:]], 1)
    rin = a["t3_radius"][ie][i0]; rout = a["t3_radius"][ie][i1]
    rbr = d.radius3(ax[m[:, 1]], ay[m[:, 1]], ax[m[:, 2]], ay[m[:, 2]], ax[m[:, 3]], ay[m[:, 3]])
    print("ev", ie, "nT5", len(i0), "rin rel diff max", np.max(np.abs(rin / a["t5_innerRadius"][ie] - 1)),
          "rbr rel diff 99pct", np.percentile(np.abs(rbr / a["t5_bridgeRadius"][ie] - 1), 99))
    feat = d.t5_features(eta[m], phi[m], z[m], r[m], rin, rbr, rout)
    for w in ("original", "retrained"):
        s = d.t5_score(feat, w)
        diff = np.abs(s - a["t5_dnnScore"][ie])
        ps = d.t5_pass(s, eta[m[:, 0]], rin)
        print(f"  {w}: |dscore| median {np.median(diff):.2e} 99% {np.percentile(diff,99):.2e} max {diff.max():.2e}; frac pass WP {ps.mean():.4f}")
    # T3 DNN validation on inner T3 of each T5
    tm = T3md[i0]
    f3 = d.t3_features(eta[tm], phi[tm], z[tm], r[tm], a["t3_radius"][ie][i0], a["t3_betaIn"][ie][i0])
    sc = d.t3_scores(f3)
    for j, k in enumerate(["t5_t3_fakeScore1", "t5_t3_promptScore1", "t5_t3_displacedScore1"]):
        dd = np.abs(sc[:, j] - a[k][ie])
        print(f"  T3 {k}: median {np.median(dd):.2e} 99% {np.percentile(dd,99):.2e} max {dd.max():.2e}")
    # all T3s pass?
    T3all = T3md
    f3a = d.t3_features(eta[T3all], phi[T3all], z[T3all], r[T3all], a["t3_radius"][ie], a["t3_betaIn"][ie])
    print("  all T3s pass T3 WP frac:", d.t3_pass(d.t3_scores(f3a), eta[T3all[:, 0]], a["t3_radius"][ie]).mean(), "nT3", len(T3all))
