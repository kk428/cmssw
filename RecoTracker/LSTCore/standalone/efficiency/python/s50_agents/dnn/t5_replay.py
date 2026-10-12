"""T5-DNN counterfactual replay on S44 base 100-evt ntuple (event 89 skipped).
For every denominator sim, enumerate sim-matched T3 pairs that share the middle MD (exactly the
CreateQuintuplets pairing), require the would-be T5 to be sim-matched (>0.75 of 10 hits, from MD-level
sim fractions), evaluate the T5 DNN offline (validated to 1e-4 vs t5_dnnScore), and check whether the pair
exists as a stored T5. Output: per-sim table (npz) for downstream summaries."""
import numpy as np, uproot, awkward as ak, sys
import dnnlib as d

F = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s44_base_100evt.root"
SKIP = {89}
t = uproot.open(F)["tree"]
SIMB = ["sim_pt", "sim_eta", "sim_phi", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_deltaR", "sim_genjet_idx",
        "genjet_pt", "genjet_eta", "sim_tcIdx", "sim_t3IdxAll", "sim_t3IdxAllFrac", "sim_t5IdxAll", "sim_t5IdxAllFrac",
        "sim_lsIdxAll", "sim_lsIdxAllFrac", "sim_mdIdxAll", "sim_pt5IdxAll", "sim_pt5IdxAllFrac", "sim_pt3IdxAll", "sim_pt3IdxAllFrac",
        "sim_plsIdxAll", "sim_plsIdxAllFrac", "sim_t4IdxAll", "sim_t4IdxAllFrac", "sim_tcIdxBestFrac"]
OBJB = ["md_anchor_x", "md_anchor_y", "md_anchor_z", "md_layer", "md_simIdxAll", "md_simIdxAllFrac", "ls_mdIdx0", "ls_mdIdx1",
        "t3_lsIdx0", "t3_lsIdx1", "t3_radius", "t5_t3Idx0", "t5_t3Idx1", "t5_dnnScore", "t5_isDupBits",
        "t5_tightCutFlag", "t5_partOfPT5", "t5_partOfTC", "t5_simIdxAll", "t5_simIdxAllFrac"]

rows = []      # per sim
pairs = []     # per good candidate pair: (ev, sim, core-region code, pt, eta, score, wp, pass, exists, feats...)
pfeat = []
first_layers = None
for ie in range(t.num_entries):
    if ie in SKIP:
        continue
    a = t.arrays(SIMB + OBJB, entry_start=ie, entry_stop=ie + 1)
    a = a[0]
    # --- denominator ---
    sim_pt = np.asarray(a.sim_pt, float); n_sim = len(sim_pt)
    gj = np.asarray(a.sim_genjet_idx, np.int32); gjpt = np.asarray(a.genjet_pt, float); gjeta = np.asarray(a.genjet_eta, float)
    gc = np.clip(gj, 0, max(len(gjpt) - 1, 0))
    gpt = gjpt[gc] if len(gjpt) else np.zeros(n_sim); geta = gjeta[gc] if len(gjeta) else np.zeros(n_sim)
    vx, vy, vz = (np.asarray(a[k], float) for k in ("sim_vx", "sim_vy", "sim_vz"))
    base = (np.asarray(a.sim_q) != 0) & (sim_pt > 0.8) & (np.abs(np.asarray(a.sim_eta)) < 4.5) & (np.abs(vz) < 30) & (np.hypot(vx, vy) < 2.5)
    injet = (gj >= 0) & (gpt > 1000) & (np.abs(geta) < 2.5)
    dr = np.asarray(a.sim_genjet_deltaR, float)
    matched = np.asarray(a.sim_tcIdx, np.int64) >= 0
    # --- MD geometry ---
    ax, ay, az = (np.asarray(a[k], np.float64) for k in ("md_anchor_x", "md_anchor_y", "md_anchor_z"))
    r = np.hypot(ax, ay); eta = np.abs(np.arcsinh(az / r)); phi = np.arctan2(ay, ax); z = np.abs(az)
    mdlayer = np.asarray(a.md_layer)
    l0, l1 = np.asarray(a.ls_mdIdx0), np.asarray(a.ls_mdIdx1)
    t0, t1 = np.asarray(a.t3_lsIdx0), np.asarray(a.t3_lsIdx1)
    T3md = np.stack([l0[t0], l0[t1], l1[t1]], 1)
    t3rad = np.asarray(a.t3_radius, np.float64)
    i0, i1 = np.asarray(a.t5_t3Idx0, np.int64), np.asarray(a.t5_t3Idx1, np.int64)
    nT3 = len(t3rad)
    t5key = {k: j for j, k in enumerate((i0 * nT3 + i1).tolist())}
    fl = set(np.unique(mdlayer[T3md[i0, 0]]).tolist())
    first_layers = fl if first_layers is None else first_layers | fl
    md_sims = a.md_simIdxAll; md_fr = a.md_simIdxAllFrac
    t5bits = np.asarray(a.t5_isDupBits); t5tight = np.asarray(a.t5_tightCutFlag)
    t5pt5 = np.asarray(a.t5_partOfPT5); t5tc = np.asarray(a.t5_partOfTC)

    sel = np.nonzero(base)[0]
    for s in sel:
        region = 0 if (injet[s] and dr[s] >= 0 and dr[s] < 0.02) else (1 if (injet[s] and dr[s] >= 0.1) else (2 if injet[s] else 3))
        t3s = np.asarray(a.sim_t3IdxAll[s], np.int64); t3f = np.asarray(a.sim_t3IdxAllFrac[s], float)
        t3s = t3s[t3f >= 0.5]
        t5s = np.asarray(a.sim_t5IdxAll[s], np.int64); t5f = np.asarray(a.sim_t5IdxAllFrac[s], float)
        t5m = t5s[t5f > 0.75]
        lss = np.asarray(a.sim_lsIdxAll[s]); lsf = np.asarray(a.sim_lsIdxAllFrac[s])
        n_good_ls = int(np.sum(lsf > 0.75))
        n_good_t3 = int(np.sum(t3f > 0.75))
        has_pt5 = bool(np.any(np.asarray(a.sim_pt5IdxAllFrac[s]) > 0.75))
        has_pt3 = bool(np.any(np.asarray(a.sim_pt3IdxAllFrac[s]) > 0.75))
        has_pls = bool(np.any(np.asarray(a.sim_plsIdxAllFrac[s]) > 0.75))
        has_t4 = bool(np.any(np.asarray(a.sim_t4IdxAllFrac[s]) > 0.75))
        # candidate pairs
        n_cand = n_cand_pass = n_cand_exist = n_cand_pass_noexist = 0
        best_score_margin = -9.0
        if len(t3s) >= 2:
            inner = t3s[np.isin(mdlayer[T3md[t3s, 0]], list(fl))]
            if len(inner):
                # outer T3 first MD == inner third MD
                om = {}
                for j in t3s:
                    om.setdefault(T3md[j, 0], []).append(j)
                P = [(ii, jj) for ii in inner for jj in om.get(T3md[ii, 2], [])]
                if P:
                    P = np.array(P, np.int64)
                    m = np.concatenate([T3md[P[:, 0]], T3md[P[:, 1], 1:]], 1)
                    # MD-level sim fraction (2 hits per MD)
                    frac = np.zeros(len(P))
                    for k in range(5):
                        for q, mdi in enumerate(m[:, k]):
                            ss = np.asarray(md_sims[mdi]); ff = np.asarray(md_fr[mdi])
                            frac[q] += ff[ss == s].sum() if np.any(ss == s) else 0.0
                    frac /= 5
                    good = frac > 0.75
                    if good.any():
                        P = P[good]; m = m[good]
                        rin, rout = t3rad[P[:, 0]], t3rad[P[:, 1]]
                        rbr = d.radius3(ax[m[:, 1]], ay[m[:, 1]], ax[m[:, 2]], ay[m[:, 2]], ax[m[:, 3]], ay[m[:, 3]])
                        feat = d.t5_features(eta[m], phi[m], z[m], r[m], rin, rbr, rout)
                        sc = d.t5_score(feat)
                        e1 = eta[m[:, 0]]; ptb = (rin * d.K2RINV * 2 > 5).astype(int)
                        wp = d.T5_WP[ptb, d.etabin(e1)]
                        ps = sc > wp
                        ex = np.array([(p0 * nT3 + p1) in t5key for p0, p1 in P])
                        n_cand = len(P); n_cand_pass = int(ps.sum()); n_cand_exist = int(ex.sum())
                        n_cand_pass_noexist = int((ps & ~ex).sum())
                        best_score_margin = float(np.max(sc - wp))
                        for q in range(len(P)):
                            pairs.append((ie, s, region, sim_pt[s], float(a.sim_eta[s]), sc[q], wp[q], ps[q], ex[q], rin[q] * d.K2RINV * 2, frac[q]))
                            pfeat.append(feat[q])
        # downstream fate of existing matched T5s
        t5_info = 0
        if len(t5m):
            t5_info = int(np.any(t5tc[t5m])) * 1 + int(np.any(t5bits[t5m] == 0)) * 2 + int(np.any(t5tight[t5m])) * 4 + int(np.any(t5pt5[t5m])) * 8
        rows.append((ie, s, region, sim_pt[s], float(a.sim_eta[s]), dr[s], int(matched[s]), len(t5m), n_good_ls, n_good_t3, len(t3s),
                     n_cand, n_cand_pass, n_cand_exist, n_cand_pass_noexist, best_score_margin, has_pt5, has_pt3, has_pls, has_t4, t5_info,
                     float(a.sim_tcIdxBestFrac[s]) if a.sim_tcIdxBestFrac[s] is not None else -1))
    print(ie, len(rows), len(pairs), file=sys.stderr, flush=True)

cols = "ev sim region pt eta dr matched n_t5m n_good_ls n_good_t3 n_t3half n_cand n_cand_pass n_cand_exist n_cand_pass_noexist best_margin has_pt5 has_pt3 has_pls has_t4 t5_info bestfrac".split()
R = np.array(rows, dtype=np.float64)
Pp = np.array(pairs, dtype=np.float64)
np.savez("t5_replay.npz", rows=R, cols=np.array(cols), pairs=Pp,
         pcols=np.array("ev sim region simpt simeta score wp pass exists t5pt frac".split()), pfeat=np.array(pfeat), first_layers=np.array(sorted(first_layers)))
print("first layers", sorted(first_layers))
