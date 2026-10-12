"""Shared loader: rebuild hit-level content of MDs/T3s/T5s/pLSs/pT5s from the S44 --allobj ntuple.
Hits are identified by their float32 (x,y,z) (anchor/other of MDs, pLS_hit*). Pixel and OT hits get disjoint ids."""
import numpy as np, uproot, awkward as ak
F100 = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s44_base_100evt.root"
BR = ["md_anchor_x","md_anchor_y","md_anchor_z","md_other_x","md_other_y","md_other_z","md_detId","md_layer",
      "ls_mdIdx0","ls_mdIdx1","t3_lsIdx0","t3_lsIdx1","t5_t3Idx0","t5_t3Idx1",
      "t5_eta","t5_phi","t5_pt","t5_score","t5_isDupBits","t5_partOfPT5","t5_tightCutFlag","t5_dnnScore","t5_embed",
      "t5_simIdx","t5_simIdxAll","t5_simIdxAllFrac","t5_isFake",
      "pLS_hit0_x","pLS_hit0_y","pLS_hit0_z","pLS_hit1_x","pLS_hit1_y","pLS_hit1_z",
      "pLS_hit2_x","pLS_hit2_y","pLS_hit2_z","pLS_hit3_x","pLS_hit3_y","pLS_hit3_z",
      "pLS_simIdx","pLS_simIdxAll","pLS_isQuad","pLS_isDup","pLS_eta","pLS_phi","pLS_pt",
      "pT5_t5Idx","pT5_plsIdx","pT5_isDupReco","pT5_score","pT5_simIdx","pT5_simIdxAll",
      "pT3_plsIdx","pT3_t3Idx","pT3_eta","pT3_phi","pT3_simIdx",
      "tc_type","tc_t5Idx","tc_pt5Idx","tc_pt3Idx","tc_plsIdx","tc_simIdx","tc_isFake",
      "sim_pt","sim_eta","sim_phi","sim_t5IdxAll","sim_t5IdxAllFrac","sim_plsIdxAll","sim_plsIdxAllFrac","sim_pt5IdxAll","sim_pt5IdxAllFrac","sim_lsIdxAll","sim_t3IdxAll","sim_mdIdxAll","sim_mdIdxAllFrac","sim_tcIdx"]

def _keys(x, y, z):
    a = np.stack([np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(z, np.float32)], 1)
    return a

def build(d):
    """d: one-event awkward record. Returns dict of numpy arrays."""
    o = {}
    A = _keys(d.md_anchor_x, d.md_anchor_y, d.md_anchor_z); B = _keys(d.md_other_x, d.md_other_y, d.md_other_z)
    P = [_keys(d[f"pLS_hit{k}_x"], d[f"pLS_hit{k}_y"], d[f"pLS_hit{k}_z"]) for k in range(4)]
    nmd = len(A); npls = len(P[0])
    allh = np.concatenate([A, B] + P, 0)
    tag = np.concatenate([np.zeros(2*nmd, np.int8)] + [np.ones(npls, np.int8)]*4)  # 0=OT/md hit, 1=pixel
    full = np.concatenate([allh.view(np.uint32), tag[:, None].astype(np.uint32)], 1)
    _, inv = np.unique(full, axis=0, return_inverse=True)
    inv = inv.ravel()
    mdh = np.stack([inv[:nmd], inv[nmd:2*nmd]], 1)
    plsh = np.stack([inv[2*nmd + k*npls: 2*nmd + (k+1)*npls] for k in range(4)], 1)
    ls0 = np.asarray(d.ls_mdIdx0); ls1 = np.asarray(d.ls_mdIdx1)
    t3l0 = np.asarray(d.t3_lsIdx0); t3l1 = np.asarray(d.t3_lsIdx1)
    t3md = np.stack([ls0[t3l0], ls1[t3l0], ls1[t3l1]], 1)
    t5a = np.asarray(d.t5_t3Idx0); t5b = np.asarray(d.t5_t3Idx1)
    t5md = np.concatenate([t3md[t5a], t3md[t5b][:, 1:]], 1)  # 5 MDs (outer T3 shares its first MD)
    t5h = mdh[t5md].reshape(len(t5md), 10)
    o.update(mdh=mdh, plsh=plsh, t3md=t3md, t5md=t5md, t5h=t5h,
             t5_lowmod=np.asarray(d.md_detId)[t5md[:, 0]], nmd=nmd)
    return o

def nshared(h1, h2):
    """rows of hit ids (N,k) vs (N,k) -> number of h1 entries present in h2 (like checkHitsT5)."""
    return (h1[:, :, None] == h2[:, None, :]).any(2).sum(1)
