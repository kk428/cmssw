"""Offline numpy re-implementation of the S44 LST T3 / T5 / T4 classifier DNNs.
Weights are parsed from the S44 worktree headers (read-only)."""
import re
import numpy as np

SRC = "/mnt/data1/kk829/s44_worktree/RecoTracker/LSTCore/src/alpaka/"
K2RINV = (2.99792458e-3 * 3.8) / 2
PHI_NORM = np.pi
ETA_SIZE = 0.25


def parse_weights(fname):
    txt = open(SRC + fname).read()
    out = {}
    for m in re.finditer(r"float\s+(\w+)((?:\[\d+\])+)\s*=\s*\{(.*?)\};", txt, re.S):
        name, dims, body = m.group(1), m.group(2), m.group(3)
        shape = tuple(int(d) for d in re.findall(r"\[(\d+)\]", dims))
        vals = np.array([float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)], dtype=np.float64)
        out[name] = vals.reshape(shape)
    return out


T5W = {"original": parse_weights("T5NeuralNetworkWeights_original.h"),
       "retrained": parse_weights("T5NeuralNetworkWeights_retrained.h")}
T3W = parse_weights("T3NeuralNetworkWeights.h")

T5_WP = np.array([[0.4493, 0.4939, 0.5715, 0.6488, 0.5709, 0.5938, 0.7164, 0.7565, 0.8103, 0.8593],
                  [0.4488, 0.4448, 0.5067, 0.5929, 0.4836, 0.4112, 0.4968, 0.4403, 0.5597, 0.5067]])
T3_WP_PROMPT = np.array([[0.4957, 0.5052, 0.5201, 0.5340, 0.4275, 0.4708, 0.4890, 0.4932, 0.5400, 0.5449],
                         [0.0302, 0.0415, 0.0994, 0.1791, 0.1960, 0.2467, 0.3227, 0.3242, 0.2367, 0.2187]])
T3_WP_DISP = np.array([[0.0334, 0.0504, 0.0748, 0.0994, 0.1128, 0.1123, 0.1118, 0.1525, 0.1867, 0.1847],
                       [0.0091, 0.0075, 0.0350, 0.0213, 0.0435, 0.0676, 0.1957, 0.1649, 0.1080, 0.1046]])
T5_ETA_NORM, T5_ZMAX, T5_RMAX = 2.5, 267.2349854, 110.1099396
T3_ETA_NORM, T3_ZMAX, T3_RMAX = 2.5, 224.149505, 98.932365


def dphi(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def mlp(x, W, names, final):
    h = x
    for i, (w, b) in enumerate(names):
        h = h @ W[w] + W[b]
        if i < len(names) - 1:
            h = np.maximum(h, 0)
    if final == "sigmoid":
        return 1.0 / (1.0 + np.exp(-h[:, 0]))
    e = np.exp(h - h.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def etabin(eta1):
    b = np.floor(eta1 / ETA_SIZE).astype(int)
    return np.where(eta1 > 2.5, 9, np.clip(b, 0, 9))


def radius3(x1, y1, x2, y2, x3, y3):
    """circle radius through 3 points (same as computeRadiusFromThreeAnchorHits up to numerics)."""
    a = np.hypot(x2 - x1, y2 - y1)
    b = np.hypot(x3 - x2, y3 - y2)
    c = np.hypot(x1 - x3, y1 - y3)
    cross = np.abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return a * b * c / (2 * cross)


def t5_features(eta, phi, z, r, rin, rbr, rout):
    """eta,phi,z,r: arrays shape (N,5) for anchor hits (eta, z already abs). Returns (N,23)."""
    f = [eta[:, 0] / T5_ETA_NORM, np.abs(phi[:, 0]) / PHI_NORM, z[:, 0] / T5_ZMAX, r[:, 0] / T5_RMAX]
    for k in range(1, 5):
        f += [eta[:, k] - eta[:, k - 1], dphi(phi[:, k], phi[:, k - 1]) / PHI_NORM,
              (z[:, k] - z[:, k - 1]) / T5_ZMAX, (r[:, k] - r[:, k - 1]) / T5_RMAX]
    f += [np.log10(rin), np.log10(rbr), np.log10(rout)]
    return np.stack(f, axis=1)


T5_FEATURE_NAMES = ["eta1", "|phi1|", "z1", "r1"] + [f"{v}{k+1}{k}" for k in range(1, 5) for v in ("deta", "dphi", "dz", "dr")] + \
    ["log10Rin", "log10Rbridge", "log10Rout"]


def t5_score(feat, which="original"):
    W = T5W[which]
    return mlp(feat, W, [("wgtT_layer1", "bias_layer1"), ("wgtT_layer2", "bias_layer2"),
                         ("wgtT_output_layer", "bias_output_layer")], "sigmoid")


def t5_pass(score, eta1, rin):
    pt = rin * K2RINV * 2
    return score > T5_WP[(pt > 5).astype(int), etabin(eta1)]


def t3_features(eta, phi, z, r, radius, betaIn):
    f = [eta[:, 0] / T3_ETA_NORM, np.abs(phi[:, 0]) / PHI_NORM, z[:, 0] / T3_ZMAX, r[:, 0] / T3_RMAX]
    for k in range(1, 3):
        f += [eta[:, k] - eta[:, k - 1], dphi(phi[:, k], phi[:, k - 1]) / PHI_NORM,
              (z[:, k] - z[:, k - 1]) / T3_ZMAX, (r[:, k] - r[:, k - 1]) / T3_RMAX]
    f += [np.log10(radius), betaIn]
    return np.stack(f, axis=1)


def t3_scores(feat):
    return mlp(feat, T3W, [("wgtT_layer1", "bias_layer1"), ("wgtT_layer2", "bias_layer2"),
                           ("wgtT_output_layer", "bias_output_layer")], "softmax")


def t3_pass(sc, eta1, radius):
    pt = radius * K2RINV * 2
    pi = (pt > 5).astype(int)
    b = etabin(eta1)
    return (sc[:, 1] > T3_WP_PROMPT[pi, b]) | (sc[:, 2] > T3_WP_DISP[pi, b])
