#!/usr/bin/env python3
"""b08: decide for each sim-matched pLS whether it survived CheckHitspLS pass 1 (pre-pT5).
isDup value 1 is ambiguous (pass-1 kill OR CrossCleanpLS at TC stage, which assigns true). A pLS used by any pT5/pT3 was
alive at pT5/pT3 building -> survived pass 1. A pLS with no >=3-slot partner within |dEta|<=0.1 cannot be a pass-1 kill.
A triplet pLS with a >=3-slot QUAD partner is a certain pass-1 kill (quads always win). Remaining = ambiguous."""
import numpy as np, uproot, pandas as pd
from bcommon import F100
x = pd.read_pickle("b07_pls.pkl")
t = uproot.open(F100)["tree"]
used = {}
for ie, d in enumerate(t.iterate(["pT5_plsIdx","pT3_plsIdx","tc_plsIdx","tc_pt5Idx","tc_pt3Idx"], step_size=1, library="np")):
    used[ie] = (set(d["pT5_plsIdx"][0].tolist()), set(d["pT3_plsIdx"][0].tolist()), set(d["tc_plsIdx"][0].tolist()))
def st(r):
    u5, u3, utc = used[r.evt]
    if r.pls in utc: return "pLS_TC"
    if r.pls in u5 or r.pls in u3: return "alive_pass1(used in pT5/pT3)"
    if (r.bits & 1) == 0: return "alive_pass1(bits)"
    if r.k3_n == 0: return "alive_pass1(no partner; later CrossCleanpLS)"
    if (not r.quad) and any(r.k3_quad): return "pass1_killed(triplet vs quad)"
    return "ambiguous(pass1 or CrossClean)"
x["state"] = x.apply(st, axis=1)
x.to_pickle("b08_pls_state.pkl")
for name, sel in [("core", x.dr < 0.02), ("noncore", x.dr >= 0.1)]:
    print("==", name); print(x[sel].state.value_counts().to_string())
