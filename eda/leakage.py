"""
leakage.py  (analisi 3)
=======================

Train e test condividono pixel? Le patch hanno 32 px di sovrapposizione con la
vicina dello stesso rotolo: l'ultima striscia di una patch è la prima della
successiva. Se lo split è per patch e non per rotolo, i vicini finiscono su lati
opposti dello split e le metriche sul test sono ottimistiche.

Due controlli:

1. **Bordi esatti**: hash dei pixel grigi della striscia. Completo (nessuna coppia
   identica sfugge) e veloce (un dizionario, nessun confronto a coppie).
2. **Hash percettivo** (pHash, Hamming <= soglia) confermato dalla correlazione di
   miniature 64×64: trova quasi-duplicati che non condividono pixel esatti.

Le coppie formano gruppi (componenti connesse); un gruppo a cavallo di train e test
è prova diretta di split non per rotolo.
"""

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .utils import save_fig


def exact_strip_pairs(scan: dict, min_std: float) -> pd.DataFrame:
    """Coppie (a, b) dove `a` è la patch sinistra/superiore di `b` e la striscia coincide."""
    rows = []
    for direction, ref, qry in (("horizontal", "right", "left"), ("vertical", "bottom", "top")):
        by_hash = {}
        for a, h in enumerate(scan["strip_hash"][ref]):
            by_hash.setdefault(int(h), []).append(a)
        for b, h in enumerate(scan["strip_hash"][qry]):
            for a in by_hash.get(int(h), []):
                if a != b:
                    std = float(scan["strip_std"][qry][b])
                    # due strisce uniformi (es. sfondo nero) coincidono senza essere vicine
                    rows.append({"a": a, "b": b, "kind": "border", "direction": direction,
                                 "informative": std >= min_std})
    return pd.DataFrame(rows, columns=["a", "b", "kind", "direction", "informative"])


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(np.ascontiguousarray(x).view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def phash_pairs(scan: dict, paths, max_hamming: int, min_corr: float, chunk: int = 256) -> pd.DataFrame:
    ph = scan["phash"]
    cand = []
    for s in range(0, len(ph), chunk):
        h = _popcount(ph[s:s + chunk][:, None] ^ ph[None, :]).astype(np.int32)
        ii, jj = np.nonzero(h <= max_hamming)
        keep = (ii + s) < jj                                   # ogni coppia una volta sola
        cand += list(zip(ii[keep] + s, jj[keep]))
    thumbs = {}

    def thumb(i):
        if i not in thumbs:
            g = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
            t = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            thumbs[i] = (t - t.mean()) / (t.std() + 1e-6)
        return thumbs[i]

    rows = []
    for a, b in cand[:50_000]:
        if float(thumb(a) @ thumb(b)) / (64 * 64) >= min_corr:
            rows.append({"a": int(a), "b": int(b), "kind": "phash", "direction": "", "informative": True})
    return pd.DataFrame(rows, columns=["a", "b", "kind", "direction", "informative"])


def connected_groups(n: int, pairs: pd.DataFrame) -> np.ndarray:
    """Componente connessa di ogni immagine (le immagini isolate formano un gruppo da una)."""
    if pairs.empty:
        return np.arange(n)
    adj = coo_matrix((np.ones(len(pairs)), (pairs["a"], pairs["b"])), shape=(n, n))
    return connected_components(adj, directed=False)[1]


def pair_statistics(idx: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Quota di coppie a cavallo dello split, osservata e attesa per uno split per patch casuale."""
    share = idx["split"].value_counts(normalize=True)
    expected = 1.0 - float((share ** 2).sum())
    rows = []
    for kind, p in pairs.groupby("kind"):
        cross = (idx["split"].to_numpy()[p["a"]] != idx["split"].to_numpy()[p["b"]])
        rows.append({"kind": kind, "n_pairs": len(p), "n_cross_split": int(cross.sum()),
                     "frac_cross_observed": cross.mean(), "frac_cross_if_random_split": expected})
    return pd.DataFrame(rows, columns=["kind", "n_pairs", "n_cross_split", "frac_cross_observed",
                                       "frac_cross_if_random_split"])


def pairs_by_group(idx: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Conta le coppie a cavallo dello split per combinazione di gruppi (le più gravi: difetto–difetto)."""
    if pairs.empty:
        return pd.DataFrame(columns=["group_a", "group_b", "n_pairs"])
    ga, gb = idx["group"].to_numpy()[pairs["a"]], idx["group"].to_numpy()[pairs["b"]]
    sa, sb = idx["split"].to_numpy()[pairs["a"]], idx["split"].to_numpy()[pairs["b"]]
    t = pd.DataFrame({"group_a": np.minimum(ga, gb), "group_b": np.maximum(ga, gb), "cross": sa != sb})
    return (t[t["cross"]].groupby(["group_a", "group_b"]).size().rename("n_pairs_cross_split").reset_index())


def plot_pair_fractions(stats: pd.DataFrame, path):
    long = stats.melt(id_vars="kind", value_vars=["frac_cross_observed", "frac_cross_if_random_split"],
                      var_name="quota", value_name="frazione")
    long["quota"] = long["quota"].map({"frac_cross_observed": "osservata",
                                       "frac_cross_if_random_split": "attesa se split per patch"})
    fig, ax = plt.subplots(figsize=(5, 3.4))
    sns.barplot(data=long, x="kind", y="frazione", hue="quota", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Coppie a cavallo di train/test")
    return save_fig(fig, path)


def run(idx: pd.DataFrame, scan: dict, cfg: dict, out_dir):
    """Ritorna una tabella per immagine con il gruppo e i flag di leakage."""
    lc = cfg["leakage"]
    paths = idx["filename"].tolist()
    border = exact_strip_pairs(scan, lc["min_strip_std"])
    ph = phash_pairs(scan, paths, lc["phash_max_hamming"], lc["phash_min_corr"])
    pairs = pd.concat([border, ph], ignore_index=True)
    pairs = pairs[pairs["informative"].astype(bool)].astype({"a": int, "b": int}).reset_index(drop=True)

    out = idx[["filename", "split", "label", "group"]].copy()
    out["group_id"] = connected_groups(len(idx), pairs)
    out["group_spans_splits"] = out.groupby("group_id")["split"].transform("nunique") > 1
    cross = pairs[idx["split"].to_numpy()[pairs["a"]] != idx["split"].to_numpy()[pairs["b"]]]
    flag = np.zeros(len(idx), bool)
    flag[np.concatenate([cross["a"].to_numpy(), cross["b"].to_numpy()])] = True
    out["neighbor_in_other_split"] = flag
    out.to_csv(f"{out_dir}/leakage_images.csv", index=False)

    stats = pair_statistics(idx, pairs)
    stats.to_csv(f"{out_dir}/leakage_pair_stats.csv", index=False)
    by_group = pairs_by_group(idx, pairs)
    by_group.to_csv(f"{out_dir}/leakage_pairs_by_group.csv", index=False)
    if not stats.empty:
        plot_pair_fractions(stats, f"{out_dir}/figures/leakage_pairs.png")
    print(stats.round(3).to_string(index=False))
    print(by_group.to_string(index=False) if len(by_group) else "nessuna coppia a cavallo dello split")
    return out
