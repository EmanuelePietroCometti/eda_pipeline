"""
projections.py
==============

Proiezioni con PCA, t-SNE e UMAP, una figura per domanda con i tre metodi affiancati:

* ``proj_polvere_vs_buoni``: `train_good`, `test_good`, `train_dust`, `test_dust` insieme (2D);
* ``proj_buoni_vs_difetti`` e ``proj_buoni_vs_difetti_3d``: buoni e difetti, senza polvere
  (sono separabili?), in 2D e in 3D.

Ogni proiezione si adatta ai soli gruppi della sua figura. Gli embedding sono già standardizzati
sui `train_good`. Le proiezioni sono grafici descrittivi, non misure: t-SNE e UMAP non conservano
le distanze globali, e in 2D/3D la PCA mostra solo una parte della varianza (riportata sugli assi).
In 3D una sola prospettiva può nascondere una separazione o crearne una apparente: va letta
insieme alla figura 2D.
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .utils import GROUP_COLORS, save_fig

NAMES = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}

# nome -> (titolo, gruppi usati, suffisso di gruppo che deve esserci, dimensioni)
FIGURES = {
    "polvere_vs_buoni": ("Buoni e polvere, train e test",
                         ("train_good", "test_good", "train_dust", "test_dust"), "_dust", (2,)),
    "buoni_vs_difetti": ("Buoni vs difetti (senza polvere)",
                         ("train_good", "test_good", "train_def", "test_def"), "_def", (2, 3)),
}

# gruppo -> marcatore, dimensione, trasparenza, bordo, ordine di sovrapposizione
STYLE = {
    "train_good": ("o", 12, 0.5, "none", 1),
    "test_good": ("o", 12, 0.5, "none", 2),
    "train_dust": ("D", 40, 0.95, "k", 4),
    "test_dust": ("s", 40, 0.95, "k", 4),
    "train_def": ("^", 28, 0.85, "k", 3),
    "test_def": ("v", 28, 0.85, "k", 3),
}


def project(Z, method, n_comp, pcfg, seed):
    """Proietta Z (già standardizzata) in n_comp dimensioni. Ritorna (coordinate, varianza spiegata o None)."""
    n = len(Z)
    if method == "pca":
        m = PCA(n_components=n_comp, random_state=seed).fit(Z)
        return m.transform(Z), m.explained_variance_ratio_
    if method == "tsne":
        X = Z
        if Z.shape[1] > 50:                                    # PCA a 50 componenti prima di t-SNE: più veloce, meno rumore
            X = PCA(n_components=min(50, n - 1), random_state=seed).fit_transform(Z)
        perplexity = min(pcfg["tsne"]["perplexity"], max(2.0, (n - 1) / 3))
        tsne = TSNE(n_components=n_comp, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
        return tsne.fit_transform(X), None
    if method == "umap":
        import umap

        u = pcfg["umap"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs.*")      # random_state fissa il seme e disattiva il parallelismo
            m = umap.UMAP(n_components=n_comp, n_neighbors=int(min(u["n_neighbors"], n - 1)),
                          min_dist=u["min_dist"], random_state=seed)
            return m.fit_transform(Z), None
    raise ValueError(f"Metodo sconosciuto: {method} (pca | tsne | umap)")


def _subsample(sub, n_max, seed):
    """Posizioni di al più n_max immagini per (gruppo, classe), a seme fisso."""
    if not n_max:
        return np.arange(len(sub))
    rng = np.random.default_rng(seed)
    pos = [ix if len(ix) <= n_max else rng.choice(ix, n_max, replace=False)
           for ix in sub.groupby(["group", "label"]).indices.values()]
    return np.sort(np.concatenate(pos))


def plot_methods(results, groups, title, path):
    """Una riga di grafici, uno per metodo; `results` è una lista di (metodo, coordinate, varianza o None).
    Con coordinate a 3 colonne i grafici sono 3D."""
    d = results[0][1].shape[1]
    fig = plt.figure(figsize=(5.2 * len(results), 5) if d == 2 else (6.4 * len(results), 6))
    axes = []
    for j, (method, P, evr) in enumerate(results, start=1):
        ax = fig.add_subplot(1, len(results), j, **({"projection": "3d"} if d == 3 else {}))
        extra = {"depthshade": False} if d == 3 else {}      # niente sfumatura con la profondità: falsa la lettura
        for g in groups.unique():
            m = (groups == g).to_numpy()
            marker, size, alpha, edge, z = STYLE[g]
            ax.scatter(*P[m].T, c=GROUP_COLORS[g], label=f"{g} (n={int(m.sum())})", marker=marker, s=size,
                       alpha=alpha, edgecolors=edge, linewidths=0.4, zorder=z, **extra)
        if evr is None:
            labels = [f"{NAMES[method]} {i + 1}" for i in range(d)]
            ax.set_title(NAMES[method])
        else:
            labels = [f"PC{i + 1} ({v:.1%})" for i, v in enumerate(evr)]
            ax.set_title(f"PCA (varianza spiegata {evr.sum():.1%})")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        if d == 3:
            ax.set_zlabel(labels[2])
            ax.view_init(elev=20, azim=45)
        axes.append(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: list(STYLE).index(labels[i].split(" ")[0]))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=len(labels), frameon=False, markerscale=1.5)
    fig.suptitle(title)
    if d == 2:
        fig.tight_layout()
    else:                                   # tight_layout non gestisce gli assi 3D: le etichette z finirebbero sul pannello accanto
        fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.1, wspace=0.12)
    return save_fig(fig, path, pad=0.1 if d == 2 else 0.45)


def run(idx: pd.DataFrame, Z: np.ndarray, cfg: dict, out_dir, version: str):
    pcfg, seed = cfg["projection"], cfg["general"]["seed"]
    if "umap" in pcfg["methods"] and importlib.util.find_spec("umap") is None:
        raise ImportError("UMAP richiede umap-learn: pip install umap-learn (oppure togli 'umap' da projection.methods)")

    for fname, (ftitle, groups, needs, dims) in FIGURES.items():
        sel = idx["group"].isin(groups).to_numpy()
        sub, Zs = idx[sel].reset_index(drop=True), Z[sel]
        present = set(sub["group"])
        if not present & {"train_good", "test_good"} or not any(p.endswith(needs) for p in present):
            print(f"[{version}] {fname}: saltata (gruppi presenti: {sorted(present)})")
            continue
        pos = _subsample(sub, pcfg.get("max_per_group"), seed)
        sub, Zs = sub.iloc[pos].reset_index(drop=True), Zs[pos]
        if len(sub) < 10:
            print(f"[{version}] {fname}: saltata ({len(sub)} immagini sono troppo poche)")
            continue

        for d in dims:
            tag = "" if d == 2 else f"_{d}d"
            results = []
            for method in pcfg["methods"]:
                print(f"[{version}] {fname}: {NAMES[method]} {d}D su {len(sub)} immagini")
                P, evr = project(Zs, method, d, pcfg, seed)
                results.append((method, P, evr))
                csv = Path(out_dir) / "projections" / f"{fname}_{method}{tag}.csv"
                csv.parent.mkdir(parents=True, exist_ok=True)
                pd.concat([sub[["filename", "split", "label", "group"]],
                           pd.DataFrame(P, columns=[f"c{i + 1}" for i in range(d)])], axis=1).to_csv(csv, index=False)
            plot_methods(results, sub["group"], f"{ftitle}, {d}D [{version}]",
                         f"{out_dir}/figures/proj_{fname}{tag}.png")
