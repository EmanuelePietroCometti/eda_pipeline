"""
domain_shift.py  (analisi 2)
============================

Le immagini nominali del test vengono dalla stessa popolazione di quelle del train?

Per ogni confronto (gruppo di riferimento contro gruppo confrontato):

* **Cohen's d** con intervallo bootstrap per media e deviazione standard di RGB e
  luminosità: quanto si sposta il gruppo, in unità di deviazione standard;
* **classificatore di dominio**: una regressione logistica cerca di distinguere i due
  gruppi. AUC ≈ 0.5 (dentro il nullo per permutazione) = indistinguibili.

Il classificatore è la parte da leggere con più attenzione: è il pezzo di sklearn
di questo modulo e lo spieghiamo nella guida (GUIDA_SKLEARN.md).
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import GROUP_COLORS, GROUPS, save_fig

COMPARISONS = [("train_good", "test_good"), ("train_good", "test_dust"), ("train_def", "test_def")]
METRICS = ["R mean", "G mean", "B mean", "R std", "G std", "B std", "gray mean", "gray std"]


def image_table(idx: pd.DataFrame, scan: dict) -> pd.DataFrame:
    """Statistiche per immagine (una riga per immagine, stesso ordine di `idx`)."""
    t = idx[["filename", "split", "label", "group"]].reset_index(drop=True)
    for i, c in enumerate("RGB"):
        t[f"{c} mean"], t[f"{c} std"] = scan["rgb_mean"][:, i], scan["rgb_std"][:, i]
    t["gray mean"], t["gray std"] = scan["gray_mean"], scan["gray_std"]
    return t


def cohens_d(ref: np.ndarray, cmp: np.ndarray) -> float:
    """(media(cmp) - media(ref)) / deviazione standard combinata."""
    nx, ny = len(ref), len(cmp)
    if nx < 2 or ny < 2:
        return np.nan
    sp = np.sqrt(((nx - 1) * ref.var(ddof=1) + (ny - 1) * cmp.var(ddof=1)) / (nx + ny - 2))
    return float((cmp.mean() - ref.mean()) / sp) if sp > 0 else np.nan


def effect_sizes(tab: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g_ref, g_cmp in COMPARISONS:
        ref, cmp = tab[tab["group"] == g_ref], tab[tab["group"] == g_cmp]
        if len(ref) < 2 or len(cmp) < 2:
            continue
        for m in METRICS:
            x, y = ref[m].to_numpy(), cmp[m].to_numpy()
            # bootstrap: si ricampionano le immagini con reinserimento e si ricalcola d
            boots = [cohens_d(rng.choice(x, len(x)), rng.choice(y, len(y))) for _ in range(n_boot)]
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            rows.append({"comparison": f"{g_cmp} vs {g_ref}", "metric": m, "n_ref": len(x), "n_cmp": len(y),
                         "mean_ref": x.mean(), "mean_cmp": y.mean(), "cohens_d": cohens_d(x, y),
                         "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- classificatore di dominio

def _cv_auc(X, y, groups, n_pca, n_splits, seed) -> float:
    """AUC del classificatore in validazione incrociata.

    Tre idee di sklearn in poche righe:
    * ``make_pipeline``: scaler, PCA e classificatore in un solo oggetto. Dentro la
      validazione incrociata ogni fold RIADATTA lo scaler e la PCA solo sui dati di
      addestramento del fold: niente informazione del fold di test.
    * ``StratifiedGroupKFold``: mantiene le proporzioni delle classi e tiene TUTTE le
      immagini di uno stesso gruppo (patch adiacenti) nello stesso fold.
    * ``cross_val_predict``: dà per ogni immagine il punteggio ottenuto quando era nel
      fold di test; l'AUC si calcola una volta sui punteggi di tutte le immagini.
    """
    n_comp = int(max(1, min(n_pca, X.shape[1], len(y) // 2)))
    model = make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=0),
                          LogisticRegression(class_weight="balanced", max_iter=2000))
    k = int(min(n_splits, np.bincount(y).min()))
    if k < 2:
        return np.nan
    if groups is not None:
        cv, kw = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed), {"groups": groups}
    else:
        cv, kw = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed), {}
    score = cross_val_predict(model, X, y, cv=cv, method="decision_function", **kw)
    return float(roc_auc_score(y, score))


def domain_classifier(X, y, groups, dcfg: dict, seed: int) -> dict:
    """AUC osservata e distribuzione nulla ottenuta mescolando le etichette."""
    auc = _cv_auc(X, y, groups, dcfg["pca_components"], dcfg["cv_splits"], seed)
    rng = np.random.default_rng(seed)
    null = np.array([_cv_auc(X, rng.permutation(y), groups, dcfg["pca_components"], dcfg["cv_splits"], seed)
                     for _ in range(dcfg["n_permutations"])])
    null = null[np.isfinite(null)]
    return {"auc": auc, "null_mean": null.mean(), "null_p95": np.percentile(null, 95)}


# ----------------------------------------------------------------- grafici

def plot_effect_sizes(es: pd.DataFrame, path):
    comps = list(es["comparison"].unique())
    fig, ax = plt.subplots(figsize=(6, 4.5))
    y0 = np.arange(len(METRICS))
    for j, c in enumerate(comps):
        d = es[es["comparison"] == c].set_index("metric").reindex(METRICS)
        off = (j - (len(comps) - 1) / 2) * 0.25
        ax.errorbar(d["cohens_d"], y0 + off, xerr=[d["cohens_d"] - d["ci_lo"], d["ci_hi"] - d["cohens_d"]],
                    fmt="o", ms=4, capsize=2, label=c)
    ax.axvline(0, color="grey", lw=0.8)
    for v in (-0.8, 0.8):
        ax.axvline(v, color="grey", lw=0.6, ls=":")
    ax.set_yticks(y0, METRICS)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d (IC 95% bootstrap; tratteggio = |d| 0.8)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    return save_fig(fig, path)


def plot_gray_hist(tab: pd.DataFrame, scan: dict, path):
    bins = scan["gray_hist"].shape[1]
    centers = (np.arange(bins) + 0.5) * 256 / bins
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for g in GROUPS:
        sel = np.flatnonzero((tab["group"] == g).to_numpy())
        if len(sel) == 0:
            continue
        q25, med, q75 = np.percentile(scan["gray_hist"][sel], [25, 50, 75], axis=0)
        ax.plot(centers, med, color=GROUP_COLORS[g], label=f"{g} (n={len(sel)})")
        ax.fill_between(centers, q25, q75, color=GROUP_COLORS[g], alpha=0.2, lw=0)
    ax.set_xlabel("Livello di grigio")
    ax.set_ylabel("Frazione di pixel (mediana e quartili tra le immagini)")
    ax.legend(frameon=False, fontsize=8)
    return save_fig(fig, path)


def plot_pca(tab: pd.DataFrame, X: np.ndarray, path, title: str):
    sel = np.flatnonzero(tab["group"].isin(["train_good", "train_dust", "test_good", "test_dust"]).to_numpy())
    Z = StandardScaler().fit_transform(X[sel])
    pca = PCA(n_components=2, random_state=0).fit(Z)
    P = pca.transform(Z)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for g in ("train_good", "train_dust", "test_good", "test_dust"):
        m = (tab.iloc[sel]["group"] == g).to_numpy()
        if m.any():
            ax.scatter(P[m, 0], P[m, 1], s=16, alpha=0.7, color=GROUP_COLORS[g], label=f"{g} (n={m.sum()})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return save_fig(fig, path)


def run(idx: pd.DataFrame, scan: dict, emb: np.ndarray, leak_groups, cfg: dict, out_dir):
    dcfg = cfg["domain_shift"]
    seed = cfg["general"]["seed"]
    tab = image_table(idx, scan)
    tab.to_csv(f"{out_dir}/domain_image_stats.csv", index=False)

    es = effect_sizes(tab, dcfg["n_bootstrap"], seed)
    es.to_csv(f"{out_dir}/domain_effect_sizes.csv", index=False)
    if len(es):
        plot_effect_sizes(es, f"{out_dir}/figures/domain_effect_sizes.png")
    plot_gray_hist(tab, scan, f"{out_dir}/figures/domain_gray_hist.png")
    plot_pca(tab, emb, f"{out_dir}/figures/domain_pca.png", "PCA degli embedding (immagini nominali)")

    color_feats = np.hstack([scan["rgb_mean"], scan["rgb_std"], scan["gray_hist"]])
    feature_sets = {"colore + istogramma": color_feats, "embedding": emb}
    rows = []
    for g_ref, g_cmp in COMPARISONS:
        sel = tab["group"].isin([g_ref, g_cmp]).to_numpy()
        y = (tab.loc[sel, "group"] == g_cmp).astype(int).to_numpy()
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        for name, X in feature_sets.items():
            row = {"comparison": f"{g_cmp} vs {g_ref}", "features": name, "n_ref": n0, "n_cmp": n1}
            if min(n0, n1) < dcfg["min_n"]:
                row["nota"] = f"non eseguito: meno di {dcfg['min_n']} immagini in un gruppo"
            else:
                g = np.asarray(leak_groups)[sel] if leak_groups is not None else None
                row.update(domain_classifier(X[sel], y, g, dcfg, seed))
            rows.append(row)
    clf = pd.DataFrame(rows)
    clf.to_csv(f"{out_dir}/domain_classifier.csv", index=False)
    print(clf.round(3).to_string(index=False))
    return {"effect_sizes": es, "classifier": clf}
