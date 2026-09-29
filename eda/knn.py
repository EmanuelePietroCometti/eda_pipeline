"""
knn.py  (analisi 4)
===================

Baseline di distanza senza addestramento: il punteggio di anomalia di un'immagine è
la distanza media dai suoi k vicini più prossimi nel riferimento (`train/good`).

Lettura onesta dei numeri: NON è PatchCore. PatchCore tiene una memoria di feature di
*patch* e valuta ogni posizione; qui c'è un solo vettore globale per immagine, in cui
un difetto piccolo pesa poco. È un limite inferiore di ciò che un metodo a patch
può ottenere, e sui micro-difetti fallisce per costruzione.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .utils import GROUP_COLORS, GROUPS, save_fig


def knn_scores(X: np.ndarray, ref_mask: np.ndarray, k: int) -> np.ndarray:
    """Distanza media dai k vicini più prossimi nel riferimento, per TUTTE le immagini.

    Le immagini del riferimento sono punteggiate in leave-one-out: il vicino più
    prossimo di un'immagine del riferimento sarebbe sé stessa (distanza 0), quindi
    la si scarta.
    """
    ref = np.flatnonzero(ref_mask)
    # lo scaler si adatta SOLO al riferimento: il test non deve influenzare la scala
    Z = StandardScaler().fit(X[ref]).transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z[ref])
    dist, ind = nn.kneighbors(Z)                       # k+1 vicini per ogni immagine
    pos = np.full(len(X), -1)
    pos[ref] = np.arange(len(ref))                     # posizione di ogni immagine dentro il riferimento
    scores = np.empty(len(X))
    for i in range(len(X)):
        d = dist[i]
        if pos[i] >= 0:                                # membro del riferimento: togli sé stesso
            hit = np.flatnonzero(ind[i] == pos[i])
            d = np.delete(d, hit[0] if len(hit) else len(d) - 1)
        scores[i] = d[:k].mean()
    return scores


def auroc_ci(y, s, n_boot, rng):
    """AUROC con intervallo bootstrap stratificato (si ricampionano positivi e negativi separatamente)."""
    auc = roc_auc_score(y, s)
    pos, neg = s[y == 1], s[y == 0]
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    boots = [roc_auc_score(lab, np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
             for _ in range(n_boot)]
    return auc, *np.percentile(boots, [2.5, 97.5])


def comparisons(group: np.ndarray, label: np.ndarray) -> list:
    """(nome, maschera positivi, maschera negativi). Punteggio alto = positivo."""
    out = [
        ("difetti vs test_good", group == "test_def", group == "test_good"),
        ("difetti vs test_good+test_dust", group == "test_def", np.isin(group, ["test_good", "test_dust"])),
        ("test_dust vs test_good", group == "test_dust", group == "test_good"),
    ]
    for c in sorted(set(label[group == "test_def"])):
        out.append((f"{c} vs test_good", (group == "test_def") & (label == c), group == "test_good"))
    return out


def evaluate(scores, group, label, n_boot, rng, min_class=2, keep=None) -> pd.DataFrame:
    """AUROC/AP per ogni confronto. `keep` (booleano) restringe le immagini considerate."""
    rows = []
    keep = np.ones(len(scores), bool) if keep is None else keep
    for name, pos, neg in comparisons(group, label):
        pos, neg = pos & keep, neg & keep
        sel = pos | neg
        y = pos[sel].astype(int)
        if y.sum() < min_class or (1 - y).sum() < min_class:
            continue
        auc, lo, hi = auroc_ci(y, scores[sel], n_boot, rng)
        rows.append({"comparison": name, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                     "auroc": auc, "auroc_ci_lo": lo, "auroc_ci_hi": hi,
                     "ap": average_precision_score(y, scores[sel]), "ap_chance": y.mean()})
    return pd.DataFrame(rows)


def plot_scores(idx: pd.DataFrame, scores, path, title):
    d = pd.DataFrame({"score": scores, "group": idx["group"].to_numpy()})
    order = [g for g in GROUPS if g in set(d["group"])]
    fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(order) + 1.8))
    sns.boxplot(data=d, x="score", y="group", order=order, color="0.92", fliersize=0, ax=ax)
    sns.stripplot(data=d, x="score", y="group", order=order, size=3, alpha=0.6, ax=ax,
                  palette=[GROUP_COLORS[g] for g in order], hue="group", legend=False)
    ax.set_xlabel("Distanza dal riferimento (più alta = più anomala)")
    ax.set_ylabel("")
    ax.set_title(title)
    return save_fig(fig, path)


def plot_roc(scores, group, label, path, title):
    fig, ax = plt.subplots(figsize=(5, 4.8))
    for name, pos, neg in comparisons(group, label):
        sel = pos | neg
        y = pos[sel].astype(int)
        if y.sum() < 2 or (1 - y).sum() < 2 or name == "test_dust vs test_good":
            continue
        fpr, tpr, _ = roc_curve(y, scores[sel])
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, scores[sel]):.2f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Tasso di falsi positivi")
    ax.set_ylabel("Tasso di veri positivi")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    return save_fig(fig, path)


def run(idx: pd.DataFrame, emb: np.ndarray, cfg: dict, out_dir):
    kc = cfg["knn"]
    rng = np.random.default_rng(cfg["general"]["seed"])
    group, label = idx["group"].to_numpy(), idx["label"].to_numpy()
    ref_mask = np.isin(group, ["train_good", "train_dust"])       # i difetti del train NON entrano mai
    ks = sorted(set(kc["k_values"]) | {kc["primary_k"]})
    scores_df = idx[["filename", "split", "label", "group"]].copy()
    metrics = []
    for k in ks:
        s = knn_scores(emb, ref_mask, k)
        scores_df[f"score_k{k}"] = s
        m = evaluate(s, group, label, kc["n_bootstrap"], rng)
        metrics.append(m.assign(k=k, n_reference=int(ref_mask.sum()), subset="tutte"))
        # sensibilità: togli dal test le immagini con pixel identici a una del riferimento
        # (distanza 0 per costruzione: abbassano i punteggi dei negativi e gonfiano l'AUROC)
        dup_test = (~ref_mask) & idx["sha"].isin(set(idx.loc[ref_mask, "sha"])).to_numpy()
        if dup_test.any():
            m2 = evaluate(s, group, label, kc["n_bootstrap"], rng, keep=~dup_test)
            metrics.append(m2.assign(k=k, n_reference=int(ref_mask.sum()), subset="senza duplicati del riferimento"))
    metrics = pd.concat(metrics, ignore_index=True)
    scores_df.to_csv(f"{out_dir}/knn_scores.csv", index=False)
    metrics.to_csv(f"{out_dir}/knn_metrics.csv", index=False)
    main = scores_df[f"score_k{kc['primary_k']}"].to_numpy()
    plot_scores(idx, main, f"{out_dir}/figures/knn_scores.png", f"kNN (k={kc['primary_k']}, {kc['backbone']})")
    plot_roc(main, group, label, f"{out_dir}/figures/knn_roc.png", f"ROC (k={kc['primary_k']})")
    if (~ref_mask & idx["sha"].isin(set(idx.loc[ref_mask, "sha"])).to_numpy()).any():
        print(f"immagini di test con pixel identici a una del riferimento: "
              f"{int((~ref_mask & idx['sha'].isin(set(idx.loc[ref_mask, 'sha'])).to_numpy()).sum())} "
              f"(righe 'senza duplicati del riferimento' nel CSV)")
    print(metrics[(metrics["k"] == kc["primary_k"]) & (metrics["subset"] == "tutte")]
          .drop(columns=["k", "subset"]).round(3).to_string(index=False))
    return metrics


def plot_contrasts(tab: pd.DataFrame, path):
    """AUROC (con IC bootstrap) per confronto e versione: dove si vede l'effetto della polvere nel train."""
    tab = tab[tab["subset"] == "tutte"]
    order = list(dict.fromkeys(tab["comparison"]))
    versions = list(dict.fromkeys(tab["version"]))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(order) + 1.8))
    h = 0.8 / len(versions)
    for j, v in enumerate(versions):
        t = tab[tab["version"] == v].set_index("comparison").reindex(order)
        y = np.arange(len(order)) + (j - (len(versions) - 1) / 2) * h
        ax.errorbar(t["auroc"], y, xerr=[t["auroc"] - t["auroc_ci_lo"], t["auroc_ci_hi"] - t["auroc"]],
                    fmt="o", ms=4, capsize=2, label=v)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.axvline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("AUROC (IC 95% bootstrap)")
    ax.legend(title="versione", frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return save_fig(fig, path)
