"""
eda.py
======

Descriptive analysis for the textile EDA: PCA and t-SNE only.

Given the handcrafted feature :class:`~pandas.DataFrame` produced by
:mod:`src.feature_extraction`, this module provides:

* **PCA** — standardised features, explained variance of every fitted component
  (scree plot), sample scores and loadings. The loadings tell *which* features
  drive each component; their squared values, summed per feature family
  (colour / GLCM / LBP), tell whether a component is mostly about colour or
  about texture.
* **t-SNE** — a 2D embedding for visual inspection, plus a stability check over
  several perplexities and seeds. Distances between clusters and cluster sizes
  in a t-SNE map are not interpretable; only structures that appear in every
  run of the stability grid should be described.
* **Sample counts** per fabric and class, reported next to every figure.

The analysis is purely descriptive: no hypothesis test is run here.

Only ``numpy``, ``pandas`` and ``scikit-learn`` are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Metadata columns that are never treated as features.
META_COLS = ("filename", "fabric", "label", "group")

# Feature families, identified by column-name prefix.
FEATURE_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "colour": ("hsv_", "lab_"),
    "glcm": ("glcm_",),
    "lbp": ("lbp_",),
}


# --------------------------------------------------------------------------- #
# Feature-column helpers
# --------------------------------------------------------------------------- #
def get_feature_columns(df: pd.DataFrame, prefixes: Sequence[str] | None = None) -> List[str]:
    """Return the numeric feature columns of ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature matrix including the metadata columns.
    prefixes : sequence of str, optional
        If given, keep only columns whose name starts with one of these
        prefixes (e.g. ``("glcm_", "lbp_")`` to select texture features).
    """
    cols = [
        c for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if prefixes is not None:
        cols = [c for c in cols if any(c.startswith(p) for p in prefixes)]
    return cols


def feature_family(col: str) -> str:
    """Return the family (``colour``/``glcm``/``lbp``/``other``) of a feature."""
    for family, prefixes in FEATURE_FAMILIES.items():
        if any(col.startswith(p) for p in prefixes):
            return family
    return "other"


def usable_feature_columns(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split ``cols`` into features with variance and constant features.

    A feature that is constant within the analysed subset carries no
    information and would only produce a zero loading, so it is excluded
    from that analysis (and reported).
    """
    std = df[list(cols)].std(ddof=0)
    kept = [c for c in cols if std[c] > 1e-12]
    dropped = [c for c in cols if std[c] <= 1e-12]
    return kept, dropped


def _standardised_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Z-score the features (mean 0, std 1) over the analysed subset.

    Standardisation is mandatory before PCA and t-SNE because the raw features
    live on very different scales (colour variances in the hundreds, LBP
    histogram bins in [0, 1]).
    """
    X = df[list(cols)].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        bad = [c for c in cols if not np.isfinite(df[c]).all()]
        raise ValueError(f"Non-finite values in features: {bad}")
    return StandardScaler().fit_transform(X)


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
@dataclass
class PCAResult:
    """Everything the figures and tables of a PCA need.

    Attributes
    ----------
    scores : DataFrame
        One row per sample (same index as the input), columns ``PC1..PCk``.
    explained_variance_ratio : ndarray
        Fraction of total variance explained by each fitted component.
    loadings : DataFrame
        Features x components. Each column is a unit vector: the squared
        loadings of a component sum to 1.
    feature_cols : list of str
        Features actually used (constant ones removed).
    dropped_constant : list of str
        Features removed because constant in the analysed subset.
    """

    scores: pd.DataFrame
    explained_variance_ratio: np.ndarray
    loadings: pd.DataFrame
    feature_cols: List[str]
    dropped_constant: List[str] = field(default_factory=list)

    def explained_variance_table(self) -> pd.DataFrame:
        """Per-component and cumulative explained variance, in percent."""
        evr = self.explained_variance_ratio
        return pd.DataFrame(
            {
                "component": [f"PC{i + 1}" for i in range(len(evr))],
                "explained_variance_pct": np.round(evr * 100, 2),
                "cumulative_pct": np.round(np.cumsum(evr) * 100, 2),
            }
        )


def run_pca(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    n_components: int = 10,
    random_state: int = 42,
) -> PCAResult:
    """Fit a PCA on the standardised features of ``df``.

    The number of components is capped by ``n_samples - 1`` and by the number
    of non-constant features. Component signs are arbitrary (a PCA is defined
    up to a sign flip per component): only relative positions matter.
    """
    cols, dropped = usable_feature_columns(df, feature_cols)
    X = _standardised_matrix(df, cols)

    k = int(min(n_components, X.shape[0] - 1, X.shape[1]))
    if k < 1:
        raise ValueError(f"PCA needs at least 2 samples, got {X.shape[0]}.")

    pca = PCA(n_components=k, random_state=random_state)
    scores = pca.fit_transform(X)
    pcs = [f"PC{i + 1}" for i in range(k)]

    return PCAResult(
        scores=pd.DataFrame(scores, index=df.index, columns=pcs),
        explained_variance_ratio=pca.explained_variance_ratio_,
        loadings=pd.DataFrame(pca.components_.T, index=cols, columns=pcs),
        feature_cols=cols,
        dropped_constant=dropped,
    )


def top_loadings(result: PCAResult, component: str, top_k: int = 8) -> pd.DataFrame:
    """Features with the largest absolute loading on ``component``."""
    col = result.loadings[component]
    order = col.abs().sort_values(ascending=False).index[:top_k]
    return pd.DataFrame(
        {
            "feature": order,
            "family": [feature_family(f) for f in order],
            "loading": col[order].to_numpy(),
        }
    )


def family_share(result: PCAResult, components: Sequence[str]) -> pd.DataFrame:
    """Share of each component due to each feature family.

    Because every component is a unit vector, its squared loadings sum to 1;
    summing them per family gives the fraction of the component built from
    colour, GLCM or LBP features. The number of features per family is
    reported too: a family with many features (the 26 LBP bins) can obtain a
    large share partly because of its size, so the share per feature is given
    as well.
    """
    families = pd.Series({f: feature_family(f) for f in result.feature_cols})
    n_per_family = families.value_counts()
    rows = []
    for comp in components:
        if comp not in result.loadings.columns:
            continue
        sq = result.loadings[comp] ** 2
        share = sq.groupby(families).sum()
        for family, value in share.items():
            rows.append(
                {
                    "component": comp,
                    "family": family,
                    "n_features": int(n_per_family[family]),
                    "share_pct": round(float(value) * 100, 2),
                    "share_per_feature_pct": round(float(value) * 100 / n_per_family[family], 3),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# t-SNE
# --------------------------------------------------------------------------- #
def resolve_perplexity(perplexity: float, n_samples: int) -> float:
    """Clip the perplexity to a value valid for ``n_samples``.

    scikit-learn requires ``perplexity < n_samples``; a common rule of thumb
    keeps it below roughly a third of the sample size.
    """
    upper = max(2.0, (n_samples - 1) / 3.0)
    return float(min(perplexity, upper))


def run_tsne(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    perplexity: float = 30.0,
    random_state: int = 42,
    pre_pca_dims: int = 30,
    init: str = "pca",
) -> Tuple[np.ndarray, float]:
    """Return a 2D t-SNE embedding of the standardised features.

    A preliminary PCA to ``pre_pca_dims`` components removes noise and speeds up
    the neighbour search, as recommended in the t-SNE literature.

    ``init="pca"`` (default, used for the figure in the chapter) starts from the
    PCA layout, which preserves the global arrangement better and makes the
    result essentially independent of the seed. ``init="random"`` is used by
    :func:`tsne_stability`: only with a random start does the seed actually
    change the optimisation, which is what a stability check needs.

    Returns
    -------
    embedding : ndarray of shape (n_samples, 2)
    perplexity : float
        The perplexity actually used (after clipping).
    """
    cols, _ = usable_feature_columns(df, feature_cols)
    X = _standardised_matrix(df, cols)

    n_pre = min(pre_pca_dims, X.shape[1], max(1, X.shape[0] - 1))
    X = PCA(n_components=n_pre, random_state=random_state).fit_transform(X)

    perp = resolve_perplexity(perplexity, X.shape[0])
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init=init,
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(X), perp


def tsne_stability(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    perplexities: Sequence[float],
    seeds: Sequence[int],
    pre_pca_dims: int = 30,
) -> Dict[Tuple[float, int], np.ndarray]:
    """Recompute the t-SNE for every (perplexity, seed) pair.

    Every run starts from a RANDOM initialisation (with a PCA start the seed
    would have no effect and all runs would be identical). Perplexities that
    collapse to the same value after clipping are computed once. The returned
    dict is keyed by ``(effective_perplexity, seed)``.
    """
    n = len(df)
    effective = sorted({resolve_perplexity(p, n) for p in perplexities})
    runs: Dict[Tuple[float, int], np.ndarray] = {}
    for perp in effective:
        for seed in seeds:
            emb, _ = run_tsne(df, feature_cols, perplexity=perp, random_state=int(seed),
                              pre_pca_dims=pre_pca_dims, init="random")
            runs[(perp, int(seed))] = emb
    return runs


# --------------------------------------------------------------------------- #
# Sample counts
# --------------------------------------------------------------------------- #
def ordered_levels(present: Sequence[str], preferred: Sequence[str] | None = None) -> List[str]:
    """Levels in the preferred order first, then any remaining ones alphabetically."""
    present = list(dict.fromkeys(present))
    preferred = list(preferred or [])
    head = [x for x in preferred if x in present]
    tail = sorted(x for x in present if x not in head)
    return head + tail


def class_counts(
    df: pd.DataFrame,
    fabric_order: Sequence[str] | None = None,
    class_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Number of samples per fabric (rows) and class (columns), with totals."""
    table = pd.crosstab(df["fabric"], df["label"])
    table = table.reindex(ordered_levels(table.index, fabric_order))
    table = table[ordered_levels(table.columns, class_order)]
    table["Totale"] = table.sum(axis=1)
    table.loc["Totale"] = table.sum(axis=0)
    return table
