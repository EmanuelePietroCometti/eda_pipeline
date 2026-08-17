"""
eda.py
======

Statistical analysis and dimensionality reduction for the textile EDA.

Given the handcrafted feature :class:`~pandas.DataFrame` produced by
:mod:`src.feature_extraction`, this module provides:

* **Dimensionality reduction** — PCA (linear, variance-maximising) and t-SNE
  (non-linear, neighbourhood-preserving) projections onto 2D.
* **Distribution-overlap metrics** — Bhattacharyya coefficient/distance and the
  Overlapping Coefficient (OVL) to quantify how much the *Paglie* class overlaps
  the dusty *Good* samples on the structural (LBP/GLCM) features.
* **Significance testing** — one-way ANOVA and the non-parametric Kruskal-Wallis
  test on the structural features between ``Good`` and ``Rotture`` / ``Nodi``,
  with a companion effect-size estimate.

Only ``numpy``, ``pandas``, ``scipy`` and ``scikit-learn`` are used.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Metadata columns that are never treated as features.
_META_COLS = ("filename", "label", "group")


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
        prefixes (e.g. ``("glcm_", "lbp_")`` to select structural features).
    """
    cols = [
        c for c in df.columns
        if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if prefixes is not None:
        cols = [c for c in cols if any(c.startswith(p) for p in prefixes)]
    return cols


def structural_feature_columns(df: pd.DataFrame) -> List[str]:
    """Convenience selector for GLCM + LBP (i.e. texture/structural) columns."""
    return get_feature_columns(df, prefixes=("glcm_", "lbp_"))


# --------------------------------------------------------------------------- #
# Dimensionality reduction
# --------------------------------------------------------------------------- #
def run_pca(
    df: pd.DataFrame,
    feature_cols: List[str] | None = None,
    n_components: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Project the samples with PCA and append ``pca_1..pca_n`` columns.

    PCA diagonalises the covariance matrix of the *standardised* features and
    keeps the ``n_components`` directions of largest variance. Standardisation is
    mandatory here because the raw features live on wildly different scales
    (e.g. colour variance in the hundreds vs. LBP histogram bins in [0, 1]).

    The per-component explained-variance ratio is stored on
    ``df.attrs["pca_explained_variance_ratio_"]`` for later annotation of plots.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X = StandardScaler().fit_transform(df[feature_cols].to_numpy())
    pca = PCA(n_components=n_components, random_state=random_state)
    proj = pca.fit_transform(X)

    out = df.copy()
    for i in range(n_components):
        out[f"pca_{i + 1}"] = proj[:, i]
    out.attrs["pca_explained_variance_ratio_"] = pca.explained_variance_ratio_
    return out


def run_tsne(
    df: pd.DataFrame,
    feature_cols: List[str] | None = None,
    n_components: int = 2,
    perplexity: float | None = None,
    pre_pca_dims: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Project the samples with t-SNE and append ``tsne_1..tsne_n`` columns.

    t-SNE models pairwise similarities as conditional probabilities in the
    high-dimensional space (Gaussian kernel) and in the embedding (heavy-tailed
    Student-t kernel), then minimises the Kullback-Leibler divergence between the
    two. It excels at revealing *local* cluster structure and non-linear class
    overlap.

    A preliminary PCA to ``pre_pca_dims`` denoises the input and speeds up the
    neighbour search, as recommended by the original authors. ``perplexity``
    defaults to a data-dependent value bounded to a sensible range.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X = StandardScaler().fit_transform(df[feature_cols].to_numpy())

    # Denoise / compress before the manifold step (never request more components
    # than available samples or features).
    n_pre = min(pre_pca_dims, X.shape[1], max(1, X.shape[0] - 1))
    X = PCA(n_components=n_pre, random_state=random_state).fit_transform(X)

    n_samples = X.shape[0]
    if perplexity is None:
        # Rule of thumb: perplexity < n_samples; keep it in [5, 50].
        perplexity = float(np.clip(n_samples / 3.0, 5.0, 50.0))
    perplexity = min(perplexity, max(2.0, (n_samples - 1) / 3.0))

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    proj = tsne.fit_transform(X)

    out = df.copy()
    for i in range(n_components):
        out[f"tsne_{i + 1}"] = proj[:, i]
    out.attrs["tsne_perplexity_"] = perplexity
    return out


# --------------------------------------------------------------------------- #
# Distribution-overlap metrics
# --------------------------------------------------------------------------- #
def _histogram_pair(a: np.ndarray, b: np.ndarray, bins: int = 50):
    """Bin two samples on a shared support and return normalised histograms."""
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi - lo < 1e-12:  # degenerate: both distributions are a single point
        return None, None
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    width = edges[1] - edges[0]
    # Convert densities into probability masses (sum to 1).
    pa = pa * width
    pb = pb * width
    return pa, pb


def bhattacharyya_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    """Bhattacharyya coefficient ``BC = sum_i sqrt(p_i q_i)`` in ``[0, 1]``.

    ``BC = 1`` means the two empirical distributions are identical, ``BC = 0``
    means disjoint support. It is the discrete estimate of
    :math:`\\int \\sqrt{p(x)q(x)}\\,dx`.
    """
    pa, pb = _histogram_pair(a, b, bins)
    if pa is None:
        return 1.0
    return float(np.sum(np.sqrt(pa * pb)))


def overlapping_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    """Overlapping Coefficient ``OVL = sum_i min(p_i, q_i)`` in ``[0, 1]``.

    OVL is the area shared by the two probability distributions — an intuitive,
    directly interpretable measure of ambiguity between two classes.
    """
    pa, pb = _histogram_pair(a, b, bins)
    if pa is None:
        return 1.0
    return float(np.sum(np.minimum(pa, pb)))


def overlap_metrics(
    df: pd.DataFrame,
    class_a: str,
    class_b: str,
    feature_cols: List[str] | None = None,
    bins: int = 50,
) -> pd.DataFrame:
    """Per-feature distribution overlap between two classes.

    Used to quantify the visual ambiguity between ``Paglie`` and the dusty
    ``Good`` samples on the structural features. For each feature it reports the
    Bhattacharyya coefficient/distance and the overlapping coefficient.

    Parameters
    ----------
    class_a, class_b : str
        Labels (values of the ``label`` column) to compare.
    feature_cols : list of str, optional
        Defaults to the structural (GLCM + LBP) features.
    """
    if feature_cols is None:
        feature_cols = structural_feature_columns(df)

    sa = df.loc[df["label"] == class_a]
    sb = df.loc[df["label"] == class_b]
    if sa.empty or sb.empty:
        raise ValueError(
            f"Cannot compute overlap: '{class_a}' has {len(sa)} samples, "
            f"'{class_b}' has {len(sb)}."
        )

    records = []
    for col in feature_cols:
        a = sa[col].to_numpy()
        b = sb[col].to_numpy()
        bc = bhattacharyya_coefficient(a, b, bins)
        ovl = overlapping_coefficient(a, b, bins)
        records.append(
            {
                "feature": col,
                "bhattacharyya_coeff": bc,
                # Distance = -ln(BC); +inf when disjoint. 0 when identical.
                "bhattacharyya_dist": float(-np.log(bc)) if bc > 0 else np.inf,
                "overlap_coeff": ovl,
            }
        )
    # Rank by ambiguity: the most-overlapping features first.
    return (
        pd.DataFrame(records)
        .sort_values("overlap_coeff", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Significance testing
# --------------------------------------------------------------------------- #
def _epsilon_squared_kw(h_stat: float, n_total: int, k_groups: int) -> float:
    """Epsilon-squared effect size for Kruskal-Wallis.

    :math:`\\epsilon^2 = (H - k + 1) / (n - k)`, bounded to ``[0, 1]``.
    """
    denom = n_total - k_groups
    if denom <= 0:
        return float("nan")
    return float(max(0.0, (h_stat - k_groups + 1) / denom))


def _eta_squared_anova(groups: Sequence[np.ndarray]) -> float:
    """Eta-squared effect size for one-way ANOVA (between-group variance share)."""
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    if ss_total < 1e-12:
        return float("nan")
    return float(ss_between / ss_total)


def significance_tests(
    df: pd.DataFrame,
    reference: str = "Good",
    targets: Sequence[str] = ("Rotture", "Nodi"),
    feature_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Test whether each structural feature separates ``reference`` from ``targets``.

    For every ``(target, feature)`` pair the reference class is compared against
    the target class with:

    * **One-way ANOVA** (:func:`scipy.stats.f_oneway`) — parametric test on the
      equality of the two group means; sensitive but assumes normality and equal
      variance.
    * **Kruskal-Wallis** (:func:`scipy.stats.kruskal`) — non-parametric rank test
      on the equality of distributions; robust to the non-Gaussian, heavy-tailed
      nature of texture features.

    Effect sizes (eta^2 for ANOVA, epsilon^2 for Kruskal-Wallis) accompany the
    p-values so that statistical significance is not confused with practical
    magnitude.

    Returns a tidy DataFrame sorted by the Kruskal-Wallis p-value.
    """
    if feature_cols is None:
        feature_cols = structural_feature_columns(df)

    ref = df.loc[df["label"] == reference]
    if ref.empty:
        raise ValueError(f"Reference class '{reference}' has no samples.")

    records = []
    for target in targets:
        tgt = df.loc[df["label"] == target]
        if tgt.empty:
            print(f"Warning: target class '{target}' not present; skipping.")
            continue

        for col in feature_cols:
            a = ref[col].to_numpy()
            b = tgt[col].to_numpy()

            # Constant-across-both features break the tests; report NaN cleanly.
            if np.ptp(np.concatenate([a, b])) < 1e-12:
                f_stat = f_p = h_stat = h_p = np.nan
                eta2 = eps2 = np.nan
            else:
                f_stat, f_p = stats.f_oneway(a, b)
                h_stat, h_p = stats.kruskal(a, b)
                eta2 = _eta_squared_anova([a, b])
                eps2 = _epsilon_squared_kw(h_stat, len(a) + len(b), 2)

            records.append(
                {
                    "comparison": f"{reference} vs {target}",
                    "feature": col,
                    "anova_F": f_stat,
                    "anova_p": f_p,
                    "anova_eta2": eta2,
                    "kruskal_H": h_stat,
                    "kruskal_p": h_p,
                    "kruskal_eps2": eps2,
                }
            )

    result = pd.DataFrame(records)
    return result.sort_values(["comparison", "kruskal_p"]).reset_index(drop=True)
