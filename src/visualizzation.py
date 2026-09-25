"""
visualizzation.py
=================

Publication-ready figures for the textile EDA (thesis-grade PNGs at 300 DPI).

Every figure reports the number of samples of each group, so that a class with
a handful of samples (e.g. the dust) is never visually confused with a well
populated one. Colours are fixed per class and per fabric across all figures,
and every group also has its own marker, so figures remain readable in black
and white and for colour-blind readers.

Figures provided:

* :func:`plot_scree` — explained variance per PCA component and cumulative.
* :func:`plot_pca_scatter` — 2D scatter of two PCA components.
* :func:`plot_tsne_scatter` — 2D t-SNE embedding.
* :func:`plot_tsne_grid` — t-SNE stability grid (perplexity x seed).
* :func:`plot_loadings` — features with the largest loadings per component.
* :func:`plot_family_share` — share of each component due to colour/GLCM/LBP.
* :func:`plot_feature_distributions` — violins for well populated groups,
  individual points for every group.

Only ``seaborn`` and ``matplotlib`` are used for rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# Human-readable axis labels for the internal feature column names.
PRETTY_LABELS: Dict[str, str] = {
    "hsv_s_mean": "HSV Saturation mean",
    "hsv_s_var": "HSV Saturation variance",
    "hsv_v_mean": "HSV Value (brightness) mean",
    "hsv_v_var": "HSV Value (brightness) variance",
    "lab_l_mean": "Lab L* mean (lightness)",
    "lab_a_mean": "Lab a* mean (green-red)",
    "lab_b_mean": "Lab b* mean (blue-yellow)",
    "glcm_contrast": "GLCM Contrast",
    "glcm_homogeneity": "GLCM Homogeneity",
    "glcm_energy": "GLCM Energy",
    "glcm_correlation": "GLCM Correlation",
    "lbp_code_dispersion": "LBP code dispersion",
}

_CB = sns.color_palette("colorblind", 8)
FAMILY_PALETTE: Dict[str, tuple] = {"colour": _CB[0], "glcm": _CB[1], "lbp": _CB[2], "other": _CB[7]}
FAMILY_HATCH: Dict[str, str] = {"colour": "", "glcm": "//", "lbp": "..", "other": "xx"}
FAMILY_NAME: Dict[str, str] = {"colour": "Colore (HSV/Lab)", "glcm": "GLCM", "lbp": "LBP", "other": "Altro"}

_MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")
_OTHER_COLOUR = (0.78, 0.78, 0.78)


_PREFIX = {"hsv": "HSV", "lab": "Lab", "glcm": "GLCM", "lbp": "LBP"}
_STAT = {"mean": "mean", "var": "variance", "skew": "skewness", "kurt": "kurtosis"}


def _nice(col: str) -> str:
    """Map an internal column name to a human-readable axis label."""
    if col in PRETTY_LABELS:
        return PRETTY_LABELS[col]
    parts = col.split("_")
    if parts[0] == "lbp" and len(parts) == 3 and parts[1] == "hist":
        return f"LBP bin {parts[2]}"
    if parts[0] in ("hsv", "lab") and len(parts) == 3:
        channel = ({"l": "L*", "a": "a*", "b": "b*"}[parts[1]] if parts[0] == "lab"
                   else parts[1].upper())
        return f"{_PREFIX[parts[0]]} {channel} {_STAT.get(parts[2], parts[2])}"
    return " ".join(_PREFIX.get(p, p) for p in parts)


def set_publication_style() -> None:
    """Configure seaborn/matplotlib for clean, academic-looking figures."""
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 120,       # on-screen preview
            "savefig.dpi": 300,      # high-resolution export for the thesis
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,      # editable text if exported to PDF/vector
        }
    )


# --------------------------------------------------------------------------- #
# Styles shared by all figures
# --------------------------------------------------------------------------- #
def make_style(levels: Sequence[str], palette: str = "colorblind") -> Tuple[Dict[str, tuple], Dict[str, str]]:
    """Fixed colour and marker for every level, in the given order.

    Build it ONCE from the full list of classes (or fabrics) and reuse it in
    every figure, so that the same group always has the same look.
    """
    if palette == "colorblind":
        # Reordered so that consecutive classes never get similar hues: in the
        # default order orange (Dust) and vermilion would fall on neighbouring
        # classes, which is exactly the pair (Dust vs Paglie) we need to tell apart.
        base = sns.color_palette("colorblind", 10)
        colours = [base[i] for i in (0, 1, 2, 4, 5, 6, 9, 3, 8, 7)]
    else:
        colours = sns.color_palette(palette, max(len(levels), 3))
    colour_map = {lvl: colours[i % len(colours)] for i, lvl in enumerate(levels)}
    marker_map = {lvl: _MARKERS[i % len(_MARKERS)] for i, lvl in enumerate(levels)}
    return colour_map, marker_map


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _scatter_groups(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    groups: pd.Series,
    order: Sequence[str],
    colours: Mapping[str, tuple],
    markers: Mapping[str, str],
    highlight: Sequence[str] | None = None,
    emphasize: Sequence[str] = (),
    point_size: float = 28.0,
) -> List[Line2D]:
    """Draw one scatter layer per group and return the legend handles.

    * Groups are drawn from the most to the least populated, so small groups
      are never hidden under large ones; ``emphasize`` groups are drawn last,
      larger and with a black edge.
    * With ``highlight``, only the listed groups are coloured: all others are
      merged into a single light-grey "other classes" layer drawn first.
    """
    groups = pd.Series(np.asarray(groups), index=np.arange(len(groups)))
    counts = groups.value_counts()
    present = [g for g in order if g in counts.index]
    handles: List[Line2D] = []

    if highlight is not None:
        others = [g for g in present if g not in highlight]
        if others:
            mask = groups.isin(others).to_numpy()
            ax.scatter(x[mask], y[mask], s=point_size * 0.6, c=[_OTHER_COLOUR],
                       marker="o", linewidths=0, alpha=0.6, zorder=1)
            handles.append(Line2D([], [], linestyle="", marker="o", markersize=6,
                                  markerfacecolor=_OTHER_COLOUR, markeredgewidth=0,
                                  label=f"Altre classi (n={int(mask.sum())})"))
        present = [g for g in present if g in highlight]

    draw_order = sorted(present, key=lambda g: (g in emphasize, -counts[g]))
    for rank, g in enumerate(draw_order):
        mask = (groups == g).to_numpy()
        big = g in emphasize
        ax.scatter(
            x[mask], y[mask],
            s=point_size * (2.6 if big else 1.0),
            c=[colours.get(g, _OTHER_COLOUR)],
            marker=markers.get(g, "o"),
            edgecolors="black", linewidths=0.9 if big else 0.25,
            alpha=0.95 if big else 0.7,
            zorder=3 + rank,
        )
    # Legend in the canonical order, not in drawing order.
    for g in present:
        big = g in emphasize
        handles.append(Line2D([], [], linestyle="", marker=markers.get(g, "o"),
                              markersize=9 if big else 7,
                              markerfacecolor=colours.get(g, _OTHER_COLOUR),
                              markeredgecolor="black", markeredgewidth=0.9 if big else 0.25,
                              label=f"{g} (n={int(counts[g])})"))
    return handles


# --------------------------------------------------------------------------- #
# PCA figures
# --------------------------------------------------------------------------- #
def plot_scree(explained_variance_ratio: np.ndarray, path: Path, title: str) -> Path:
    """Bar = variance explained by each component, line = cumulative variance."""
    evr = np.asarray(explained_variance_ratio) * 100
    comps = np.arange(1, len(evr) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(comps, evr, color=sns.color_palette("colorblind")[0], alpha=0.85,
                  label="Varianza della componente")
    for bar, v in zip(bars, evr):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.plot(comps, np.cumsum(evr), color="black", marker="o", linewidth=1.5,
            label="Varianza cumulata")
    ax.set_xticks(comps)
    ax.set_xticklabels([f"PC{i}" for i in comps])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Varianza spiegata (%)")
    ax.set_title(title)
    ax.legend(loc="center right", frameon=True)
    fig.tight_layout()
    return _save(fig, path)


def plot_pca_scatter(
    scores: pd.DataFrame,
    explained_variance_ratio: np.ndarray,
    groups: pd.Series,
    pcs: Tuple[int, int],
    order: Sequence[str],
    colours: Mapping[str, tuple],
    markers: Mapping[str, str],
    path: Path,
    title: str,
    legend_title: str,
    highlight: Sequence[str] | None = None,
    emphasize: Sequence[str] = (),
) -> Path:
    """Scatter of two PCA components (1-based indices in ``pcs``)."""
    i, j = pcs
    evr = np.asarray(explained_variance_ratio) * 100
    fig, ax = plt.subplots(figsize=(8.5, 7))
    handles = _scatter_groups(
        ax, scores[f"PC{i}"].to_numpy(), scores[f"PC{j}"].to_numpy(),
        groups, order, colours, markers, highlight=highlight, emphasize=emphasize,
    )
    ax.axhline(0, color="grey", linewidth=0.6, zorder=0)
    ax.axvline(0, color="grey", linewidth=0.6, zorder=0)
    ax.set_xlabel(f"PC{i} ({evr[i - 1]:.1f}%)")
    ax.set_ylabel(f"PC{j} ({evr[j - 1]:.1f}%)")
    ax.set_title(title)
    ax.legend(handles=handles, title=legend_title, bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=True)
    fig.tight_layout()
    return _save(fig, path)


def plot_loadings(
    loadings: pd.DataFrame,
    explained_variance_ratio: np.ndarray,
    components: Sequence[str],
    top_k: int,
    path: Path,
    title: str,
) -> Path:
    """Horizontal bars of the ``top_k`` largest |loadings| for each component.

    Bars are coloured by feature family, so it is immediately visible whether a
    component is driven by colour or by texture descriptors.
    """
    from .eda import feature_family  # local import: avoid a circular import

    comps = [c for c in components if c in loadings.columns]
    evr = np.asarray(explained_variance_ratio) * 100
    fig, axes = plt.subplots(1, len(comps), figsize=(5.2 * len(comps), 0.45 * top_k + 1.8),
                             squeeze=False)
    for ax, comp in zip(axes.ravel(), comps):
        col = loadings[comp]
        top = col.reindex(col.abs().sort_values(ascending=False).index[:top_k])[::-1]
        fams = [feature_family(f) for f in top.index]
        bars = ax.barh([_nice(f) for f in top.index], top.to_numpy(),
                       color=[FAMILY_PALETTE[f] for f in fams], edgecolor="black", linewidth=0.4)
        for bar, fam in zip(bars, fams):
            bar.set_hatch(FAMILY_HATCH[fam])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-1, 1)
        idx = int(comp[2:]) - 1
        ax.set_title(f"{comp} ({evr[idx]:.1f}%)")
        ax.set_xlabel("Loading")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=FAMILY_PALETTE[f], edgecolor="black", hatch=FAMILY_HATCH[f],
                     label=FAMILY_NAME[f]) for f in ("colour", "glcm", "lbp")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def plot_family_share(share: pd.DataFrame, path: Path, title: str) -> Path:
    """100% stacked bars: fraction of each component due to each feature family."""
    comps = list(dict.fromkeys(share["component"]))
    families = [f for f in ("colour", "glcm", "lbp", "other") if f in set(share["family"])]

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(comps) + 1.8))
    left = np.zeros(len(comps))
    for fam in families:
        vals = np.array([
            share.loc[(share["component"] == c) & (share["family"] == fam), "share_pct"].sum()
            for c in comps
        ])
        n_feat = int(share.loc[share["family"] == fam, "n_features"].iloc[0])
        ax.barh(comps, vals, left=left, color=FAMILY_PALETTE[fam], hatch=FAMILY_HATCH[fam],
                edgecolor="black", linewidth=0.4, label=f"{FAMILY_NAME[fam]} ({n_feat} feature)")
        for y, (l, v) in enumerate(zip(left, vals)):
            if v >= 6:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="none", alpha=0.85))
        left += vals
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Quota della componente (% dei loading al quadrato)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(families), frameon=False)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# t-SNE figures
# --------------------------------------------------------------------------- #
def plot_tsne_scatter(
    embedding: np.ndarray,
    groups: pd.Series,
    order: Sequence[str],
    colours: Mapping[str, tuple],
    markers: Mapping[str, str],
    path: Path,
    title: str,
    legend_title: str,
    highlight: Sequence[str] | None = None,
    emphasize: Sequence[str] = (),
) -> Path:
    """Scatter of a 2D t-SNE embedding.

    Axes carry no unit: t-SNE coordinates, distances between clusters and
    cluster sizes are not interpretable, so ticks are hidden on purpose.
    """
    fig, ax = plt.subplots(figsize=(8.5, 7))
    handles = _scatter_groups(ax, embedding[:, 0], embedding[:, 1], groups, order,
                              colours, markers, highlight=highlight, emphasize=emphasize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(handles=handles, title=legend_title, bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=True)
    fig.tight_layout()
    return _save(fig, path)


def plot_tsne_grid(
    runs: Mapping[Tuple[float, int], np.ndarray],
    groups: pd.Series,
    order: Sequence[str],
    colours: Mapping[str, tuple],
    markers: Mapping[str, str],
    path: Path,
    title: str,
    legend_title: str,
    emphasize: Sequence[str] = (),
) -> Path:
    """Small multiples: one t-SNE per (perplexity, seed), rows = perplexity.

    Runs use a random initialisation (see :func:`src.eda.tsne_stability`), so
    panels are rotated/mirrored with respect to each other: compare which
    groups stay together, not where they are.
    """
    perps = sorted({p for p, _ in runs})
    seeds = sorted({s for _, s in runs})
    fig, axes = plt.subplots(len(perps), len(seeds),
                             figsize=(3.6 * len(seeds), 3.4 * len(perps)), squeeze=False)
    handles: List[Line2D] = []
    for r, perp in enumerate(perps):
        for c, seed in enumerate(seeds):
            ax = axes[r, c]
            emb = runs[(perp, seed)]
            handles = _scatter_groups(ax, emb[:, 0], emb[:, 1], groups, order, colours,
                                      markers, emphasize=emphasize, point_size=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"perplexity {perp:.3g} · seed {seed}", fontsize=10, fontweight="normal")
    fig.legend(handles=handles, title=legend_title, loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=True)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Feature distributions
# --------------------------------------------------------------------------- #
def plot_feature_distributions(
    df: pd.DataFrame,
    x_col: str,
    features: Sequence[str],
    order: Sequence[str],
    colours: Mapping[str, tuple],
    path: Path,
    title: str,
    min_n_violin: int = 20,
) -> Path:
    """One panel per feature, groups on the x-axis.

    Groups with at least ``min_n_violin`` samples get a violin (density shape,
    quartiles as dashed lines) plus their points; smaller groups are shown
    ONLY as points, because a density estimated from a handful of samples is an
    artefact of the kernel, not a property of the data. The sample size of
    every group is written under its tick.
    """
    counts = df[x_col].value_counts()
    order = [g for g in order if g in counts.index]
    big = [g for g in order if counts[g] >= min_n_violin]
    small = [g for g in order if counts[g] < min_n_violin]
    features = [f for f in features if f in df.columns]
    if not features:
        raise ValueError("None of the requested distribution features is available.")

    n = len(features)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    panel_w = 1.6 * len(order) + 3.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, 4.6 * nrows), squeeze=False)

    for ax, feat in zip(axes.ravel(), features):
        if big:
            sub = df[df[x_col].isin(big)]
            sns.violinplot(data=sub, x=x_col, y=feat, order=order, hue=x_col,
                           hue_order=big, palette=colours, legend=False, cut=0,
                           inner="quart", density_norm="width", linewidth=0.8,
                           saturation=0.9, ax=ax)
            for coll in ax.collections:
                coll.set_alpha(0.45)
            sns.stripplot(data=sub, x=x_col, y=feat, order=order, color="black",
                          size=1.8, alpha=0.35, jitter=0.22, ax=ax)
        if small:
            sub = df[df[x_col].isin(small)]
            sns.stripplot(data=sub, x=x_col, y=feat, order=order, hue=x_col,
                          hue_order=small, palette=colours, legend=False, size=6,
                          jitter=0.12, edgecolor="black", linewidth=0.8, ax=ax)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{g}\n(n={int(counts[g])})" for g in order])
        ax.set_xlabel("")
        ax.set_ylabel(_nice(feat))
        ax.set_title(_nice(feat))

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return _save(fig, path)
