"""
visualizzation.py
=================

Publication-ready figures for the textile EDA (thesis-grade PNGs at 300 DPI).

All plots share a consistent class ordering and colour palette so that the same
class is always drawn in the same colour across every figure. Three families of
plots are provided:

* :func:`plot_multivariate_violins` — class-wise violin plots for a set of
  metrics (e.g. HSV saturation variance, GLCM contrast).
* :func:`plot_tsne_scatter` — the 2D t-SNE embedding coloured by class.
* :func:`plot_overlap_density` — overlapping density curves of a GLCM metric for
  two ambiguous classes (e.g. dusty *Good* vs *Paglie*).

Only ``seaborn`` and ``matplotlib`` are used for rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

# Canonical left-to-right class ordering used on every categorical axis.
CLASS_ORDER: List[str] = ["Good", "Dust", "Macchie", "Paglie", "Nodi", "Rotture"]

# Fixed class -> colour mapping (colour-blind friendly qualitative palette).
_PALETTE_COLORS = sns.color_palette("colorblind", len(CLASS_ORDER))
CLASS_PALETTE: Dict[str, tuple] = dict(zip(CLASS_ORDER, _PALETTE_COLORS))

# Human-readable axis labels for the internal feature column names.
PRETTY_LABELS: Dict[str, str] = {
    "hsv_s_var": "HSV Saturation variance",
    "hsv_h_var": "HSV Hue variance",
    "hsv_v_mean": "HSV Value (brightness) mean",
    "lab_a_mean": "Lab a* mean (green-red)",
    "lab_b_mean": "Lab b* mean (blue-yellow)",
    "glcm_contrast": "GLCM Contrast",
    "glcm_homogeneity": "GLCM Homogeneity",
    "glcm_energy": "GLCM Energy",
    "glcm_correlation": "GLCM Correlation",
    "lbp_var": "LBP variance",
}


def _nice(col: str) -> str:
    """Map an internal column name to a human-readable axis label."""
    return PRETTY_LABELS.get(col, col.replace("_", " "))


def set_publication_style() -> None:
    """Configure seaborn/matplotlib for clean, academic-looking figures."""
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 120,       # on-screen preview
            "savefig.dpi": 300,      # high-resolution export for the thesis
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,      # editable text if exported to PDF/vector
        }
    )


def _present_order(df, label_col: str) -> List[str]:
    """Return CLASS_ORDER restricted to the classes actually present in ``df``."""
    present = set(df[label_col].unique())
    return [c for c in CLASS_ORDER if c in present]


# --------------------------------------------------------------------------- #
# 1. Multivariate violin plots
# --------------------------------------------------------------------------- #
def plot_multivariate_violins(
    df,
    metrics: Sequence[str],
    out_dir: Path,
    label_col: str = "label",
    filename: str = "violin_multivariate.png",
) -> Path:
    """Draw one violin sub-plot per metric, classes on the x-axis.

    A violin plot overlays a mirrored kernel-density estimate on a box-plot,
    exposing the full shape (modality, skew, spread) of each metric per class —
    far more informative than a bare box-plot for the heavy-tailed texture and
    colour distributions studied here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = _present_order(df, label_col)

    n = len(metrics)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False
    )

    for ax, metric in zip(axes.ravel(), metrics):
        sns.violinplot(
            data=df, x=label_col, y=metric, order=order,
            hue=label_col, palette=CLASS_PALETTE, legend=False,
            cut=0, inner="box", density_norm="width", ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(_nice(metric))
        ax.set_title(_nice(metric))
        ax.tick_params(axis="x", rotation=20)

    # Hide any unused axes in the grid.
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    fig.suptitle("Class-wise distribution of discriminative features", y=1.02)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 2. t-SNE scatter
# --------------------------------------------------------------------------- #
def plot_tsne_scatter(
    df,
    out_dir: Path,
    x_col: str = "tsne_1",
    y_col: str = "tsne_2",
    label_col: str = "label",
    filename: str = "tsne_scatter.png",
) -> Path:
    """Scatter the 2D t-SNE embedding, coloured by class.

    This figure visualises how the broad ``Good`` manifold spreads across the
    latent space and how structural defects and dust overlap with it — the core
    qualitative evidence for the ambiguity discussed in the thesis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = _present_order(df, label_col)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.scatterplot(
        data=df, x=x_col, y=y_col,
        hue=label_col, hue_order=order, palette=CLASS_PALETTE,
        s=45, alpha=0.75, edgecolor="black", linewidth=0.3, ax=ax,
    )
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("t-SNE projection of handcrafted features")
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pca_scatter(
    df,
    out_dir: Path,
    label_col: str = "label",
    filename: str = "pca_scatter.png",
) -> Path:
    """Scatter the 2D PCA projection, annotating explained variance if available."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = _present_order(df, label_col)

    evr = df.attrs.get("pca_explained_variance_ratio_")
    xlab = "PC 1" if evr is None else f"PC 1 ({evr[0] * 100:.1f}%)"
    ylab = "PC 2" if evr is None else f"PC 2 ({evr[1] * 100:.1f}%)"

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.scatterplot(
        data=df, x="pca_1", y="pca_2",
        hue=label_col, hue_order=order, palette=CLASS_PALETTE,
        s=45, alpha=0.75, edgecolor="black", linewidth=0.3, ax=ax,
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title("PCA projection of handcrafted features")
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 3. Overlapping density histograms
# --------------------------------------------------------------------------- #
def plot_overlap_density(
    df,
    metric: str,
    class_a: str,
    class_b: str,
    out_dir: Path,
    label_col: str = "label",
    filename: str | None = None,
) -> Path:
    """Overlay the density of ``metric`` for two classes to expose their ambiguity.

    Filled KDE curves (with a light histogram underlay) make the shared area
    between, e.g., dusty *Good* and *Paglie* immediately visible — the visual
    counterpart of the Bhattacharyya / overlap coefficients from :mod:`src.eda`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"overlap_{metric}_{class_a}_vs_{class_b}.png"

    subset = df[df[label_col].isin([class_a, class_b])]
    palette = {c: CLASS_PALETTE.get(c) for c in (class_a, class_b)}

    fig, ax = plt.subplots(figsize=(9, 6))
    # Light histogram context...
    sns.histplot(
        data=subset, x=metric, hue=label_col, hue_order=[class_a, class_b],
        palette=palette, stat="density", common_norm=False,
        element="step", alpha=0.20, ax=ax, legend=False,
    )
    # ...with smooth, filled KDEs on top to highlight the overlap region.
    sns.kdeplot(
        data=subset, x=metric, hue=label_col, hue_order=[class_a, class_b],
        palette=palette, fill=True, alpha=0.35, common_norm=False,
        linewidth=2, ax=ax,
    )
    ax.set_xlabel(_nice(metric))
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution overlap: {class_a} vs {class_b} — {_nice(metric)}")

    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path
