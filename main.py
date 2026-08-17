"""
main.py
=======

Orchestrator for the textile Visual-Anomaly-Detection EDA pipeline.

Steps
-----
1. Extract handcrafted colour (HSV/Lab) and texture (GLCM/LBP) features from the
   512x512 patches described in ``config.yaml``.
2. Reduce dimensionality with PCA and t-SNE.
3. Quantify the Paglie-vs-dusty-Good overlap and run ANOVA / Kruskal-Wallis
   significance tests on the structural features (Good vs Rotture / Nodi).
4. Render publication-ready figures and dump every intermediate table to disk.

Usage
-----
    python main.py                         # uses paths from config.yaml
    python main.py --root /path/dataset    # override the dataset root
    python main.py --config other.yaml     # use a different config file
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.feature_extraction import extract_features
from src.eda import (
    get_feature_columns,
    structural_feature_columns,
    run_pca,
    run_tsne,
    overlap_metrics,
    significance_tests,
)
from src.visualizzation import (
    set_publication_style,
    plot_multivariate_violins,
    plot_tsne_scatter,
    plot_pca_scatter,
    plot_overlap_density,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Textile EDA: handcrafted features, statistics and figures."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Override the dataset root directory from the config.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override the output directory from the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    gen = config.setdefault("general_configuration", {})

    # CLI overrides take precedence over the config file.
    if args.root:
        gen["dataset_root"] = args.root
    if args.output:
        gen["output_dir"] = args.output

    out_dir = Path(gen.get("output_dir", "outputs"))
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Feature extraction ------------------------------------------- #
    features = extract_features(config)
    features.to_csv(out_dir / "features.csv", index=False)
    print(f"[1/4] Features saved -> {out_dir / 'features.csv'}")

    # --- 2. Dimensionality reduction ------------------------------------- #
    random_state = int(gen.get("random_state", 42))
    feature_cols = get_feature_columns(features)
    features = run_pca(features, feature_cols, n_components=2, random_state=random_state)
    features = run_tsne(features, feature_cols, n_components=2, random_state=random_state)
    features.to_csv(out_dir / "features_with_projections.csv", index=False)
    print(f"[2/4] PCA + t-SNE projections computed "
          f"(perplexity={features.attrs.get('tsne_perplexity_'):.1f}).")

    # --- 3. Statistical analysis ----------------------------------------- #
    struct_cols = structural_feature_columns(features)

    # 3a. Overlap between Paglie and dusty Good on structural features.
    try:
        overlap = overlap_metrics(features, class_a="Paglie", class_b="Dust",
                                  feature_cols=struct_cols)
        overlap.to_csv(out_dir / "overlap_paglie_vs_dust.csv", index=False)
        print("[3/4] Paglie-vs-Dust overlap:")
        print(overlap.head(5).to_string(index=False))
    except ValueError as exc:
        overlap = None
        print(f"[3/4] Overlap skipped: {exc}")

    # 3b. ANOVA + Kruskal-Wallis: Good vs Rotture / Nodi.
    tests = significance_tests(
        features, reference="Good", targets=("Rotture", "Nodi"),
        feature_cols=struct_cols,
    )
    tests.to_csv(out_dir / "significance_tests.csv", index=False)
    print("      Significance tests (top rows):")
    print(tests.head(6).to_string(index=False))

    # --- 4. Visualisation ------------------------------------------------ #
    set_publication_style()

    violin_metrics = [
        m for m in ("hsv_s_var", "glcm_contrast", "glcm_energy", "lbp_var")
        if m in features.columns
    ]
    saved = [
        plot_multivariate_violins(features, violin_metrics, fig_dir),
        plot_tsne_scatter(features, fig_dir),
        plot_pca_scatter(features, fig_dir),
    ]

    # Overlap density: dusty Good vs Paglie on the GLCM energy/contrast metrics.
    for metric in ("glcm_energy", "glcm_contrast"):
        if metric in features.columns and {"Dust", "Paglie"}.issubset(
            set(features["label"].unique())
        ):
            saved.append(
                plot_overlap_density(features, metric, "Dust", "Paglie", fig_dir)
            )

    print(f"[4/4] Figures saved to {fig_dir}:")
    for p in saved:
        print(f"   - {p}")


if __name__ == "__main__":
    main()
