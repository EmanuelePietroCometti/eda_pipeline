"""
main.py
=======

Orchestrator for the textile Visual-Anomaly-Detection EDA (thesis chapter 4).

Steps
-----
1. Extract handcrafted colour (HSV/Lab) and texture (GLCM/LBP) features from the
   512x512 patches described in ``config.yaml`` (one sub-folder per fabric).
2. Section 4.2 — differences between fabrics: PCA on the Good samples only,
   coloured by fabric; scree plot, loadings, share of colour vs texture per
   component, feature distributions per fabric.
3. Section 4.3 — defects vs sound fabric: for EACH fabric, PCA and t-SNE
   coloured by class, plus a t-SNE stability grid for the appendix.
4. Section 4.4 — dust: for each fabric that contains dust, the same projections
   with only Good, Dust and the reference defect coloured, plus the feature
   distributions with individual points.

The analysis is purely descriptive (PCA and t-SNE); no hypothesis test is run.
Every table is written as CSV and every figure as a 300-DPI PNG under
``output_dir``; ``run_info.json`` records configuration and library versions.

Usage
-----
    python main.py                         # uses paths from config.yaml
    python main.py --root /path/dataset    # override the dataset root
    python main.py --config other.yaml     # use a different config file
    python main.py --features outputs/features.csv   # skip feature extraction
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.config import load_config
from src.feature_extraction import extract_features
from src.eda import (
    PCAResult,
    class_counts,
    family_share,
    get_feature_columns,
    ordered_levels,
    run_pca,
    run_tsne,
    top_loadings,
    tsne_stability,
)
from src.visualizzation import (
    make_style,
    plot_family_share,
    plot_feature_distributions,
    plot_loadings,
    plot_pca_scatter,
    plot_scree,
    plot_tsne_grid,
    plot_tsne_scatter,
    set_publication_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Textile EDA (chapter 4): handcrafted features, PCA, t-SNE."
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument("--root", type=str, default=None,
                        help="Override the dataset root directory from the config.")
    parser.add_argument("--output", type=str, default=None,
                        help="Override the output directory from the config.")
    parser.add_argument("--features", type=str, default=None,
                        help="Reuse an existing features.csv instead of re-extracting.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _save_pca_tables(res: PCAResult, out: Path, components: Sequence[str], top_k: int) -> None:
    """Explained variance, full loadings, top loadings and family share as CSV."""
    res.explained_variance_table().to_csv(out / "pca_varianza_spiegata.csv", index=False)
    res.loadings.round(4).to_csv(out / "pca_loadings.csv", index_label="feature")
    tops = [top_loadings(res, c, top_k).assign(component=c)
            for c in components if c in res.loadings.columns]
    if tops:
        pd.concat(tops).to_csv(out / "pca_loadings_top.csv", index=False)
    family_share(res, components).to_csv(out / "pca_quota_famiglie.csv", index=False)


def _valid_pairs(pairs: Sequence[Sequence[int]], n_components: int) -> List[tuple]:
    return [tuple(p) for p in pairs if max(p) <= n_components]


def _pca_figures(
    res: PCAResult,
    groups: pd.Series,
    order: Sequence[str],
    colours: Dict,
    markers: Dict,
    out: Path,
    label: str,
    legend_title: str,
    pca_cfg: dict,
    highlight: Sequence[str] | None = None,
    emphasize: Sequence[str] = (),
    with_loadings: bool = True,
) -> List[Path]:
    """Scree plot, scatter plots of the requested PC pairs, loadings, family share."""
    saved: List[Path] = []
    evr = res.explained_variance_ratio
    comps = [f"PC{i}" for i in pca_cfg.get("loadings_components", [1, 2, 3])]

    if with_loadings:
        saved.append(plot_scree(evr, out / "pca_scree.png", f"Varianza spiegata — {label}"))
    for i, j in _valid_pairs(pca_cfg.get("plot_pairs", [[1, 2]]), len(evr)):
        saved.append(plot_pca_scatter(
            res.scores, evr, groups, (i, j), order, colours, markers,
            out / f"pca_pc{i}_pc{j}.png", f"PCA — {label}", legend_title,
            highlight=highlight, emphasize=emphasize,
        ))
    if with_loadings:
        top_k = int(pca_cfg.get("loadings_top_k", 8))
        saved.append(plot_loadings(res.loadings, evr, comps, top_k,
                                   out / "pca_loadings.png", f"Loadings principali — {label}"))
        saved.append(plot_family_share(family_share(res, comps), out / "pca_quota_famiglie.png",
                                       f"Colore o texture? Composizione delle componenti — {label}"))
    return saved


def _report_dropped(res: PCAResult, label: str) -> None:
    if res.dropped_constant:
        print(f"      [{label}] constant features excluded: {res.dropped_constant}")


# --------------------------------------------------------------------------- #
# Section 4.2 — differences between fabrics (Good only)
# --------------------------------------------------------------------------- #
def section_fabrics(features, feature_cols, cfg, styles, out_root) -> Dict:
    cls = cfg["classes"]
    good_label = cls.get("good_label", "Good")
    pca_cfg = cfg["analysis"]["pca"]
    dist_cfg = cfg["analysis"]["distributions"]

    good = features[features["label"] == good_label]
    fabrics = ordered_levels(good["fabric"].unique(), styles["fabric_order"])
    if len(fabrics) < 2:
        print(f"[4.2] Skipped: fewer than two fabrics with '{good_label}' samples.")
        return {}

    out = out_root / "cap4_2_tessuti"
    out.mkdir(parents=True, exist_ok=True)

    res = run_pca(good, feature_cols, n_components=int(pca_cfg.get("n_components", 10)),
                  random_state=cfg["general_configuration"].get("random_state", 42))
    _report_dropped(res, "4.2")
    comps = [f"PC{i}" for i in pca_cfg.get("loadings_components", [1, 2, 3])]
    _save_pca_tables(res, out, comps, int(pca_cfg.get("loadings_top_k", 8)))

    saved = _pca_figures(res, good["fabric"], fabrics, styles["fabric_colours"],
                         styles["fabric_markers"], out, f"soli {good_label}, tutti i tessuti",
                         "Tessuto", pca_cfg)
    saved.append(plot_feature_distributions(
        good, "fabric", dist_cfg["features"], fabrics, styles["fabric_colours"],
        out / "distribuzioni_per_tessuto.png",
        f"Distribuzione delle feature per tessuto (soli {good_label})",
        min_n_violin=int(dist_cfg.get("min_n_violin", 20)),
    ))

    evr = res.explained_variance_table()
    print(f"[4.2] PCA on {len(good)} {good_label} samples, {len(fabrics)} fabrics:")
    print("      " + evr.head(3).to_string(index=False).replace("\n", "\n      "))
    return {"n_samples": len(good), "fabrics": fabrics,
            "dropped_constant": res.dropped_constant, "figures": [str(p) for p in saved]}


# --------------------------------------------------------------------------- #
# Section 4.3 — defects vs sound fabric, per fabric
# --------------------------------------------------------------------------- #
def section_defects(features, feature_cols, cfg, styles, out_root, fabric) -> Dict:
    cls = cfg["classes"]
    pca_cfg = cfg["analysis"]["pca"]
    tsne_cfg = cfg["analysis"]["tsne"]
    dust = cls.get("dust_label", "Dust")

    sub = features[features["fabric"] == fabric]
    if len(sub) < 5:
        print(f"[4.3] {fabric}: skipped ({len(sub)} samples).")
        return {}

    out = out_root / "cap4_3_difetti" / fabric
    out.mkdir(parents=True, exist_ok=True)
    order = ordered_levels(sub["label"].unique(), styles["class_order"])
    emphasize = [dust] if dust in order else []

    res = run_pca(sub, feature_cols, n_components=int(pca_cfg.get("n_components", 10)),
                  random_state=cfg["general_configuration"].get("random_state", 42))
    _report_dropped(res, f"4.3 {fabric}")
    comps = [f"PC{i}" for i in pca_cfg.get("loadings_components", [1, 2, 3])]
    _save_pca_tables(res, out, comps, int(pca_cfg.get("loadings_top_k", 8)))

    saved = _pca_figures(res, sub["label"], order, styles["class_colours"],
                         styles["class_markers"], out, f"tessuto {fabric}", "Classe",
                         pca_cfg, emphasize=emphasize)

    emb, perp = run_tsne(sub, feature_cols, perplexity=float(tsne_cfg.get("perplexity", 30)),
                         random_state=int(tsne_cfg.get("seed", 42)),
                         pre_pca_dims=int(tsne_cfg.get("pre_pca_dims", 30)))
    saved.append(plot_tsne_scatter(
        emb, sub["label"], order, styles["class_colours"], styles["class_markers"],
        out / "tsne.png", f"t-SNE — tessuto {fabric} (perplexity {perp:.3g})", "Classe",
        emphasize=emphasize,
    ))

    runs = tsne_stability(sub, feature_cols,
                          perplexities=tsne_cfg.get("stability_perplexities", [15, 30, 50]),
                          seeds=tsne_cfg.get("stability_seeds", [0, 1, 2]),
                          pre_pca_dims=int(tsne_cfg.get("pre_pca_dims", 30)))
    saved.append(plot_tsne_grid(
        runs, sub["label"], order, styles["class_colours"], styles["class_markers"],
        out / "tsne_stabilita.png", f"Stabilità del t-SNE (inizializzazione casuale) — tessuto {fabric}", "Classe",
        emphasize=emphasize,
    ))

    # Per-sample coordinates: lets you open the images behind any point.
    n_pc = min(3, res.scores.shape[1])
    coords = sub[["filename", "label"]].copy()
    for k in range(1, n_pc + 1):
        coords[f"PC{k}"] = res.scores[f"PC{k}"].round(4)
    coords["tsne_1"], coords["tsne_2"] = emb[:, 0].round(4), emb[:, 1].round(4)
    coords.to_csv(out / "coordinate_campioni.csv", index=False)

    evr = res.explained_variance_ratio * 100
    print(f"[4.3] {fabric}: {len(sub)} samples | PC1 {evr[0]:.1f}%  PC2 {evr[1]:.1f}%"
          + (f"  PC3 {evr[2]:.1f}%" if len(evr) > 2 else "")
          + f" | t-SNE perplexity {perp:.3g}")
    return {"n_samples": len(sub), "classes": order, "tsne_perplexity": perp,
            "tsne_stability_runs": [list(k) for k in runs],
            "dropped_constant": res.dropped_constant, "figures": [str(p) for p in saved],
            "_pca": res, "_tsne": emb}


# --------------------------------------------------------------------------- #
# Section 4.4 — dust (descriptive only)
# --------------------------------------------------------------------------- #
def section_dust(features, cfg, styles, out_root, fabric, pca_res, tsne_emb, perp) -> Dict:
    cls = cfg["classes"]
    pca_cfg = cfg["analysis"]["pca"]
    dist_cfg = cfg["analysis"]["distributions"]
    good, dust = cls.get("good_label", "Good"), cls.get("dust_label", "Dust")
    ref = cls.get("dust_reference_defect", "Paglie")

    sub = features[features["fabric"] == fabric]
    if dust not in set(sub["label"]):
        return {}

    out = out_root / "cap4_4_polvere" / fabric
    out.mkdir(parents=True, exist_ok=True)
    order = ordered_levels(sub["label"].unique(), styles["class_order"])
    highlight = [c for c in (good, dust, ref) if c in order]
    n_dust = int((sub["label"] == dust).sum())

    # Same projections as section 4.3 (not re-fitted), only the colouring changes.
    saved = _pca_figures(pca_res, sub["label"], order, styles["class_colours"],
                         styles["class_markers"], out, f"polvere, tessuto {fabric}", "Classe",
                         pca_cfg, highlight=highlight, emphasize=[dust], with_loadings=False)
    saved.append(plot_tsne_scatter(
        tsne_emb, sub["label"], order, styles["class_colours"], styles["class_markers"],
        out / "tsne.png", f"t-SNE — polvere, tessuto {fabric} (perplexity {perp:.3g})",
        "Classe", highlight=highlight, emphasize=[dust],
    ))
    saved.append(plot_feature_distributions(
        sub, "label", dist_cfg["features"], order, styles["class_colours"],
        out / "distribuzioni_per_classe.png",
        f"Distribuzione delle feature per classe — tessuto {fabric}",
        min_n_violin=int(dist_cfg.get("min_n_violin", 20)),
    ))
    print(f"[4.4] {fabric}: {n_dust} {dust} samples shown next to {highlight}.")
    return {"n_dust": n_dust, "highlighted": highlight, "figures": [str(p) for p in saved]}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _versions() -> Dict[str, str]:
    import cv2, matplotlib, scipy, seaborn, skimage, sklearn  # noqa: E401
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "scikit-image": skimage.__version__,
            "opencv": cv2.__version__, "matplotlib": matplotlib.__version__,
            "seaborn": seaborn.__version__}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    gen = config.setdefault("general_configuration", {})
    config.setdefault("classes", {})
    config.setdefault("analysis", {}).setdefault("pca", {})
    config["analysis"].setdefault("tsne", {})
    config["analysis"].setdefault("distributions", {}).setdefault(
        "features", ["hsv_s_var", "glcm_contrast", "glcm_energy", "lbp_code_dispersion"])

    if args.root:
        gen["dataset_root"] = args.root
    if args.output:
        gen["output_dir"] = args.output
    out_root = Path(gen.get("output_dir", "outputs"))
    out_root.mkdir(parents=True, exist_ok=True)

    # --- 1. Features -------------------------------------------------------- #
    if args.features:
        features = pd.read_csv(args.features)
        print(f"[1] Features loaded from {args.features} ({len(features)} samples).")
    else:
        features = extract_features(config)
        features.to_csv(out_root / "features.csv", index=False)
        print(f"[1] Features saved -> {out_root / 'features.csv'}")
    feature_cols = get_feature_columns(features)

    cls = config["classes"]
    fabric_order = ordered_levels(features["fabric"].unique(), gen.get("fabric_order"))
    class_order = ordered_levels(features["label"].unique(), cls.get("order"))
    class_colours, class_markers = make_style(class_order, palette="colorblind")
    fabric_colours, fabric_markers = make_style(fabric_order, palette="Set2")
    styles = {"fabric_order": fabric_order, "class_order": class_order,
              "class_colours": class_colours, "class_markers": class_markers,
              "fabric_colours": fabric_colours, "fabric_markers": fabric_markers}

    counts = class_counts(features, fabric_order, class_order)
    counts.to_csv(out_root / "conteggi_tessuto_classe.csv")
    print("[1] Samples per fabric and class:")
    print("    " + counts.to_string().replace("\n", "\n    "))

    set_publication_style()

    # --- 2. Section 4.2 ----------------------------------------------------- #
    info: Dict = {"sections": {}}
    info["sections"]["4.2"] = section_fabrics(features, feature_cols, config, styles, out_root)

    # --- 3/4. Sections 4.3 and 4.4, per fabric ------------------------------ #
    info["sections"]["4.3"], info["sections"]["4.4"] = {}, {}
    for fabric in fabric_order:
        res43 = section_defects(features, feature_cols, config, styles, out_root, fabric)
        if not res43:
            continue
        res44 = section_dust(features, config, styles, out_root, fabric,
                             res43.pop("_pca"), res43.pop("_tsne"), res43["tsne_perplexity"])
        info["sections"]["4.3"][fabric] = res43
        if res44:
            info["sections"]["4.4"][fabric] = res44
    if not info["sections"]["4.4"]:
        print("[4.4] No dust samples in any fabric: section skipped.")

    # --- Run record ---------------------------------------------------------- #
    info.update({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": _versions(),
        "n_samples": int(len(features)),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "config": config,
    })
    with open(out_root / "run_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[done] Tables, figures and run_info.json written to {out_root}/")


if __name__ == "__main__":
    main()
