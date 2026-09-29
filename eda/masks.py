"""
masks.py  (analisi 1)
=====================

Statistiche dei difetti dalle maschere: componenti connesse, area in pixel e in
% della patch, bounding box, e confronto con la dimensione delle celle dei backbone.
"""

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

from .utils import save_fig


def component_table(idx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Una riga per componente connessa di ogni maschera non vuota."""
    thr = cfg["masks"]["threshold"]
    conn = cfg["masks"]["connectivity"]
    model = cfg["general"]["model_input_size"]
    rows = []
    for r in idx[idx["has_mask"]].itertuples(index=False):
        mask = cv2.imread(r.mask_path, cv2.IMREAD_GRAYSCALE) > thr
        h, w = mask.shape
        for k, p in enumerate(regionprops(label(mask, connectivity=conn)), start=1):
            y0, x0, y1, x1 = p.bbox
            rows.append({
                "filename": r.filename, "split": r.split, "label": r.label, "component": k,
                "area_px": int(p.area),
                "area_pct": 100 * p.area / (h * w),
                # i modelli vedono 256 px, le maschere sono a 512: si riscala l'area
                "area_model_px": p.area * (model / h) * (model / w),
                "bbox_h": y1 - y0, "bbox_w": x1 - x0, "extent": float(p.extent),
            })
    return pd.DataFrame(rows)


def summarize(comps: pd.DataFrame, cell_sizes) -> pd.DataFrame:
    """Per split e classe: numerosità, area (mediana e quartili) e frazione sotto una cella."""
    g = comps.groupby(["split", "label"])
    out = g.agg(n_images=("filename", "nunique"), n_components=("area_px", "size"),
                area_px_median=("area_px", "median"),
                area_px_q25=("area_px", lambda s: s.quantile(0.25)),
                area_px_q75=("area_px", lambda s: s.quantile(0.75)),
                area_pct_median=("area_pct", "median")).reset_index()
    for c in cell_sizes:
        # una cella di stride c copre c*c pixel a risoluzione del modello
        frac = g["area_model_px"].apply(lambda s, c=c: (s < c * c).mean()).rename(f"frac_below_{c}px_cell")
        out = out.merge(frac.reset_index(), on=["split", "label"])
    return out


def plot_area_ecdf(comps: pd.DataFrame, cell_sizes, path):
    g = sns.displot(data=comps, x="area_model_px", hue="label", col="split", kind="ecdf",
                    log_scale=True, height=3.6, aspect=1.2, facet_kws={"sharey": True})
    for ax in g.axes.ravel():
        for c in cell_sizes:
            ax.axvline(c * c, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("Area della componente a 256 px [px, scala log]")
    g.set_ylabels("Frazione di componenti")
    return save_fig(g.figure, path)


def plot_area_pct(comps: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6, 0.7 * comps["label"].nunique() + 2))
    sns.boxplot(data=comps, x="area_pct", y="label", hue="split", ax=ax, fliersize=2)
    ax.set_xscale("log")
    ax.set_xlabel("Area della componente [% della patch, scala log]")
    ax.set_ylabel("")
    return save_fig(fig, path)


def run(idx: pd.DataFrame, cfg: dict, out_dir):
    comps = component_table(idx, cfg)
    comps.to_csv(f"{out_dir}/mask_components.csv", index=False)
    if comps.empty:
        print("[maschere] nessuna componente trovata")
        return comps
    cells = cfg["masks"]["cell_sizes_px"]
    summary = summarize(comps, cells)
    summary.to_csv(f"{out_dir}/mask_summary.csv", index=False)
    plot_area_ecdf(comps, cells, f"{out_dir}/figures/mask_area_ecdf.png")
    plot_area_pct(comps, f"{out_dir}/figures/mask_area_pct.png")
    print(summary.round(3).to_string(index=False))
    return comps
