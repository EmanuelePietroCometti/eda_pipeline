"""
masks.py
========

Numerosità per versione e gruppo, con l'area mediana dei difetti presa dalle maschere.

L'unità dell'area è la componente connessa della maschera (un "difetto"). Per ogni versione e
gruppo (`train_good`, `train_dust`, `train_def`, `test_good`, `test_dust`, `test_def`) la tabella
riporta il numero di immagini; per i difetti anche il numero di componenti e l'area mediana, sia
per tutte le classi insieme (``ALL``) sia per classe. Buoni e polvere non hanno maschera: area vuota.
"""

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

GROUP_ORDER = ["train_good", "train_dust", "train_def", "test_good", "test_dust", "test_def"]
COMP_COLUMNS = ["filename", "group", "label", "component", "area_px", "area_pct"]


def component_table(idx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Una riga per componente connessa di ogni maschera non vuota."""
    thr = cfg["masks"]["threshold"]
    conn = cfg["masks"]["connectivity"]
    rows = []
    for r in idx[idx["has_mask"]].itertuples(index=False):
        mask = cv2.imread(r.mask_path, cv2.IMREAD_GRAYSCALE) > thr
        h, w = mask.shape
        for k, p in enumerate(regionprops(label(mask, connectivity=conn)), start=1):
            rows.append({"filename": r.filename, "group": r.group, "label": r.label, "component": k,
                         "area_px": int(p.area), "area_pct": 100 * p.area / (h * w)})
    return pd.DataFrame(rows, columns=COMP_COLUMNS)


def count_table(idx_by_version: dict, cfg: dict) -> pd.DataFrame:
    """Numerosità per versione, gruppo e classe, con l'area mediana dei difetti.

    `class` è ``-`` per buoni e polvere, ``ALL`` per tutti i difetti del gruppo, altrimenti il
    nome della cartella di difetto.
    """
    parts = []
    for v, idx in idx_by_version.items():
        imgs = pd.concat([idx.assign(cls=np.where(idx["is_defect"], "ALL", "-")),
                          idx[idx["is_defect"]].assign(cls=lambda d: d["label"])])
        n = imgs.groupby(["group", "cls"]).size().rename("n_images")
        comps = component_table(idx, cfg)
        comps = pd.concat([comps.assign(cls="ALL"), comps.assign(cls=comps["label"])])
        area = comps.groupby(["group", "cls"]).agg(
            n_defects=("area_px", "size"), area_px_median=("area_px", "median"),
            area_pct_median=("area_pct", "median"))
        t = n.to_frame().join(area).reset_index()
        t["n_defects"] = t["n_defects"].astype("Int64")
        t.insert(0, "version", v)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True).rename(columns={"cls": "class"})
    # ordine di lettura: per versione, i gruppi come in GROUP_ORDER, ALL prima delle singole classi
    out["_g"] = out["group"].map(GROUP_ORDER.index)
    out["_c"] = (~out["class"].isin(["-", "ALL"])).astype(int)
    return out.sort_values(["_g", "_c", "class", "version"]).drop(columns=["_g", "_c"]).reset_index(drop=True)


def run(idx_by_version: dict, cfg: dict, out):
    tab = count_table(idx_by_version, cfg)
    tab.to_csv(out / "counts.csv", index=False)
    wide = tab.pivot(index=["group", "class"], columns="version", values=["n_images", "area_px_median"])
    wide = wide.reindex(tab[["group", "class"]].drop_duplicates().itertuples(index=False, name=None))
    wide.to_csv(out / "counts_wide.csv")
    show = pd.concat({"n_images": wide["n_images"].fillna(0).astype(int),
                      "area_px_median": wide["area_px_median"].round(1).astype("object").fillna("")}, axis=1)
    print("Numerosità per gruppo e versione, area mediana del difetto [px della patch]:\n" + show.to_string())
    return tab
