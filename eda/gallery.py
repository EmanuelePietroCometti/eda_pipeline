"""
gallery.py  (analisi 5)
=======================

Confronto qualitativo tra la classe di difetto più simile alla polvere e la polvere
(`test_dust`). Campione casuale a seme fisso, nessun test statistico.

La polvere non ha maschera (le maschere di `good` sono vuote): la sua area non è
misurabile e non si confronta con quella del difetto.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

from .utils import save_fig


def _rgb(path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1]


def _largest_component(mask_path, thr):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > thr
    props = regionprops(label(m, connectivity=2))
    if not props:
        return m, None, 0
    p = max(props, key=lambda q: q.area)
    return m, p.centroid, int(p.area)


def _window(center, size, h, w):
    size = min(size, h, w)
    y0 = int(np.clip(round(center[0] - size / 2), 0, h - size))
    x0 = int(np.clip(round(center[1] - size / 2), 0, w - size))
    return y0, y0 + size, x0, x0 + size


def _grid(samples: dict, n_cols, panel, title, path):
    names = list(samples)
    n_rows = max(int(np.ceil(len(s) / n_cols)) for s in samples.values())
    fig, axes = plt.subplots(n_rows, n_cols * len(names), figsize=(1.9 * n_cols * len(names), 1.9 * n_rows + 0.8),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for b, name in enumerate(names):
        for i, row in enumerate(samples[name].itertuples(index=False)):
            panel(axes[i // n_cols, b * n_cols + i % n_cols], row)
        left = axes[0, b * n_cols].get_position().x0
        right = axes[0, b * n_cols + n_cols - 1].get_position().x1
        fig.text((left + right) / 2, 0.985, f"{name} (n={len(samples[name])})", ha="center", fontsize=11)
    fig.suptitle(title, y=1.05, fontsize=9, color="0.4")
    return save_fig(fig, path)


def run(idx: pd.DataFrame, cfg: dict, out_dir):
    gc = cfg["gallery"]
    seed = cfg["general"]["seed"]
    thr = cfg["masks"]["threshold"]
    picks = {
        gc["defect_class"]: idx[(idx["group"] == "test_def") & (idx["label"] == gc["defect_class"])],
        "polvere": idx[idx["group"] == "test_dust"],
    }
    samples = {}
    for name, d in picks.items():
        if d.empty:
            print(f"[galleria] nessuna immagine per '{name}': saltata")
            continue
        n = gc["n_per_class"]
        d = d.sample(n=n, random_state=seed) if len(d) > n else d          # seme fisso: campione riproducibile
        samples[name] = d.sort_values("filename").reset_index(drop=True)
    if not samples:
        return
    info = {}
    for d in samples.values():
        for r in d.itertuples(index=False):
            info[r.filename] = _largest_component(r.mask_path, thr) if r.has_mask else (None, None, 0)

    def patch(ax, row):
        ax.imshow(_rgb(row.filename))
        m = info[row.filename][0]
        if m is not None and m.any():
            ax.contour(m, levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title(Path(row.filename).stem[-22:], fontsize=6)
        ax.set_axis_off()

    def zoom(ax, row):
        m, c, area = info[row.filename]
        if c is None:
            ax.text(0.5, 0.5, "senza maschera", ha="center", va="center", fontsize=7, color="0.5")
            ax.set_axis_off()
            return
        img = _rgb(row.filename)
        y0, y1, x0, x1 = _window(c, gc["crop_px"], *img.shape[:2])
        ax.imshow(img[y0:y1, x0:x1])
        ax.contour(m[y0:y1, x0:x1], levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title(f"{area} px", fontsize=6)
        ax.set_axis_off()

    _grid(samples, gc["n_cols"], patch, f"Campione casuale (seme {seed}); rosso = maschera",
          f"{out_dir}/figures/gallery_patches.png")
    if any(v[1] is not None for v in info.values()):
        _grid(samples, gc["n_cols"], zoom, f"Ritaglio di {gc['crop_px']} px sulla componente più grande",
              f"{out_dir}/figures/gallery_zoom.png")
    pd.concat([d.assign(gallery_class=n) for n, d in samples.items()])[
        ["gallery_class", "filename", "split"]].to_csv(f"{out_dir}/gallery_samples.csv", index=False)
