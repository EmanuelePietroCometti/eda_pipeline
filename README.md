# EDA pipeline for Textile Visual Anomaly Detection

Exploratory Data Analysis pipeline for quality control of woven fabrics
(thesis chapter 4). It extracts **handcrafted colour and texture descriptors**
from image patches and studies them with **PCA** and **t-SNE** only. The
analysis is purely descriptive: no hypothesis test is run.

The pipeline is intentionally built on classical, fully interpretable Computer
Vision features (`scikit-image`, `OpenCV`, `NumPy`, `pandas`) rather than a deep
backbone, so that every axis of every plot maps to a physically meaningful
quantity that can be discussed in the thesis.

---

## 1. Dataset layout

One sub-folder per fabric (configurable in `config.yaml`):

```
dataset_root/
  nero/
    good/
      good/      # nominal fabric (no defects)                -> label "Good"
      dust/      # nominal fabric with ambient dust/dirt      -> label "Dust"
    reject/
      paglie/    # foreign structural inclusions between yarns -> label "Paglie"
      nodi/      # yarn knots / slubs                          -> label "Nodi"
      ...
  chiaro/
    ...
  quadrettoni/
    ...
```

* The **fabric** is the first folder below `dataset_root`.
* The **class** is the folder that directly contains the image, mapped to a
  label by `classes.folder_to_label`. Several folders can map to the same label
  (e.g. `grappola: "Nodi"`). Folders not listed are skipped with a warning.
* The legacy layout without the fabric level (`dataset_root/good/...`,
  `dataset_root/reject/...`) still works: all images get `default_fabric`.

---

## 2. What the pipeline produces (one folder per section of chapter 4)

| Output folder | Chapter section | Content |
|---|---|---|
| `features.csv`, `conteggi_tessuto_classe.csv` | 4.1 | feature matrix; samples per fabric and class |
| `cap4_2_tessuti/` | 4.2 Differences between fabrics | PCA on **Good only**, coloured by fabric: scree plot, PC1–PC2 and PC1–PC3 scatter, loadings, share of colour vs GLCM vs LBP per component, feature distributions per fabric |
| `cap4_3_difetti/<fabric>/` | 4.3 Defects vs sound fabric | per fabric: PCA (scree, scatter, loadings, family share), t-SNE, t-SNE stability grid (appendix), per-sample coordinates |
| `cap4_4_polvere/<fabric>/` | 4.4 Dust | only fabrics with dust: the same projections of 4.3 with Good, Dust and the reference defect coloured and everything else grey, feature distributions per class |
| `run_info.json` | — | configuration, library versions, features used, excluded features |

Every figure reports the number of samples per group. Classes with fewer than
`analysis.distributions.min_n_violin` samples are drawn as individual points,
never as a violin (a density estimated on a handful of points is an artefact).

---

## 3. Features — `src/feature_extraction.py`

Every image is centre-cropped to a `512 x 512` patch and described by:

* **Colour** — mean, variance, skewness and kurtosis of HSV **S, V** and CIE-Lab
  **L\*, a\*, b\***. The hue channel is excluded: it is an angle (179 and 0 are
  neighbours) and is mostly noise on black and white fabrics.
* **GLCM** — Contrast, Homogeneity, Energy, Correlation, averaged over the
  configured distances and angles.
* **LBP** — normalised uniform-pattern histogram (`lbp_hist_XX`) and
  `lbp_code_dispersion`, the variance of the LBP-coded image. This is *not* the
  local-contrast operator VAR of Ojala et al. (2002).

All descriptors are computed on the whole patch, so a small defect shifts them
only slightly: keep this limitation in mind when reading the projections.

---

## 4. Analysis — `src/eda.py`

* **PCA** on z-scored features (fitted separately for each analysis). Explained
  variance of up to `analysis.pca.n_components` components, loadings, and the
  share of each component due to each feature family (sum of squared loadings;
  the number of features per family is reported because the 27 LBP features
  can win a large share partly by number).
* **t-SNE** after a PCA pre-reduction to 30 components. The figure in the
  chapter uses a PCA initialisation; the stability grid recomputes it with a
  random initialisation for every perplexity x seed. Distances and cluster
  sizes in t-SNE are not interpretable: describe only structures that appear
  in every panel of the grid.

---

## 5. Installation & usage

```bash
poetry install          # or: pip install -r requirements.txt
python main.py                                  # paths from config.yaml
python main.py --root /path/to/dataset_root --output results
python main.py --features outputs/features.csv  # re-plot without re-extracting
```

---

## 6. Configuration reference (`config.yaml`)

* `general_configuration` — `dataset_root`, `patch_size`, `valid_extensions`,
  `output_dir`, `random_state`, `default_fabric`, `fabric_order`.
* `classes` — `folder_to_label`, `order`, `nominal`, `good_label`,
  `dust_label`, `dust_reference_defect`.
* `feature_extraction.glcm` — `distances`, `angles_deg`, `levels`.
* `feature_extraction.lbp` — `radius`, `n_points`, `method`.
* `analysis.pca` — `n_components`, `plot_pairs`, `loadings_top_k`,
  `loadings_components`.
* `analysis.tsne` — `perplexity`, `seed`, `pre_pca_dims`,
  `stability_perplexities`, `stability_seeds`.
* `analysis.distributions` — `features` (fix them **before** looking at the
  results), `min_n_violin`.
