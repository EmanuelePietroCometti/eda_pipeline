# EDA pipeline for Textile Visual Anomaly Detection

Exploratory Data Analysis pipeline for quality control of woven fabrics. It
extracts **handcrafted colour and texture descriptors** from image patches,
runs **statistical and dimensionality-reduction analyses**, and produces
**publication-ready figures** for a master's thesis on Visual Anomaly Detection.

The pipeline is intentionally built on classical, fully interpretable Computer
Vision features (`scikit-image`, `OpenCV`, `NumPy`, `pandas`) rather than a deep
backbone, so that every axis of every plot maps to a physically meaningful
quantity that can be discussed in the thesis.

---

## 1. Dataset layout

The dataset root is expected to follow this structure (configurable in
`config.yaml`):

```
dataset_root/
  good/
    good/      # nominal fabric (no defects)                -> label "Good"
    dust/      # nominal fabric with ambient dust/dirt      -> label "Dust"
  reject/
    macchie/   # chromatic alterations on light fabrics     -> label "Macchie"
    paglie/    # foreign structural inclusions between yarns -> label "Paglie"
    nodi/      # yarn knots / slubs                          -> label "Nodi"
    rotture/   # weave tears                                 -> label "Rotture"
```

Splitting `good/` into `good/` and `dust/` is what allows the analysis to isolate
dusty-but-nominal samples and quantify their ambiguity against real structural
defects such as *paglie*.

---

## 2. Pipeline stages

### 2.1 Feature extraction — `src/feature_extraction.py`
Every image is centre-cropped to a `512 x 512` patch (spatial frequency of the
weave is preserved instead of being blurred by a resize) and described by:

* **Chromatic features** — mean, variance, skewness and kurtosis of each channel
  in **HSV** and **CIE-Lab**. These isolate colour anomalies (*macchie*).
* **GLCM** (Gray-Level Co-occurrence Matrix) — Contrast, Homogeneity, Energy and
  Correlation, averaged over multiple distances/angles for rotation robustness.
* **LBP** (Local Binary Patterns) — the normalised uniform-pattern histogram
  plus the local LBP variance, quantifying the micro-texture of the weave.

The result is a tidy `pandas.DataFrame` (`outputs/features.csv`).

### 2.2 Statistical analysis & dimensionality reduction — `src/eda.py`
* **PCA** and **t-SNE** projections to 2D (features are standardised first).
* **Distribution-overlap metrics** — Bhattacharyya coefficient/distance and the
  Overlapping Coefficient (OVL) between **Paglie** and dusty **Good** on the
  structural features, to numerically express their visual ambiguity.
* **Significance tests** — one-way **ANOVA** and non-parametric
  **Kruskal-Wallis**, with eta² / epsilon² effect sizes, comparing **Good**
  against **Rotture** and **Nodi** on the structural features.

### 2.3 Visualisation — `src/visualizzation.py`
300-DPI PNGs saved to `outputs/figures/`:
* **Multivariate violin plots** — classes on X, discriminative metrics on Y
  (e.g. HSV saturation variance, GLCM contrast).
* **t-SNE scatter** — coloured by class, showing how *Good* spans a broad region
  and how dust / structural defects overlap it in the latent space.
* **Overlapping density histograms** — GLCM Energy/Contrast for dusty *Good* vs
  *Paglie*, visually demonstrating their ambiguity.

---

## 3. Installation & usage

Dependencies are declared in `pyproject.toml` (Poetry) and `requirements.txt`.

```bash
poetry install          # or: pip install -r requirements.txt
```

Set `general_configuration.dataset_root` in `config.yaml`, then run:

```bash
python main.py
```

CLI overrides are available:

```bash
python main.py --root /path/to/dataset_root --output results
```

All tables (`features.csv`, `overlap_paglie_vs_dust.csv`,
`significance_tests.csv`) and every figure are written under the configured
`output_dir` (default `outputs/`).

---

## 4. Configuration reference (`config.yaml`)

* `general_configuration`
  * `dataset_root` — root folder following the layout above.
  * `patch_size` — crop size in pixels (default `512`).
  * `valid_extensions` — image suffixes to read.
  * `output_dir`, `random_state`.
* `feature_extraction.glcm` — `distances`, `angles_deg`, `levels` (grey-level
  quantisation of the co-occurrence matrix).
* `feature_extraction.lbp` — `radius`, `n_points`, `method`.
