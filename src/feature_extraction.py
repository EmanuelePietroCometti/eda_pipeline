"""
feature_extraction.py
=====================

Handcrafted feature extraction for textile Visual Anomaly Detection.

The module reads image patches (nominally 512x512) organised by defect class and
computes three complementary families of descriptors, storing everything in a
single tidy :class:`pandas.DataFrame`:

A. **Chromatic features** (HSV + CIE-Lab colour spaces): per-channel mean,
   variance, skewness and kurtosis. These isolate colour alterations such as the
   *macchie* (stains) on light fabrics.
B. **Structural / texture features**:
   * **GLCM** (Gray-Level Co-occurrence Matrix) -> Contrast, Homogeneity,
     Energy, Correlation. Captures the periodic warp/weft pattern of the weave.
   * **LBP** (Local Binary Patterns) -> normalised histogram + local variance,
     quantifying the micro-texture.

Only ``scikit-image``, ``cv2``, ``numpy`` and ``pandas`` are used, as required.
Paths are handled exclusively through :mod:`pathlib`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
# The dataset is expected to be laid out as:
#
#   dataset_root/
#     good/
#       good/     -> nominal fabric                       (label "Good")
#       dust/     -> nominal fabric with ambient dust     (label "Dust")
#     reject/
#       macchie/  -> chromatic alterations                (label "Macchie")
#       paglie/   -> foreign structural inclusions        (label "Paglie")
#       nodi/     -> yarn knots / slubs                   (label "Nodi")
#       rotture/  -> weave tears                           (label "Rotture")
#
# The "good" branch is deliberately split into two sub-folders so that dusty
# (but still nominal) samples can be analysed separately from the pristine ones.

# Mapping between the on-disk sub-folder name and the canonical class label used
# throughout the pipeline. Keys are lower-cased folder names.
_SUBFOLDER_TO_LABEL: Dict[str, str] = {
    "good": "Good",
    "dust": "Dust",
    "macchie": "Macchie",
    "paglie": "Paglie",
    "nodi": "Nodi",
    "rotture": "Rotture",
}

# Coarse grouping (nominal vs defective) derived from the fine label.
_NOMINAL_LABELS = {"Good", "Dust"}


def discover_images(
    dataset_root: Path,
    valid_extensions: Tuple[str, ...],
) -> pd.DataFrame:
    """Walk ``dataset_root`` and build an index of ``(path, label, group)``.

    Parameters
    ----------
    dataset_root : pathlib.Path
        Root directory following the ``good/{good,dust}`` and
        ``reject/{macchie,paglie,nodi,rotture}`` layout.
    valid_extensions : tuple of str
        Accepted file suffixes (compared case-insensitively, e.g. ``".bmp"``).

    Returns
    -------
    pandas.DataFrame
        Columns: ``filename`` (str), ``label`` (fine class),
        ``group`` (``"Good"``/``"Reject"``).
    """
    dataset_root = Path(dataset_root)
    exts = {e.lower() for e in valid_extensions}
    records: List[dict] = []

    # ``rglob("*")`` traverses the whole tree; the parent folder name determines
    # the label, which makes the discovery robust to an extra nesting level.
    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in exts:
            continue

        folder_name = file_path.parent.name.lower()
        label = _SUBFOLDER_TO_LABEL.get(folder_name)
        if label is None:
            # Image sitting in an unrecognised folder: skip but warn once.
            continue

        group = "Good" if label in _NOMINAL_LABELS else "Reject"
        records.append(
            {"filename": str(file_path), "label": label, "group": group}
        )

    if not records:
        raise FileNotFoundError(
            f"No valid images found under '{dataset_root}'. "
            f"Checked extensions: {sorted(exts)}."
        )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Patch loading
# --------------------------------------------------------------------------- #
def load_patch(path: Path, patch_size: int = 512) -> np.ndarray:
    """Load an image and return a ``patch_size x patch_size`` RGB ``uint8`` array.

    A **centre crop** is preferred over a plain resize because it preserves the
    native spatial frequency of the weave (critical for GLCM/LBP): rescaling
    would blur the warp/weft period. If the image is smaller than the requested
    patch it is up-scaled first so that a full crop is always available.
    """
    path = Path(path)
    # cv2.imread copes with BMP/PNG/TIFF; it returns BGR, hence the conversion.
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Unable to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    # Up-scale the shorter side so a centre crop of ``patch_size`` always fits.
    if h < patch_size or w < patch_size:
        scale = patch_size / min(h, w)
        rgb = cv2.resize(
            rgb, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        h, w = rgb.shape[:2]

    top = (h - patch_size) // 2
    left = (w - patch_size) // 2
    return rgb[top: top + patch_size, left: left + patch_size]


# --------------------------------------------------------------------------- #
# A. Chromatic features
# --------------------------------------------------------------------------- #
def _moment_stats(channel: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute the first four statistical moments of a single image channel.

    Given the pixel population :math:`x`:

    * mean     :math:`\\mu = \\mathbb{E}[x]`
    * variance :math:`\\sigma^2 = \\mathbb{E}[(x-\\mu)^2]`
    * skewness :math:`\\gamma_1 = \\mathbb{E}[(x-\\mu)^3]/\\sigma^3`
      (asymmetry of the distribution)
    * kurtosis :math:`\\gamma_2 = \\mathbb{E}[(x-\\mu)^4]/\\sigma^4 - 3`
      (excess/Fisher kurtosis, tailedness)

    A stain shifts the colour distribution and inflates variance/skewness, which
    is exactly what these moments are meant to capture.
    """
    x = channel.astype(np.float64).ravel()
    mean = float(np.mean(x))
    var = float(np.var(x))
    # For a (nearly) constant channel skew/kurtosis are undefined; guard to 0.
    if var < 1e-12:
        skew = kurt = 0.0
    else:
        skew = float(stats.skew(x))
        kurt = float(stats.kurtosis(x))  # Fisher definition -> 0 for a Gaussian
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_var": var,
        f"{prefix}_skew": skew,
        f"{prefix}_kurt": kurt,
    }


def extract_color_features(rgb: np.ndarray) -> Dict[str, float]:
    """Extract per-channel moments in the HSV and CIE-Lab colour spaces.

    HSV decouples chromaticity (H, S) from luminance (V); Lab is
    perceptually-uniform with an explicit luminance (L) and two opponent-colour
    axes (a: green-red, b: blue-yellow). Together they give a rich, largely
    redundant-free description of colour anomalies.

    Note on ranges (OpenCV, 8-bit): H in [0, 179], S/V in [0, 255];
    L/a/b in [0, 255]. Absolute scale is irrelevant downstream because features
    are standardised before the statistical analysis.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    feats: Dict[str, float] = {}
    for idx, name in enumerate(("h", "s", "v")):
        feats.update(_moment_stats(hsv[:, :, idx], f"hsv_{name}"))
    for idx, name in enumerate(("l", "a", "b")):
        feats.update(_moment_stats(lab[:, :, idx], f"lab_{name}"))
    return feats


# --------------------------------------------------------------------------- #
# B. Structural / texture features
# --------------------------------------------------------------------------- #
def extract_glcm_features(
    gray: np.ndarray,
    distances: List[int],
    angles_deg: List[int],
    levels: int = 256,
) -> Dict[str, float]:
    """Compute Haralick GLCM descriptors on a grayscale patch.

    The GLCM :math:`P` counts how often a pixel of intensity :math:`i` occurs at
    a given offset :math:`(d, \\theta)` from a pixel of intensity :math:`j`.
    From the normalised matrix we derive:

    * **Contrast**    :math:`\\sum_{i,j} (i-j)^2 P_{ij}` — local intensity
      variation; a tear (*rottura*) raises it sharply.
    * **Homogeneity** :math:`\\sum_{i,j} P_{ij}/(1+(i-j)^2)` — closeness of the
      distribution to the diagonal (smoothness).
    * **Energy**      :math:`\\sqrt{\\sum_{i,j} P_{ij}^2}` (Angular Second
      Moment) — textural uniformity of the periodic weave.
    * **Correlation** linear dependency of grey levels along the offset.

    The matrix is computed for every ``(distance, angle)`` pair and each property
    is **averaged over all offsets** to obtain a rotation/scale-robust scalar.
    """
    # Quantise to ``levels`` grey levels to keep the co-occurrence matrix compact
    # and statistically well-populated (fewer, denser bins -> less noisy GLCM).
    if levels < 256:
        gray_q = (gray.astype(np.uint16) * levels // 256).astype(np.uint8)
    else:
        gray_q = gray

    angles_rad = [np.deg2rad(a) for a in angles_deg]
    glcm = graycomatrix(
        gray_q,
        distances=distances,
        angles=angles_rad,
        levels=levels,
        symmetric=True,   # treat (i,j) and (j,i) as equivalent -> direction-agnostic
        normed=True,      # turn counts into a probability distribution
    )

    feats: Dict[str, float] = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        # graycoprops returns a (n_distances, n_angles) matrix; average it out.
        feats[f"glcm_{prop}"] = float(np.mean(graycoprops(glcm, prop)))
    return feats


def extract_lbp_features(
    gray: np.ndarray,
    n_points: int = 24,
    radius: int = 3,
    method: str = "uniform",
) -> Dict[str, float]:
    """Compute the Local Binary Pattern histogram and local variance.

    For every pixel, LBP thresholds the ``n_points`` neighbours sampled on a
    circle of the given ``radius`` against the centre and encodes the result as a
    binary number. The ``"uniform"`` variant collapses rotationally-equivalent
    patterns, yielding ``n_points + 2`` bins (one per uniform pattern plus a
    single non-uniform bin).

    We store:

    * the **normalised histogram** (a probability distribution over the LBP
      codes) — this is the primary micro-texture signature;
    * the **variance of the LBP-coded image** (``lbp_var``), a compact scalar
      summarising the spread of local patterns.

    Foreign inclusions (*paglie*) and knots (*nodi*) perturb the regular weave
    and therefore shift mass across the LBP bins.
    """
    lbp = local_binary_pattern(gray, P=n_points, R=radius, method=method)

    n_bins = n_points + 2  # valid for the "uniform" method
    hist, _ = np.histogram(
        lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
    )

    feats: Dict[str, float] = {
        f"lbp_hist_{i:02d}": float(hist[i]) for i in range(n_bins)
    }
    feats["lbp_var"] = float(np.var(lbp))
    return feats


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def extract_features_from_patch(
    rgb: np.ndarray,
    glcm_cfg: dict,
    lbp_cfg: dict,
) -> Dict[str, float]:
    """Run the full descriptor stack on a single RGB patch."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    feats: Dict[str, float] = {}
    feats.update(extract_color_features(rgb))
    feats.update(
        extract_glcm_features(
            gray,
            distances=glcm_cfg.get("distances", [1]),
            angles_deg=glcm_cfg.get("angles_deg", [0, 45, 90, 135]),
            levels=glcm_cfg.get("levels", 256),
        )
    )
    feats.update(
        extract_lbp_features(
            gray,
            n_points=lbp_cfg.get("n_points", 24),
            radius=lbp_cfg.get("radius", 3),
            method=lbp_cfg.get("method", "uniform"),
        )
    )
    return feats


def extract_features(config: dict) -> pd.DataFrame:
    """Extract handcrafted features for every image described by ``config``.

    Parameters
    ----------
    config : dict
        Parsed ``config.yaml``. Relevant keys:
        ``general_configuration.dataset_root``, ``.patch_size``,
        ``.valid_extensions`` and the ``feature_extraction.{glcm,lbp}`` blocks.

    Returns
    -------
    pandas.DataFrame
        One row per image: metadata columns (``filename``, ``label``,
        ``group``) followed by all numeric feature columns.
    """
    gen = config.get("general_configuration", {})
    fx = config.get("feature_extraction", {})

    dataset_root = Path(gen["dataset_root"])
    patch_size = int(gen.get("patch_size", 512))
    valid_extensions = tuple(gen.get("valid_extensions", [".bmp", ".BMP"]))
    glcm_cfg = fx.get("glcm", {})
    lbp_cfg = fx.get("lbp", {})

    index = discover_images(dataset_root, valid_extensions)
    print(
        f"Discovered {len(index)} images across "
        f"{index['label'].nunique()} classes: "
        f"{index['label'].value_counts().to_dict()}"
    )

    rows: List[dict] = []
    for row in tqdm(
        index.itertuples(index=False),
        total=len(index),
        desc="Extracting handcrafted features",
    ):
        try:
            patch = load_patch(row.filename, patch_size)
            feats = extract_features_from_patch(patch, glcm_cfg, lbp_cfg)
        except (IOError, ValueError) as exc:  # corrupted / unreadable file
            print(f"Warning: skipping '{row.filename}': {exc}")
            continue

        rows.append(
            {
                "filename": row.filename,
                "label": row.label,
                "group": row.group,
                **feats,
            }
        )

    df = pd.DataFrame(rows)
    print(f"Feature matrix: {df.shape[0]} samples x "
          f"{df.shape[1] - 3} numeric features.")
    return df
