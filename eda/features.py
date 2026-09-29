"""
features.py
===========

Tutto ciò che si calcola leggendo le immagini:

* ``scan_images``: statistiche di colore, istogramma di luminosità, hash percettivo
  e hash delle strisce di bordo (per il leakage);
* ``compute_embeddings``: un vettore per immagine, per PCA, classificatore di
  dominio e kNN. Backbone ``simple`` (statistiche classiche, nessuna dipendenza)
  oppure ``resnet18`` (torchvision).
"""

import hashlib

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

STRIPS = ("left", "right", "top", "bottom")


# --------------------------------------------------------------------- scansione

def _strips(gray: np.ndarray, o: int) -> dict:
    """Strisce di bordo di o pixel. left/right: H×o; top/bottom: o×W."""
    return {"left": gray[:, :o], "right": gray[:, -o:], "top": gray[:o, :], "bottom": gray[-o:, :]}


def _strip_hash(strip: np.ndarray) -> np.uint64:
    """Hash dei valori grigi grezzi: due strisce hanno lo stesso hash solo se i pixel sono identici."""
    d = hashlib.blake2b(np.ascontiguousarray(strip).tobytes(), digest_size=8).digest()
    return np.uint64(int.from_bytes(d, "little"))


def phash64(gray: np.ndarray) -> np.uint64:
    """Hash percettivo a 64 bit: segno dei 64 coefficienti DCT a bassa frequenza rispetto alla mediana."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    block = cv2.dct(small)[:8, :8].ravel()
    bits = block > np.median(block[1:])            # il termine DC è escluso dalla mediana
    return np.frombuffer(np.packbits(bits).tobytes(), dtype=">u8")[0].astype(np.uint64)


def scan_images(paths, overlap_px=32, hist_bins=64) -> dict:
    """Una lettura di tutte le immagini. Ritorna un dizionario di array (una riga per immagine)."""
    n = len(paths)
    out = {
        "rgb_mean": np.zeros((n, 3), np.float32), "rgb_std": np.zeros((n, 3), np.float32),
        "gray_mean": np.zeros(n, np.float32), "gray_std": np.zeros(n, np.float32),
        "gray_hist": np.zeros((n, hist_bins), np.float32),
        "phash": np.zeros(n, np.uint64),
        "strip_hash": {s: np.zeros(n, np.uint64) for s in STRIPS},
        "strip_std": {s: np.zeros(n, np.float32) for s in STRIPS},
    }
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = bgr[:, :, ::-1].reshape(-1, 3)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        out["rgb_mean"][i], out["rgb_std"][i] = rgb.mean(0), rgb.std(0)
        out["gray_mean"][i], out["gray_std"][i] = gray.mean(), gray.std()
        counts, _ = np.histogram(gray, bins=hist_bins, range=(0, 256))
        out["gray_hist"][i] = counts / counts.sum()
        out["phash"][i] = phash64(gray)
        for name, s in _strips(gray, overlap_px).items():
            out["strip_hash"][name][i] = _strip_hash(s)
            out["strip_std"][name][i] = s.std()
    return out


# --------------------------------------------------------------------- embedding

def _simple_embedding(path: str, size: int = 256) -> np.ndarray:
    """34 numeri: momenti di colore, istogramma di grigio, LBP e gradiente.

    Non è un backbone profondo: serve a far girare l'EDA ovunque. Per il risultato
    da riportare in tesi usa lo stesso backbone dei modelli (knn.backbone).
    """
    bgr = cv2.resize(cv2.imread(str(path), cv2.IMREAD_COLOR), (size, size), interpolation=cv2.INTER_AREA)
    rgb = bgr[:, :, ::-1].reshape(-1, 3).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, bins=16, range=(0, 256))
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    return np.concatenate([rgb.mean(0), rgb.std(0), hist / hist.sum(), lbp_hist / lbp_hist.sum(),
                           [mag.mean(), mag.std()]]).astype(np.float32)


def _resnet_embedding(paths, name="resnet18", size=256, batch=32) -> np.ndarray:
    """layer2 + layer3 di una ResNet ImageNet, con pooling globale medio e massimo (come PatchCore)."""
    import torch
    from torchvision import models

    net = models.get_model(name, weights=models.get_model_weights(name).DEFAULT).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = [cv2.resize(cv2.imread(str(p))[:, :, ::-1], (size, size), interpolation=cv2.INTER_AREA)
                    for p in paths[i:i + batch]]
            x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div(255)
            x = (x - mean) / std
            x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
            f2 = net.layer2(net.layer1(x))
            f3 = net.layer3(f2)
            feats.append(torch.cat([f.mean((2, 3)) for f in (f2, f3)] +
                                   [f.amax((2, 3)) for f in (f2, f3)], dim=1).numpy())
    return np.concatenate(feats).astype(np.float32)


def compute_embeddings(paths, backbone: str = "simple", size: int = 256) -> np.ndarray:
    paths = [str(p) for p in paths]
    if backbone == "simple":
        return np.stack([_simple_embedding(p) for p in paths])
    if backbone.startswith("resnet"):
        return _resnet_embedding(paths, backbone, size)
    raise ValueError(f"Backbone sconosciuto: {backbone} (simple | resnet18)")
