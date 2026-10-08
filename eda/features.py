"""
features.py
===========

Un vettore (embedding) per immagine, usato per distanze, PCA, t-SNE e UMAP.

* backbone ``simple``: statistiche classiche (colore, LBP, gradiente), nessuna dipendenza;
* backbone ``resnet18``: layer2 + layer3 di una ResNet ImageNet con pooling medio e massimo
  (richiede torch e torchvision).

Le versioni V0, V1, V2 condividono quasi tutte le immagini: ``embed_versions`` calcola
l'embedding una sola volta per contenuto (hash dei pixel) e lo salva in una cache su disco.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler


def _simple_embedding(path: str, size: int = 256) -> np.ndarray:
    """34 numeri: momenti di colore, istogramma di grigio, LBP e gradiente.

    Non è un backbone profondo: serve a far girare l'EDA ovunque. Per il risultato
    da riportare in tesi usa lo stesso backbone dei modelli (embedding.backbone).
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


def standardize(emb: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Media e deviazione standard si imparano SOLO sui buoni di riferimento (`train_good`);
    tutte le altre immagini vengono solo trasformate, così non influenzano la scala."""
    return StandardScaler().fit(emb[ref_mask]).transform(emb)


def embed_versions(idx_by_version: dict, backbone: str, size: int, cache=None) -> dict:
    """Embedding di tutte le versioni, calcolati una sola volta per contenuto (colonna `sha`).

    Ritorna ``{versione: array (n_immagini, n_feature)}`` con le righe nello stesso ordine
    dell'indice di quella versione. Con `cache` (file .npz) i vettori già calcolati non si
    ricalcolano: cancella il file se cambi backbone, pesi o codice dell'embedding.
    """
    cache = Path(cache) if cache else None
    stored = {}
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        stored = dict(zip(z["sha"].tolist(), z["emb"]))

    unique = pd.concat([i[["sha", "filename"]] for i in idx_by_version.values()]).drop_duplicates("sha")
    todo = unique[~unique["sha"].isin(stored)]
    if len(todo):
        print(f"[embedding] {backbone}: {len(todo)} immagini da calcolare ({len(unique) - len(todo)} in cache)")
        X = compute_embeddings(todo["filename"].tolist(), backbone, size)
        stored.update(zip(todo["sha"].tolist(), X))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, sha=np.array(list(stored)), emb=np.stack(list(stored.values())))
    return {v: np.stack([stored[s] for s in i["sha"]]) for v, i in idx_by_version.items()}
