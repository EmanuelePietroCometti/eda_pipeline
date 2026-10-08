"""
dataset.py
==========

Indice del dataset (una riga per immagine), identificazione della polvere per
differenza tra versioni e gruppi (vedi README).

Layout atteso di ogni versione::

    root/train/{good, <difetto>, ...}
    root/test/{good, <difetto>, ...}
    root/ground_truth/{good, <difetto>, ...}     # una maschera per immagine
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _hash_array(a: np.ndarray) -> str:
    """Hash del CONTENUTO dei pixel (non del file): resta uguale se il file viene ricodificato."""
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=8).hexdigest()


def build_index(root, cfg) -> pd.DataFrame:
    """Una riga per immagine: split, cartella, maschera, hash del contenuto.

    Nota su pandas 3: un valore mancante in una colonna di stringhe diventa NaN,
    non None. Per questo la colonna ``mask_path`` usa la stringa vuota ""
    e ``has_mask`` è un booleano: nessun ``is None`` da controllare.
    """
    root = Path(str(root).replace("\\", "/"))   # i percorsi del config possono avere i backslash di Windows
    dcfg = cfg["dataset"]
    exts = {e.lower() for e in dcfg["extensions"]}
    suffix = dcfg.get("mask_suffix", "")
    good = dcfg["good_folder"]

    # (cartella, stem) -> percorso della maschera
    masks = {}
    gt = root / "ground_truth"
    if gt.is_dir():
        for p in gt.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                stem = p.stem[: -len(suffix)] if suffix and p.stem.endswith(suffix) else p.stem
                masks[(p.parent.name, stem)] = p

    rows = []
    for split in ("train", "test"):
        for p in sorted((root / split).rglob("*")):
            if not (p.is_file() and p.suffix.lower() in exts):
                continue
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Immagine illeggibile: {p}")
            folder = p.parent.name
            m = masks.get((folder, p.stem))
            m_hash = ""
            if m is not None:
                arr = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
                m_hash = "" if arr is None else _hash_array(arr)
            rows.append({
                "filename": str(p), "split": split, "folder": folder, "label": folder,
                "is_defect": folder != good,
                "has_mask": m is not None, "mask_path": str(m) if m is not None else "",
                "sha": _hash_array(img), "mask_sha": m_hash,
                "height": img.shape[0], "width": img.shape[1],
            })
    if not rows:
        raise FileNotFoundError(f"Nessuna immagine trovata sotto {root}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- polvere e gruppi

def identify_dust(idx_ref: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """Segna come polvere le immagini di `good` presenti in `idx` e assenti nella
    versione di riferimento NELLO STESSO SPLIT (confronto per contenuto).

    Questo funziona perché la polvere ha sostituito parte dei good: le immagini
    con polvere sono nuove, i good rimossi sono spariti.
    """
    known = set(zip(idx_ref["split"], idx_ref["sha"]))
    new = np.array([(s, h) not in known for s, h in zip(idx["split"], idx["sha"])])
    out = idx.copy()
    out["dust"] = new & ~out["is_defect"].to_numpy()
    return out


def assign_groups(idx: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge la colonna `group` con i sei gruppi del README."""
    out = idx.copy()
    out["group"] = np.select(
        [out["is_defect"], out["dust"]],
        [out["split"] + "_def", out["split"] + "_dust"],
        default=out["split"] + "_good",
    )
    return out
