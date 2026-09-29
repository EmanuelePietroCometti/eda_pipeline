"""
dataset.py
==========

Indice del dataset (una riga per immagine), identificazione della polvere per
differenza tra versioni e controlli di coerenza (design.md, sezioni 2 e 3).

Layout atteso di ogni versione::

    root/train/{good, <difetto>, ...}
    root/test/{good, <difetto>, ...}
    root/ground_truth/{good, <difetto>, ...}     # una maschera per immagine
"""

import hashlib
from collections import Counter
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
    root = Path(root)
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
    """Aggiunge la colonna `group` con i sei gruppi di design.md."""
    out = idx.copy()
    out["group"] = np.select(
        [out["is_defect"], out["dust"]],
        [out["split"] + "_def", out["split"] + "_dust"],
        default=out["split"] + "_good",
    )
    return out


# --------------------------------------------------------------------- controlli

def find_duplicates(idx: pd.DataFrame) -> pd.DataFrame:
    """Immagini con pixel identici dentro una versione (una riga per contenuto ripetuto).

    Un duplicato esatto a cavallo di train e test è leakage nella forma più semplice; un
    duplicato dentro `train/good` fa avere distanza 0 al leave-one-out del kNN.
    """
    d = idx[idx.duplicated("sha", keep=False)]
    rows = []
    for sha, g in d.groupby("sha"):
        rows.append({"sha": sha, "n_copie": len(g), "splits": "+".join(sorted(set(g["split"]))),
                     "gruppi": "+".join(sorted(set(g["group"]))), "cartelle": "+".join(sorted(set(g["label"]))),
                     "file": " | ".join(Path(f).name for f in g["filename"])})
    return pd.DataFrame(rows, columns=["sha", "n_copie", "splits", "gruppi", "cartelle", "file"])


def check_masks(idx: pd.DataFrame, threshold: int = 127) -> pd.DataFrame:
    """Controlli di integrità sulle maschere di una versione. Ritorna i problemi trovati."""
    issues = []
    for r in idx.itertuples(index=False):
        if r.is_defect and not r.has_mask:
            issues.append((r.filename, "difetto senza maschera"))
            continue
        if not r.has_mask:
            continue
        m = cv2.imread(r.mask_path, cv2.IMREAD_GRAYSCALE)
        if m.shape != (r.height, r.width):
            issues.append((r.filename, f"maschera {m.shape} diversa dall'immagine"))
        empty = not (m > threshold).any()
        if r.is_defect and empty:
            issues.append((r.filename, "maschera di difetto vuota"))
        if not r.is_defect and not empty:
            issues.append((r.filename, "maschera di good non vuota (etichetta sbagliata?)"))
    return pd.DataFrame(issues, columns=["filename", "problema"])


def compare_versions(idx_ref: pd.DataFrame, idx: pd.DataFrame, dust_in_train: bool) -> pd.DataFrame:
    """Controlli di coerenza tra la versione di riferimento (V0) e un'altra (design.md, sez. 3).

    Ritorna una tabella (controllo, ok, dettaglio). Un controllo fallito non si
    corregge in silenzio: si guarda cosa non torna.
    """
    checks = []

    def add(name, ok, detail=""):
        checks.append({"controllo": name, "ok": bool(ok), "dettaglio": detail})

    # 1. numerosità per split e tipo (good/difetto) uguale
    c0 = idx_ref.groupby(["split", "is_defect"]).size()
    c1 = idx.groupby(["split", "is_defect"]).size()
    add("numerosità per split e tipo uguale", c0.equals(c1), f"V0={c0.to_dict()} altra={c1.to_dict()}")

    key0 = set(zip(idx_ref["split"], idx_ref["sha"]))
    key1 = set(zip(idx["split"], idx["sha"]))
    removed = idx_ref[[(s, h) not in key1 for s, h in zip(idx_ref["split"], idx_ref["sha"])]]
    added = idx[[(s, h) not in key0 for s, h in zip(idx["split"], idx["sha"])]]

    # 2. le immagini rimosse e quelle aggiunte sono solo good
    add("rimosse rispetto a V0: solo good", not removed["is_defect"].any(), f"{len(removed)} rimosse")
    add("aggiunte rispetto a V0: solo good", not added["is_defect"].any(), f"{len(added)} aggiunte")

    # 3. per split, rimosse = aggiunte. Il confronto è tra MULTINSIEMI di (split, hash): se il
    #    dataset contiene immagini con pixel identici (duplicati), un confronto tra insiemi
    #    conta le righe in modo asimmetrico e il controllo fallirebbe senza che manchi nulla.
    m0, m1 = Counter(zip(idx_ref["split"], idx_ref["sha"])), Counter(zip(idx["split"], idx["sha"]))
    rem, add_ = m0 - m1, m1 - m0
    nr = Counter(s for (s, _), n in rem.items() for _ in range(n))
    na = Counter(s for (s, _), n in add_.items() for _ in range(n))
    add("per split: n. rimosse = n. aggiunte (contando i duplicati)", all(nr[s] == na[s] for s in ("train", "test")),
        f"rimosse={dict(nr)} aggiunte={dict(na)}")

    # 4. se la polvere non è nel train, il train è identico a V0
    if not dust_in_train:
        t0 = set(idx_ref.loc[idx_ref["split"] == "train", "sha"])
        t1 = set(idx.loc[idx["split"] == "train", "sha"])
        add("train identico a V0 (polvere non nel train)", t0 == t1)

    # 5. difetti e maschere identici tra le versioni
    d0 = set(zip(idx_ref.loc[idx_ref["is_defect"], "split"], idx_ref.loc[idx_ref["is_defect"], "label"],
                 idx_ref.loc[idx_ref["is_defect"], "sha"], idx_ref.loc[idx_ref["is_defect"], "mask_sha"]))
    d1 = set(zip(idx.loc[idx["is_defect"], "split"], idx.loc[idx["is_defect"], "label"],
                 idx.loc[idx["is_defect"], "sha"], idx.loc[idx["is_defect"], "mask_sha"]))
    add("difetti e maschere identici a V0", d0 == d1, f"solo in V0: {len(d0 - d1)}, solo nell'altra: {len(d1 - d0)}")

    # 6. le aggiunte non sono duplicati di immagini di V0 spostate di split
    moved = added[added["sha"].isin(set(idx_ref["sha"]))]
    add("nessuna aggiunta è un'immagine di V0 spostata di split", moved.empty, f"{len(moved)} spostate")
    return pd.DataFrame(checks)
