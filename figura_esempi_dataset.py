#!/usr/bin/env python3
"""Crea la figura «una patch good e una per classe di difetto, per ogni tessuto».

Layout atteso (MVTec): <cartella_tessuto>/{train,test,ground_truth}/<classe>/<immagine>
Le maschere stanno in ground_truth/<classe>/ con lo stesso nome dell'immagine,
oppure con suffisso _mask (qualunque estensione).

Uso:
  python3 figura_esempi_dataset.py \
      --tessuto "Nero=/dati/tessuto_nero" \
      --tessuto "Bianco/chiaro=/dati/tessuto_chiaro" \
      --tessuto "Quadrettoni=/dati/tessuto_quadrettoni" \
      --split test --seed 42 --out ds_esempi.png

Opzioni utili: --senza-maschera (solo immagini), --larghezza-cm 15.5, --dpi 300.
Dipendenze: numpy, matplotlib, pillow.
"""
import argparse, random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# colonna -> nomi di cartella accettati (minuscolo)
COLONNE = [("Good", ["good"]),
           ("Nodo", ["nodo"]),
           ("Grappola", ["grappola"]),
           ("Paglia", ["paglia"]),
           ("Macchia", ["macchia", "macchie"])]
EST = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def cartella(split_dir, alias):
    if not split_dir.is_dir():
        return None
    for d in sorted(split_dir.iterdir()):
        if d.is_dir() and d.name.lower() in alias:
            return d
    return None

def immagini(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EST)

def trova_maschera(gt_dir, img):
    if gt_dir is None:
        return None
    for nome in (img.stem, img.stem + "_mask"):
        for p in gt_dir.glob(nome + ".*"):
            if p.suffix.lower() in EST:
                return p
    return None

def contorno(m, spessore=4):
    m = m.astype(bool)
    er = m.copy()
    for _ in range(spessore):
        s = er.copy()
        s[1:, :] &= er[:-1, :]; s[:-1, :] &= er[1:, :]
        s[:, 1:] &= er[:, :-1]; s[:, :-1] &= er[:, 1:]
        er = s
    return m & ~er

def scegli(d, rng):
    return rng.choice(immagini(d))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tessuto", action="append", required=True, help='"Etichetta=percorso"')
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ds_esempi.png")
    ap.add_argument("--senza-maschera", action="store_true")
    ap.add_argument("--larghezza-cm", type=float, default=15.5)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    tess = [t.split("=", 1) for t in a.tessuto]
    nr, nc = len(tess), len(COLONNE)
    w = a.larghezza_cm / 2.54
    fig, axs = plt.subplots(nr, nc, figsize=(w, w * nr / nc * 1.12), squeeze=False)
    scelte = []
    for r, (etich, path) in enumerate(tess):
        root = Path(path)
        sp, gt = root / a.split, root / "ground_truth"
        for c, (nome, alias) in enumerate(COLONNE):
            ax = axs[r][c]; ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            d = cartella(sp, alias)
            if d is None or not immagini(d):
                ax.text(0.5, 0.5, "assente", ha="center", va="center", fontsize=7,
                        color="#8a8982", transform=ax.transAxes)
                ax.set_facecolor("#f1f0ec")
            else:
                img = scegli(d, rng)
                arr = np.array(Image.open(img).convert("RGB"))
                if nome != "Good" and not a.senza_maschera:
                    mp = trova_maschera(cartella(gt, alias), img)
                    if mp is not None:
                        m = np.array(Image.open(mp).convert("L").resize(
                            (arr.shape[1], arr.shape[0]), Image.NEAREST)) > 127
                        arr[contorno(m)] = (230, 30, 30)
                    else:
                        print("maschera non trovata per", img)
                ax.imshow(arr); scelte.append(str(img))
            if r == 0:
                ax.set_title(nome, fontsize=8)
            if c == 0:
                ax.set_ylabel(etich, fontsize=8, rotation=90, labelpad=4)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    print("salvata", a.out)
    Path(a.out).with_suffix(".txt").write_text("\n".join(scelte) + "\n")
    print("elenco delle immagini scelte in", Path(a.out).with_suffix(".txt"))

if __name__ == "__main__":
    main()