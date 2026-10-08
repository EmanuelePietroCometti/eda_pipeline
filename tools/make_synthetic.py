"""
make_synthetic.py: dataset sintetico con la struttura descritta nel README.

Serve SOLO a provare che il codice funzioni dove la verità è nota; non sostituisce
il dataset reale. Genera tre versioni (V0, V1, V2) del tessuto "nero":

* le patch sono ritagli 512 px con stride 480 di grandi teli (32 px di sovrapposizione);
* il train contiene good e difetti (come nel tuo dataset), il test pure;
* `ground_truth/` ha una maschera per ogni immagine (vuota per good e polvere);
* V0: nessuna polvere. V1: alcune immagini di test/good sostituite da immagini con
  polvere. V2: come V1, più alcune immagini di train/good sostituite da polvere.
  La numerosità di ogni split resta costante (la polvere sostituisce dei good).

Scrive anche `truth.json` con i nomi dei file con polvere e i rimossi.

    python tools/make_synthetic.py --out /tmp/synth --split-mode leaky
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def canvas(rng, size, base=70):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    tex = 14 * np.sin(2 * np.pi * xx / 7) * np.sin(2 * np.pi * yy / 9) + 6 * np.sin(2 * np.pi * (xx + yy) / 23)
    noise = gaussian_filter(rng.normal(0, 1, (size, size)).astype(np.float32), 1.0) * 12
    g = (base + tex + noise) * (1 + 0.08 * (xx / size - 0.5))
    return np.clip(g[..., None] * np.array([1.0, 0.95, 0.9], np.float32), 0, 255)


def add_paglia(rng, img):
    mask = np.zeros(img.shape[:2], np.uint8)
    for _ in range(rng.integers(1, 4)):
        L = int(rng.choice([6, 12, 25, 60, 120]))
        x, y, a = rng.integers(40, 472), rng.integers(40, 472), rng.uniform(0, np.pi)
        cv2.line(mask, (int(x), int(y)), (int(x + L * np.cos(a)), int(y + L * np.sin(a))), 255, int(rng.integers(2, 5)))
    m = mask > 0
    img[m] = 0.35 * img[m] + 0.65 * np.array([225, 200, 120], np.float32)
    return mask


def add_nodo(rng, img):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(mask, (int(rng.integers(40, 470)), int(rng.integers(40, 470))),
                (int(rng.integers(4, 15)), int(rng.integers(4, 15))), float(rng.uniform(0, 180)), 0, 360, 255, -1)
    m = mask > 0
    img[m] = 0.5 * img[m] + 0.5 * 235
    return mask


def add_dust(rng, img):
    for _ in range(int(rng.integers(30, 80))):
        cv2.circle(img, (int(rng.integers(40, 472)), int(rng.integers(40, 472))), 1, (25, 25, 25), -1)


def write(root, split, folder, stem, img, mask):
    d = Path(root) / split / folder
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{stem}.png"), cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    g = Path(root) / "ground_truth" / folder
    g.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(g / f"{stem}_mask.png"), np.zeros(img.shape[:2], np.uint8) if mask is None else mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-mode", choices=["leaky", "clean"], default="leaky")
    ap.add_argument("--rolls", type=int, default=8)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--dust-test", type=int, default=6)
    ap.add_argument("--dust-train", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    size = 512 + 480 * (a.grid - 1)

    cells = []                                   # (stem, split, kind, patch, mask)
    reserve = []                                 # patch buoni tenuti da parte per diventare polvere
    for roll in range(a.rolls):
        cv = canvas(rng, size)
        for r in range(a.grid):
            for c in range(a.grid):
                patch = cv[r * 480:r * 480 + 512, c * 480:c * 480 + 512].copy()
                stem = f"nero_roll{roll}_r{r}_c{c}"
                u = rng.random()
                if u < 0.17:
                    reserve.append((stem, patch))
                    continue
                kind = "paglia" if u < 0.24 else "nodo" if u < 0.30 else "good"
                if a.split_mode == "clean":
                    split = "test" if roll >= a.rolls - max(2, a.rolls // 4) else "train"
                else:
                    split = "train" if rng.random() < 0.7 else "test"
                mask = add_paglia(rng, patch) if kind == "paglia" else add_nodo(rng, patch) if kind == "nodo" else None
                cells.append((stem, split, kind, patch, mask))

    need = a.dust_test + a.dust_train
    assert len(reserve) >= need, f"servono {need} patch di riserva, ce ne sono {len(reserve)}: aumenta --rolls"
    rng.shuffle(reserve)
    dust_test, dust_train = reserve[:a.dust_test], reserve[a.dust_test:need]

    def good_cells(split):
        return [i for i, c in enumerate(cells) if c[1] == split and c[2] == "good"]

    rm_test = set(rng.choice(good_cells("test"), a.dust_test, replace=False).tolist())
    rm_train = set(rng.choice(good_cells("train"), a.dust_train, replace=False).tolist())

    truth = {"V1": {"dust": [], "removed": []}, "V2": {"dust": [], "removed": []}}
    for v in ("V0", "V1", "V2"):
        root = out / v
        if root.exists():
            shutil.rmtree(root)
        removed = set()
        if v in ("V1", "V2"):
            removed |= rm_test
        if v == "V2":
            removed |= rm_train
        for i, (stem, split, kind, patch, mask) in enumerate(cells):
            if i in removed:
                truth[v]["removed"].append(f"{split}/{stem}")
                continue
            write(root, split, kind, stem, patch, mask)
        added = []
        if v in ("V1", "V2"):
            added += [("test", s, p) for s, p in dust_test]
        if v == "V2":
            added += [("train", s, p) for s, p in dust_train]
        for split, stem, patch in added:
            p = patch.copy()
            add_dust(rng, p)
            write(root, split, "good", stem, p, None)
            truth[v]["dust"].append(f"{split}/{stem}")
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    for v in ("V0", "V1", "V2"):
        n = {s: sum(1 for _ in (out / v / s).rglob("*.png")) for s in ("train", "test")}
        print(v, n)


if __name__ == "__main__":
    main()
