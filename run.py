"""
run.py: esegue l'EDA descritta nel README.

    python run.py                              # tutto, con config.yaml
    python run.py --steps counts               # solo alcuni passi
    python run.py --config altro.yaml

Passi:
    counts       numerosità per gruppo e versione, con l'area mediana dei difetti
    projections  PCA, t-SNE e UMAP in 2D: buoni e polvere; buoni vs difetti (per versione)
"""

import argparse
import json
from pathlib import Path

from eda import masks, projections
from eda.dataset import assign_groups, build_index, identify_dust
from eda.features import embed_versions, standardize
from eda.utils import load_config, set_style

STEPS = ["counts", "projections"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_style()
    out = Path(cfg["general"]["out_dir"])
    ref = cfg["dataset"]["reference_version"]
    vcfg = cfg["dataset"]["versions"]
    if ref not in vcfg:
        raise SystemExit(f"reference_version={ref} non è tra le versioni del config: {list(vcfg)}")

    # ---- indice di ogni versione, polvere per differenza dalla versione di riferimento, gruppi
    idx = {v: build_index(vc["path"], cfg) for v, vc in vcfg.items()}
    for v in idx:
        idx[v] = idx[v].assign(dust=False) if v == ref else identify_dust(idx[ref], idx[v])
        idx[v] = assign_groups(idx[v])
        (out / v / "figures").mkdir(parents=True, exist_ok=True)
        idx[v].to_csv(out / v / "index.csv", index=False)

    if "counts" in args.steps:
        print("\n=== numerosità e area mediana ===")
        masks.run(idx, cfg, out)

    if "projections" in args.steps:
        bb, size = cfg["embedding"]["backbone"], cfg["general"]["model_input_size"]
        emb = embed_versions(idx, bb, size, cache=out / f"embeddings_{bb}_{size}.npz")
        for v in vcfg:
            print(f"\n=== proiezioni ({v}) ===")
            Z = standardize(emb[v], (idx[v]["group"] == "train_good").to_numpy())
            projections.run(idx[v], Z, cfg, str(out / v), v)

    (out / "run_info.json").write_text(json.dumps({"config": cfg}, indent=2, default=str))
    print(f"\nFatto. Risultati in {out}")


if __name__ == "__main__":
    main()
