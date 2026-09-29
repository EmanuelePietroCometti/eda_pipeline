"""
run.py: esegue l'EDA descritta in design.md.

    python run.py                          # tutto, con config.yaml
    python run.py --steps integrity masks  # solo alcuni passi
    python run.py --config altro.yaml

Ordine: integrity -> masks -> leakage -> domain -> knn -> gallery.
Maschere e coerenza si fanno una volta sulla versione di riferimento (V0);
leakage, domain shift e kNN si ripetono per ogni versione.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from eda import domain_shift, gallery, knn, leakage, masks
from eda.dataset import assign_groups, build_index, check_masks, compare_versions, find_duplicates, identify_dust
from eda.features import compute_embeddings, scan_images
from eda.utils import load_config, set_style

STEPS = ["integrity", "masks", "leakage", "domain", "knn", "gallery"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_style()
    out = Path(cfg["general"]["out_dir"])
    ref = cfg["dataset"]["reference_version"]
    versions = cfg["dataset"]["versions"]

    # ---- indice di ogni versione, polvere per differenza da V0, gruppi
    idx = {v: build_index(vc["path"], cfg) for v, vc in versions.items()}
    for v in idx:
        idx[v] = idx[v].assign(dust=False) if v == ref else identify_dust(idx[ref], idx[v])
        idx[v] = assign_groups(idx[v])
        (out / v).mkdir(parents=True, exist_ok=True)
        idx[v].to_csv(out / v / "index.csv", index=False)
        print(f"\n[{v}] immagini per gruppo:\n{idx[v]['group'].value_counts().sort_index().to_string()}")

    # ---- 0. controlli di integrità
    if "integrity" in args.steps:
        print("\n=== controlli di integrità ===")
        ok = True
        for v, vc in versions.items():
            issues = check_masks(idx[v], cfg["masks"]["threshold"])
            issues.to_csv(out / v / "mask_issues.csv", index=False)
            print(f"[{v}] problemi sulle maschere: {len(issues)}")
            dup = find_duplicates(idx[v])
            dup.to_csv(out / v / "duplicates.csv", index=False)
            if len(dup):
                cross = int((dup["splits"] == "test+train").sum())
                print(f"[{v}] contenuti duplicati (pixel identici): {len(dup)} ({int(dup['n_copie'].sum())} file), "
                      f"di cui a cavallo di train e test: {cross}  -> {out / v / 'duplicates.csv'}")
            if v != ref:
                checks = compare_versions(idx[ref], idx[v], vc["dust_train"])
                checks.to_csv(out / v / "version_checks.csv", index=False)
                print(checks.to_string(index=False))
                ok &= bool(checks["ok"].all())
            ok &= issues.empty
        if not ok:
            print("\nATTENZIONE: almeno un controllo è fallito. Guarda i CSV prima di fidarti delle analisi.")

    # ---- 1. maschere (una volta, su V0)
    if "masks" in args.steps:
        print(f"\n=== 1. maschere ({ref}) ===")
        masks.run(idx[ref], cfg, str(out / ref))

    # ---- 2-4 per versione
    knn_all = {}
    for v in versions:
        need_scan = any(s in args.steps for s in ("leakage", "domain"))
        need_emb = any(s in args.steps for s in ("domain", "knn"))
        paths = idx[v]["filename"].tolist()
        scan = scan_images(paths, cfg["leakage"]["overlap_px"]) if need_scan else None
        emb = compute_embeddings(paths, cfg["knn"]["backbone"], cfg["general"]["model_input_size"]) if need_emb else None
        groups = None
        if "leakage" in args.steps:
            print(f"\n=== 3. leakage ({v}) ===")
            leak = leakage.run(idx[v], scan, cfg, str(out / v))
            groups = leak["group_id"].to_numpy()
        if "domain" in args.steps:
            print(f"\n=== 2. domain shift ({v}) ===")
            domain_shift.run(idx[v], scan, emb, groups, cfg, str(out / v))
        if "knn" in args.steps:
            print(f"\n=== 4. kNN ({v}) ===")
            knn_all[v] = knn.run(idx[v], emb, cfg, str(out / v))

    # ---- contrasti tra versioni (design.md, 4.4)
    if knn_all:
        pk = cfg["knn"]["primary_k"]
        tab = pd.concat([m[(m["k"] == pk) & (m["subset"] == "tutte")].assign(version=v) for v, m in knn_all.items()])
        wide = tab.pivot(index="comparison", columns="version", values="auroc")
        wide.to_csv(out / "knn_contrasts.csv")
        print("\n=== AUROC del kNN per versione ===\n" + wide.round(3).to_string())
        knn.plot_contrasts(tab, out / "figures" / "knn_contrasts.png")

    # ---- 5. galleria: sulla prima versione con polvere nel test
    if "gallery" in args.steps:
        v = next((v for v, vc in versions.items() if vc["dust_test"]), ref)
        print(f"\n=== 5. galleria ({v}) ===")
        gallery.run(idx[v], cfg, str(out / v))

    (out / "run_info.json").write_text(json.dumps({"config": cfg}, indent=2, default=str))
    print("\n=== figure scritte ===")
    for d in sorted(p for p in out.rglob("figures") if p.is_dir()):
        print(f"{d.relative_to(out).parent}: " + ", ".join(sorted(f.name for f in d.glob("*.png"))))
    print(f"\nFatto. Risultati in {out}")


if __name__ == "__main__":
    main()
