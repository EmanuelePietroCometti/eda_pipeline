"""Piccole funzioni condivise: configurazione, colori, stile dei grafici, salvataggio."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # i grafici si salvano su file, non si aprono finestre
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402

# Colori dei sei gruppi del README (palette "colorblind"), scelti perché train e test, buoni e
# polvere si distinguano anche quando i punti si sovrappongono.
GROUP_COLORS = {"train_good": "#0173b2", "test_good": "#de8f05", "train_dust": "#cc78bc",
                "test_dust": "#029e73", "train_def": "#d55e00", "test_def": "#ca9161"}


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def save_fig(fig, path, dpi=200, pad=0.1):
    """`pad`: margine in pollici attorno al ritaglio automatico (nei grafici 3D il ritaglio
    ignora l'etichetta z del primo pannello, che senza margine viene tagliata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    return path
