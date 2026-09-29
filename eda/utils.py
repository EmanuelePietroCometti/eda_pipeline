"""Piccole funzioni condivise: configurazione, stile dei grafici, salvataggio."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # i grafici si salvano su file, non si aprono finestre
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402

# I sei gruppi definiti in design.md. L'ordine è quello delle figure.
GROUPS = ["train_good", "train_dust", "train_def", "test_good", "test_dust", "test_def"]
GROUP_COLORS = dict(zip(GROUPS, sns.color_palette("colorblind", len(GROUPS))))


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def save_fig(fig, path, dpi=200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
