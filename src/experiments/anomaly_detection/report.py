"""Anomaly Detection - Per-subclass score report

Writes the anomaly score of every held-out image with its subclass, a
per-subclass summary, and a box plot. This answers where junk / blur /
suspicious normals fall relative to the abnormal CTCs.
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Plot order: the positive class first, then the hardest negative, then the
# extra normal subclasses (any unknown subclass is appended alphabetically).
SUBCLASS_ORDER = ["abnormal", "suspicious", "junk", "blur", "red", "green", "blue"]


def build_score_table(
    entries: List[Dict[str, Any]], scores: np.ndarray, partition: str
) -> pd.DataFrame:
    """One row per scored image: path, subclass, label, partition, score."""
    return pd.DataFrame(
        {
            "path": [e["path"] for e in entries],
            "subclass": [e["subclass"] for e in entries],
            "label": [e["label"] for e in entries],
            "partition": partition,
            "score": np.asarray(scores, dtype=float),
        }
    )


def _ordered_subclasses(subclasses: List[str]) -> List[str]:
    known = [s for s in SUBCLASS_ORDER if s in subclasses]
    return known + sorted(set(subclasses) - set(known))


def summarize_by_subclass(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-subclass score statistics and the fraction flagged as anomalous."""
    rows = []
    for subclass in _ordered_subclasses(table["subclass"].unique().tolist()):
        scores = table.loc[table["subclass"] == subclass, "score"].to_numpy()
        rows.append(
            {
                "subclass": subclass,
                "n": len(scores),
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "q25": float(np.percentile(scores, 25)),
                "q75": float(np.percentile(scores, 75)),
                "frac_above_threshold": float(np.mean(scores > threshold)),
            }
        )
    return pd.DataFrame(rows)


def plot_subclass_scores(
    table: pd.DataFrame, threshold: float, output_path: Path, title: str
) -> None:
    """Horizontal box plot of scores per subclass with the decision threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = _ordered_subclasses(table["subclass"].unique().tolist())
    data = [table.loc[table["subclass"] == s, "score"].to_numpy() for s in order]
    labels = [f"{s} (n={len(d)})" for s, d in zip(order, data)]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(order) + 1.5))
    ax.boxplot(
        data,
        orientation="horizontal",
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": "#c9d7ee", "edgecolor": "#3d5a8a", "linewidth": 1},
        medianprops={"color": "#1f3a66", "linewidth": 2},
        whiskerprops={"color": "#3d5a8a", "linewidth": 1},
        capprops={"color": "#3d5a8a", "linewidth": 1},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "#3d5a8a",
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
    )
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(threshold, color="#555555", linestyle="--", linewidth=1)
    ax.text(
        threshold,
        0.02,
        " threshold",
        color="#555555",
        va="bottom",
        ha="left",
        fontsize=9,
        transform=ax.get_xaxis_transform(),
    )
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
