"""
src/utils/helpers.py
====================
Shared utility functions used by the training pipeline and Streamlit UI.

Public API
----------
compute_perplexity    : cross-entropy loss → perplexity score
format_metric_table   : raw float metrics → display-ready strings
plot_training_curves  : save accuracy + loss PNG charts to results/
human_readable_size   : bytes → "4.8 MB" string
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed on server
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ── Perplexity ────────────────────────────────────────────────────────────────

def compute_perplexity(loss: float) -> float:
    """
    Convert cross-entropy loss to perplexity.

    Perplexity = e^(cross_entropy_loss)

    Interpretation
    --------------
    - Lower perplexity = better model.
    - A model guessing randomly over V words has perplexity ≈ V.
    - Good language models target perplexity < 100 on their domain.
    - Example: loss=3.7 → perplexity≈40 (model "considers" ~40 words
      equally likely at each step).

    Parameters
    ----------
    loss : float  Cross-entropy loss (must be non-negative).

    Returns
    -------
    float  Perplexity score (≥ 1.0).
    """
    return float(np.exp(max(0.0, loss)))


# ── Metric Formatting ─────────────────────────────────────────────────────────

def format_metric_table(metrics: dict[str, float]) -> dict[str, str]:
    """
    Convert raw float metrics to human-readable display strings.

    Used by the Streamlit UI to show formatted results without
    repeating the formatting logic across multiple widgets.

    Formatting rules
    ----------------
    Keys ending in '_acc'   → percentage with 1 decimal  e.g. "62.3%"
    Key  'perplexity'       → decimal with 2 places       e.g. "42.05"
    Keys ending in '_loss'  → decimal with 4 places       e.g. "1.8234"
    Anything else           → decimal with 4 places

    Parameters
    ----------
    metrics : dict[str, float]
        Raw metrics from evaluate(), e.g.:
        {"train_acc": 0.6213, "val_loss": 2.1, "perplexity": 42.0}

    Returns
    -------
    dict[str, str]  Formatted strings keyed by the same names.
    """
    formatted: dict[str, str] = {}

    for key, value in metrics.items():
        if key.endswith("_acc"):
            formatted[key] = f"{value * 100:.1f}%"
        elif key == "perplexity":
            formatted[key] = f"{value:.2f}"
        elif key.endswith("_loss"):
            formatted[key] = f"{value:.4f}"
        else:
            formatted[key] = f"{value:.4f}"

    return formatted


# ── Training Curve Plots ──────────────────────────────────────────────────────

def plot_training_curves(
    history  : object,
    save_dir : Path,
) -> list[Path]:
    """
    Save accuracy and loss training curve PNGs to save_dir.

    Generates two files:
        training_curves.png  — accuracy + loss side by side
        accuracy_summary.png — horizontal bar chart of final metrics

    Parameters
    ----------
    history  : keras.callbacks.History  Object returned by model.fit().
    save_dir : Path                     Directory to write PNG files.

    Returns
    -------
    list[Path]  Paths of the saved PNG files.

    Note
    ----
    Uses the Agg matplotlib backend so this runs without a display,
    which is required in both headless servers and Streamlit Cloud.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    # ── Plot 1: Accuracy + Loss side by side ─────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "LSTM Next Word Predictor — Training Curves",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Accuracy subplot
    axes[0].plot(epochs, hist.get("accuracy", []),
                 color="#2a78d6", linewidth=2, label="Train")
    if "val_accuracy" in hist:
        axes[0].plot(epochs, hist["val_accuracy"],
                     color="#e34948", linewidth=2,
                     linestyle="--", label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    # Loss subplot
    axes[1].plot(epochs, hist.get("loss", []),
                 color="#2a78d6", linewidth=2, label="Train")
    if "val_loss" in hist:
        axes[1].plot(epochs, hist["val_loss"],
                     color="#e34948", linewidth=2,
                     linestyle="--", label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = save_dir / "training_curves.png"
    fig.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(curves_path)
    logger.info("Training curves saved → %s", curves_path)

    # ── Plot 2: Accuracy summary bar chart ────────────────────────────────────

    labels = []
    values = []
    colors = []

    metric_map = {
        "accuracy"     : ("Train Acc",  "#2a78d6"),
        "val_accuracy" : ("Val Acc",    "#1baf7a"),
    }

    for key, (label, color) in metric_map.items():
        if key in hist and hist[key]:
            labels.append(label)
            values.append(hist[key][-1])   # last epoch value
            colors.append(color)

    if labels:
        fig2, ax = plt.subplots(figsize=(6, 3))
        bars = ax.barh(labels, values, color=colors, height=0.4)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Accuracy")
        ax.set_title("Final Epoch Accuracy", fontweight="bold")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va="center", fontsize=10,
            )
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        summary_path = save_dir / "accuracy_summary.png"
        fig2.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        saved.append(summary_path)
        logger.info("Accuracy summary saved → %s", summary_path)

    return saved


# ── Human-Readable File Size ──────────────────────────────────────────────────

def human_readable_size(n_bytes: int) -> str:
    """
    Convert a byte count to a human-readable string.

    Parameters
    ----------
    n_bytes : int  File size in bytes.

    Returns
    -------
    str  e.g. "4.8 MB", "53.6 KB", "964 B"
    """
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}" if unit != "B" else f"{n_bytes} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"
