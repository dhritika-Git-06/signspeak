"""Classification report and confusion-matrix figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def classification_payload(y_true, y_pred, labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    report = classification_report(
        y_true, y_pred, target_names=labels, digits=4, zero_division=0, output_dict=True
    )
    text = classification_report(
        y_true, y_pred, target_names=labels, digits=4, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    per_class = {}
    for name in labels:
        row = report[name]
        per_class[name] = {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1-score"]),
            "support": int(row["support"]),
        }
    confusions = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i != j and matrix[i, j] > 0:
                confusions.append(
                    {"actual": actual, "predicted": predicted, "count": int(matrix[i, j])}
                )
    confusions.sort(key=lambda item: item["count"], reverse=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "labels": labels,
        "per_class": per_class,
        "report": text,
        "confusion_matrix": matrix.tolist(),
        "top_confusions": confusions[:8],
        "test_size": int(len(y_true)),
    }


def save_confusion_png(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix, dtype=np.float32)
    fig_w = max(8, len(labels) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.86))
    image = ax.imshow(arr, cmap="YlOrBr")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
