import sys
from pathlib import Path
import os
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def import_face_module(backend_path: Path):
    sys.path.insert(0, str(backend_path))
    import face_module
    return face_module


def collect_test_images(test_root: Path):
    images = []
    for user_dir in sorted(test_root.iterdir()):
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name
        face_dir = user_dir / "face"
        if not face_dir.exists():
            continue
        for p in sorted(face_dir.iterdir()):
            if p.is_file():
                images.append((user_id, p))
    return images


def plot_confusion_matrix(cm, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels, ylabel="Actual", xlabel="Predicted", title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate counts
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def bar_single_value(value, title, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar([0], [value], width=0.6)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_title(title)
    ax.text(0, value + 0.02, f"{value:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def bar_per_class(values, labels, title, out_path: Path):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_dataset", default="evaluation_dataset", help="Path to evaluation_dataset folder")
    parser.add_argument("--results", default="evaluation_results", help="Path to write evaluation results")
    parser.add_argument("--backend", default="backend", help="Path to backend folder containing face_module.py")
    args = parser.parse_args()

    repo_root = Path(__file__).parent
    eval_root = repo_root / args.evaluation_dataset
    results_root = repo_root / args.results
    backend_path = repo_root / args.backend

    if not eval_root.exists():
        raise RuntimeError(f"Evaluation dataset not found: {eval_root}. Run create_train_test_split.py first.")

    (results_root).mkdir(parents=True, exist_ok=True)

    face_module = import_face_module(backend_path)

    # Use train split as known dataset for recognition
    face_module.DATASET_PATH = str(eval_root / "train")
    # Clear any encoding cache
    if hasattr(face_module, "_FACE_ENCODING_CACHE"):
        face_module._FACE_ENCODING_CACHE = None

    test_root = eval_root / "test"
    images = collect_test_images(test_root)

    y_true = []
    y_pred = []
    confidences = []
    recognized_flags = []

    for actual_user, img_path in images:
        try:
            res = face_module.recognize_face(str(img_path))
            predicted = res.get("user_id")
            confidence = float(res.get("confidence", 0.0))
            recognized = bool(res.get("recognized", False))
        except Exception as exc:
            predicted = "__ERROR__"
            confidence = 0.0
            recognized = False
            print(f"Error processing {img_path}: {exc}")

        y_true.append(actual_user)
        y_pred.append(predicted)
        confidences.append(confidence)
        recognized_flags.append(recognized)

    labels = sorted(list(set(y_true) | set(y_pred)))

    # Metrics (use weighted averages for global numbers)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, labels, results_root / "confusion_matrix.png")

    # Performance graphs
    bar_single_value(acc, "Accuracy", results_root / "accuracy_graph.png")

    # Per-class metrics
    per_prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    bar_per_class(per_prec, labels, "Precision (per class)", results_root / "precision_graph.png")
    bar_per_class(per_rec, labels, "Recall (per class)", results_root / "recall_graph.png")
    bar_per_class(per_f1, labels, "F1 Score (per class)", results_root / "f1_graph.png")

    # Write report
    report_path = results_root / "evaluation_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Total Test Samples: {len(y_true)}\n")
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        fh.write(f"Correct Predictions: {correct}\n")
        fh.write(f"Incorrect Predictions: {len(y_true) - correct}\n")
        fh.write(f"Accuracy: {acc:.6f}\n")
        fh.write(f"Precision: {prec:.6f}\n")
        fh.write(f"Recall: {rec:.6f}\n")
        fh.write(f"F1 Score: {f1:.6f}\n\n")
        fh.write("Classification Report:\n")
        fh.write(classification_report(y_true, y_pred, zero_division=0))

    print(f"Evaluation complete. Results saved to: {results_root}")


if __name__ == "__main__":
    main()
