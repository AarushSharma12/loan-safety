from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt

from data import load_raw, ensure_target, prepare_labels
from features import split_X_y


def plot_feature_importance(
    pipeline, model_type="unknown", top_n=20, out_path="outputs/feature_importance.png"
):
    """Plot feature importance for tree-based models."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]

    if not isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
        return

    # Get feature names
    cat_names = pre.named_transformers_["cat"].get_feature_names_out()
    num_names = np.array(pre.transformers_[1][2])
    names = np.concatenate([cat_names, num_names])

    # Get importances
    importances = model.feature_importances_

    # For Random Forest, we can also get std
    if isinstance(model, RandomForestClassifier):
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    else:
        std = None

    # Sort and select top features
    idx = np.argsort(importances)[::-1][:top_n]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    if std is not None:
        ax.barh(range(len(idx)), importances[idx][::-1], xerr=std[idx][::-1], alpha=0.8, capsize=3)
    else:
        ax.barh(range(len(idx)), importances[idx][::-1], alpha=0.8)

    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(names[idx][::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features - {model_type}")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close()


def plot_roc_curve(y_true, y_proba, out_path="outputs/roc_curve.png"):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    Path("outputs").mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close()


def evaluate(model_path: str, config_path: str = "configs/config.yaml") -> dict:
    """Comprehensive model evaluation."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    raw_csv = cfg["data"]["raw_csv"]
    preferred_target = cfg["data"]["target"]

    # Load model bundle
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    model_type = bundle.get("model_type", "unknown")

    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {Path(model_path).name}")
    print(f"Model Type: {model_type}")
    print(f"{'='*60}\n")

    # Load and prepare data
    df = load_raw(raw_csv)
    df, resolved_target = ensure_target(df, preferred_target=preferred_target)
    df = prepare_labels(df, target_col=resolved_target)
    X, y = split_X_y(df, resolved_target)

    # Make predictions
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }

    # Print metrics
    print("PERFORMANCE METRICS:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.upper():12s}: {value:.4f}")

    # Print classification report
    print("\nCLASSIFICATION REPORT:")
    print("-" * 40)
    print(classification_report(y, y_pred, target_names=["Risky (0)", "Safe (1)"], digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    print("CONFUSION MATRIX:")
    print("-" * 40)
    print(f"              Predicted")
    print(f"              Risky  Safe")
    print(f"Actual Risky  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Safe   {cm[1,0]:5d}  {cm[1,1]:5d}")
    print()

    # Calculate business metrics
    total_loans = len(y)
    actual_safe = y.sum()
    actual_risky = total_loans - actual_safe

    true_positives = cm[1, 1]  # Correctly identified safe loans
    false_positives = cm[0, 1]  # Risky loans incorrectly labeled as safe
    true_negatives = cm[0, 0]  # Correctly identified risky loans
    false_negatives = cm[1, 0]  # Safe loans incorrectly labeled as risky

    print("BUSINESS IMPACT:")
    print("-" * 40)
    print(f"Total loans evaluated: {total_loans:,}")
    print(
        f"Actual distribution: {actual_safe:,} safe ({actual_safe/total_loans*100:.1f}%), "
        f"{actual_risky:,} risky ({actual_risky/total_loans*100:.1f}%)"
    )
    print()
    print(f"✓ Correctly approved safe loans: {true_positives:,}")
    print(f"✓ Correctly rejected risky loans: {true_negatives:,}")
    print(f"✗ Risky loans wrongly approved: {false_positives:,} (potential losses)")
    print(f"✗ Safe loans wrongly rejected: {false_negatives:,} (missed opportunities)")
    print()

    # Create visualizations
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True, parents=True)

    # 1. Confusion Matrix Plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Risky(0)", "Safe(1)"])
    plt.figure(figsize=(8, 6))
    disp.plot(values_format="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_type}")
    out_img = outputs_dir / "confusion_matrix.png"
    plt.savefig(out_img, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"[evaluate] Saved confusion matrix → {out_img}")

    # 2. ROC Curve
    plot_roc_curve(y, y_proba, outputs_dir / "roc_curve.png")
    print(f"[evaluate] Saved ROC curve → outputs/roc_curve.png")

    # 3. Feature Importance
    try:
        plot_feature_importance(
            pipe, model_type, top_n=20, out_path=outputs_dir / "feature_importance.png"
        )
        print(f"[evaluate] Saved feature importance → outputs/feature_importance.png")
    except Exception as e:
        print(f"[evaluate] Feature importance skipped: {e}")

    # Save metrics to file
    metrics_file = outputs_dir / "evaluation_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Model: {Path(model_path).name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"{'='*50}\n\n")

        f.write("Performance Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")

        f.write(f"\n{classification_report(y, y_pred, target_names=['Risky', 'Safe'])}")

        if "training_score" in bundle:
            f.write(f"\nTraining Score: {bundle['training_score']:.4f}\n")
        if "validation_score" in bundle:
            f.write(f"Validation Score: {bundle['validation_score']:.4f}\n")

    print(f"[evaluate] Saved metrics summary → {metrics_file}")
    print(f"\n{'='*60}\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    evaluate(args.model, args.config)
