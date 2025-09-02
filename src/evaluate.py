from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from data import load_raw, prepare_labels
from features import split_X_y


def evaluate(model_path: str, config_path: str = "configs/config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    raw_csv = cfg["data"]["raw_csv"]
    target_col = cfg["data"]["target"]

    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]

    df = load_raw(raw_csv)
    df = prepare_labels(df, target_col=target_col)
    X, y = split_X_y(df, target_col)

    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"[evaluate] accuracy={acc:.4f}  f1={f1:.4f}")

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["risky(0)", "safe(1)"]
    )
    plt.figure()
    disp.plot(values_format="d")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True, parents=True)
    out_img = outputs_dir / "confusion_matrix.png"
    plt.savefig(out_img, bbox_inches="tight")
    print(f"[evaluate] saved confusion matrix → {out_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    evaluate(args.model, args.config)
