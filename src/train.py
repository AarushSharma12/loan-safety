from __future__ import annotations
import argparse
import time
from pathlib import Path
import joblib
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from features import get_feature_lists, build_preprocessor, split_X_y
from data import load_raw, prepare_labels


def train(config_path: str = "configs/config.yaml") -> str:
    cfg = yaml.safe_load(Path(config_path).read_text())

    raw_csv = Path(cfg["data"]["raw_csv"])
    target_col = cfg["data"]["target"]
    seed = int(cfg["training"]["random_seed"])
    test_size = float(cfg["training"]["test_size"])
    max_depth = int(cfg["model"]["decision_tree"]["max_depth"])

    cat, num = get_feature_lists()
    df = load_raw(raw_csv)
    # Keep only selected columns + target
    keep_cols = cat + num + [target_col]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")

    df = df[keep_cols]
    df = prepare_labels(df, target_col=target_col)

    X, y = split_X_y(df, target_col)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    pre = build_preprocessor(cat, num)
    clf = DecisionTreeClassifier(
        max_depth=max_depth, class_weight="balanced", random_state=seed
    )
    pipe = Pipeline(steps=[("pre", pre), ("model", clf)])
    pipe.fit(X_train, y_train)

    artifacts_dir = Path(cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    model_path = artifacts_dir / f"model_decision_tree_depth{max_depth}_{ts}.joblib"
    joblib.dump(
        {"pipeline": pipe, "features": {"categorical": cat, "numeric": num}}, model_path
    )

    # quick val accuracy
    acc = pipe.score(X_val, y_val)
    print(f"[train] saved: {model_path} | val_accuracy={acc:.4f}")
    return str(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
