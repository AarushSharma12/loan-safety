from __future__ import annotations
import argparse
import time
from pathlib import Path
import joblib
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from features import get_feature_lists, build_preprocessor, split_X_y
from data import load_raw, ensure_target, prepare_labels


def create_model(model_type: str, cfg: dict, seed: int):
    """Create a model based on type and configuration."""
    if model_type == "decision_tree":
        params = cfg["model"].get("decision_tree", {})
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 6),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight="balanced",
            random_state=seed,
        )
    elif model_type == "random_forest":
        params = cfg["model"].get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,  # Use all CPU cores
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(config_path: str = "configs/config.yaml", model_type: str = None) -> str:
    cfg = yaml.safe_load(Path(config_path).read_text())

    raw_csv = Path(cfg["data"]["raw_csv"])
    target_col = cfg["data"]["target"]
    seed = int(cfg["training"]["random_seed"])
    test_size = float(cfg["training"]["test_size"])

    # Determine model type
    if model_type is None:
        model_type = cfg.get("model", {}).get("type", "decision_tree")

    # Cross-validation settings
    cv_folds = cfg["training"].get("cross_validation", {}).get("folds", 5)
    perform_cv = cfg["training"].get("cross_validation", {}).get("enabled", False)

    cat, num = get_feature_lists()
    df = load_raw(raw_csv)

    df, resolved_target = ensure_target(df, preferred_target=target_col)

    keep_cols = cat + num + [resolved_target]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")

    df = df[keep_cols]
    df = prepare_labels(df, target_col=resolved_target)

    X, y = split_X_y(df, resolved_target)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    pre = build_preprocessor(cat, num)
    clf = create_model(model_type, cfg, seed)
    pipe = Pipeline(steps=[("pre", pre), ("model", clf)])

    # Optional cross-validation
    if perform_cv:
        print(f"[train] Performing {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring="accuracy")
        print(f"[train] CV scores: {cv_scores}")
        print(f"[train] CV mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train final model
    print(f"[train] Training {model_type} model...")
    pipe.fit(X_train, y_train)

    # Get model parameters for filename
    if model_type == "decision_tree":
        max_depth = cfg["model"]["decision_tree"]["max_depth"]
        model_desc = f"decision_tree_depth{max_depth}"
    else:  # random_forest
        n_est = cfg["model"]["random_forest"]["n_estimators"]
        max_depth = cfg["model"]["random_forest"]["max_depth"]
        model_desc = f"random_forest_n{n_est}_depth{max_depth}"

    artifacts_dir = Path(cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    model_path = artifacts_dir / f"model_{model_desc}_{ts}.joblib"

    # Save model with metadata
    joblib.dump(
        {
            "pipeline": pipe,
            "model_type": model_type,
            "features": {"categorical": cat, "numeric": num},
            "config": cfg["model"],
            "training_score": pipe.score(X_train, y_train),
            "validation_score": pipe.score(X_val, y_val),
        },
        model_path,
    )

    # Report accuracies
    train_acc = pipe.score(X_train, y_train)
    val_acc = pipe.score(X_val, y_val)
    print(f"[train] saved: {model_path}")
    print(f"[train] train_accuracy={train_acc:.4f} | val_accuracy={val_acc:.4f}")

    # Feature importance for tree-based models
    if hasattr(clf, "feature_importances_"):
        pre_fitted = pipe.named_steps["pre"]
        cat_names = pre_fitted.named_transformers_["cat"].get_feature_names_out()
        import numpy as np

        num_names = np.array(pre_fitted.transformers_[1][2])
        feature_names = np.concatenate([cat_names, num_names])
        importances = clf.feature_importances_

        # Top 5 features
        top_idx = np.argsort(importances)[::-1][:5]
        print(f"[train] Top 5 features:")
        for i, idx in enumerate(top_idx, 1):
            print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")

    return str(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--model",
        choices=["decision_tree", "random_forest"],
        help="Model type to train (overrides config)",
    )
    args = parser.parse_args()
    train(args.config, args.model)
