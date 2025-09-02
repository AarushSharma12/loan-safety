from __future__ import annotations
import pandas as pd
from pathlib import Path


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    return df


def prepare_labels(df: pd.DataFrame, target_col: str = "safe_loans") -> pd.DataFrame:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in dataframe.")
    # Map +1 -> 1 (safe), -1 -> 0 (risky)
    y_map = {+1: 1, -1: 0}
    df = df.copy()
    df[target_col] = df[target_col].map(y_map)
    # Drop rows with unknown labels
    df = df[df[target_col].isin([0, 1])]
    return df
