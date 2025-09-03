from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path.resolve()}")
    # Avoid mixed-type chunking warning
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def ensure_target(
    df: pd.DataFrame, preferred_target: str = "safe_loans"
) -> tuple[pd.DataFrame, str]:
    """
    Ensure we have a binary 'safe_loans' target.
    If missing but 'bad_loans' exists, create: safe_loans = 1 - bad_loans.
    Returns (df_with_target, target_col_name).
    """
    df = df.copy()
    if preferred_target in df.columns:
        return df, preferred_target

    if "bad_loans" in df.columns:
        # bad_loans: 1 = bad/risky, 0 = good → map to safe_loans: 1=safe, 0=risky
        df["safe_loans"] = (df["bad_loans"] == 0).astype(int)
        return df, "safe_loans"

    raise KeyError("Missing target column. Expected 'safe_loans' or 'bad_loans' in the CSV.")


def prepare_labels(df: pd.DataFrame, target_col: str = "safe_loans") -> pd.DataFrame:
    """
    Normalize labels to {0,1} where 1 = safe, 0 = risky.
    Accepts {+1,-1} or {1,0}. Filters out any unknowns.
    """
    df = df.copy()
    if set(pd.unique(df[target_col])) <= {+1, -1}:
        df[target_col] = df[target_col].map({+1: 1, -1: 0})
    else:
        df[target_col] = df[target_col].astype(int)
    df = df[df[target_col].isin([0, 1])]
    return df
