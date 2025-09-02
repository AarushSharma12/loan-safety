from __future__ import annotations
from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def get_feature_lists() -> Tuple[List[str], List[str]]:
    categorical = [
        "grade",
        "sub_grade",
        "short_emp",
        "home_ownership",
        "purpose",
        "term",
        "last_delinq_none",
        "last_major_derog_none",
    ]
    numeric = ["emp_length_num", "dti", "revol_util", "total_rec_late_fee"]
    return categorical, numeric


def build_preprocessor(categorical: List[str], numeric: List[str]) -> ColumnTransformer:
    # Trees don't need scaling; we one-hot categoricals and passthrough numerics
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical),
            ("num", "passthrough", numeric),
        ]
    )
    return preprocessor


def split_X_y(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y
