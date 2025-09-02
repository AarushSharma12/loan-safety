import pandas as pd
from sklearn.model_selection import train_test_split
from features import get_feature_lists, build_preprocessor


def test_preprocessor_shapes():
    cat, num = get_feature_lists()
    df = pd.DataFrame(
        {
            "grade": ["A", "B", "A", "C"],
            "sub_grade": ["A1", "B2", "A1", "C3"],
            "short_emp": [0, 1, 0, 1],
            "home_ownership": ["RENT", "OWN", "RENT", "MORTGAGE"],
            "purpose": ["debt_consolidation"] * 4,
            "term": ["36 months", "60 months", "36 months", "60 months"],
            "last_delinq_none": [1, 1, 1, 0],
            "last_major_derog_none": [1, 1, 1, 1],
            "emp_length_num": [5, 2, 10, 7],
            "dti": [10.5, 22.1, 15.2, 30.0],
            "revol_util": [45.0, 60.2, 12.0, 88.8],
            "total_rec_late_fee": [0.0, 1.2, 0.0, 3.4],
        }
    )
    y = pd.Series([1, 0, 1, 0])

    pre = build_preprocessor(cat, num)
    Xt = pre.fit_transform(df)
    assert Xt.shape[0] == 4
    assert Xt.shape[1] >= len(num)  # + one-hot columns


def test_split_is_deterministic():
    import numpy as np

    X = pd.DataFrame({"a": range(100)})
    y = pd.Series([0, 1] * 50)
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X3, X4, y3, y4 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    assert np.all(X1.index == X3.index)
    assert np.all(y1.index == y3.index)
