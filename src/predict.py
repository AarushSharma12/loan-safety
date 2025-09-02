from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import pandas as pd


def predict(model_path: str, json_path: str):
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]

    record = json.loads(Path(json_path).read_text())
    # Accept single object or list
    if isinstance(record, dict):
        df = pd.DataFrame([record])
    else:
        df = pd.DataFrame(record)

    proba = pipe.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    for i, (p, s) in enumerate(zip(pred, proba)):
        label = "safe(1)" if p == 1 else "risky(0)"
        print(f"row {i}: {label} | p(safe)={s:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--input", required=True, help="Path to JSON with feature values"
    )
    args = parser.parse_args()
    predict(args.model, args.input)
