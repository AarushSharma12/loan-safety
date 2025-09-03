import sys, json
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import yaml

# make 'src/' importable
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

# import your project code
from train import train as train_fn
from data import load_raw, ensure_target, prepare_labels
from features import get_feature_lists, build_preprocessor, split_X_y

CFG_PATH_DEFAULT = "configs/config.yaml"


@st.cache_data
def load_cfg(cfg_path: str):
    return yaml.safe_load(Path(cfg_path).read_text())


@st.cache_data
def load_choices(csv_path: str, target_preferred: str):
    df = load_raw(csv_path)
    # derive target if needed so we can drop it later
    df, target = ensure_target(df, preferred_target=target_preferred)
    cat_cols, num_cols = get_feature_lists()
    # keep only the columns we need (skip rows with NAs on needed cols)
    keep = [c for c in cat_cols + num_cols if c in df.columns]
    df_small = df[keep].dropna().head(10_000)  # limit for speed
    choices = {
        c: sorted(map(str, df_small[c].dropna().unique().tolist()))
        for c in cat_cols
        if c in df_small
    }
    return choices, cat_cols, num_cols


def choose_latest_model():
    arts = sorted(
        (ROOT / "artifacts").glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return str(arts[0]) if arts else ""


st.set_page_config(page_title="Loan Safety — Train & Predict", layout="centered")
st.title("Loan Safety — Train & Predict")

with st.sidebar:
    st.header("Config")
    cfg_path = st.text_input("Config path", CFG_PATH_DEFAULT)
    try:
        cfg = load_cfg(cfg_path)
        st.caption("Config loaded ✅")
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        st.stop()

    csv_path = cfg["data"]["raw_csv"]
    target_name = cfg["data"]["target"]
    st.markdown(f"**Dataset:** `{csv_path}`")
    st.markdown(f"**Target:** `{target_name}`")

tab_train, tab_predict = st.tabs(["🛠️ Train", "🔮 Predict"])

with tab_train:
    st.subheader("Train a Decision Tree")
    st.write(
        "Uses your existing pipeline with `max_depth` from the config. Saves a model to `artifacts/`."
    )

    if st.button("Train model", type="primary"):
        with st.spinner("Training..."):
            try:
                model_path = train_fn(cfg_path)
                st.session_state["model_path"] = model_path
                st.success(f"Model saved: {model_path}")
                # quick whole-dataset metrics for a friendly number
                bundle = joblib.load(model_path)
                pipe = bundle["pipeline"]
                df = load_raw(csv_path)
                df, resolved_target = ensure_target(df, preferred_target=target_name)
                cat_cols, num_cols = get_feature_lists()
                keep = [c for c in cat_cols + num_cols + [resolved_target] if c in df.columns]
                df = df[keep].dropna()
                df = prepare_labels(df, target_col=resolved_target)
                X, y = split_X_y(df, resolved_target)
                acc = pipe.score(X, y)
                st.metric("Accuracy (full data, quick check)", f"{acc:.3f}")
                st.info("For proper validation metrics, use `python src/evaluate.py ...`")
            except Exception as e:
                st.error(f"Training failed: {e}")

with tab_predict:
    st.subheader("Make a Prediction")
    model_path = st.text_input(
        "Model path", st.session_state.get("model_path", choose_latest_model())
    )
    col_l, col_r = st.columns([1, 1])
    with col_l:
        if st.button("Use latest model"):
            latest = choose_latest_model()
            if latest:
                model_path = latest
                st.session_state["model_path"] = latest
                st.success(f"Selected latest: {latest}")
            else:
                st.warning("No model found in artifacts/")

    # Load choices for categorical widgets
    try:
        choices, cat_cols, num_cols = load_choices(csv_path, target_name)
    except Exception as e:
        st.error(f"Could not read dataset for form choices: {e}")
        st.stop()

    st.write("Fill the features below, then click **Predict**.")
    # Build input form using dataset-driven choices where possible
    with st.form("predict_form"):
        inputs = {}
        for c in cat_cols:
            if c in choices and len(choices[c]) > 0:
                default = choices[c][0]
                inputs[c] = st.selectbox(c, choices[c], index=0, key=f"cat_{c}")
            else:
                inputs[c] = st.text_input(c, "", key=f"cat_{c}")
        for c in num_cols:
            # numeric inputs as float; allow empty -> 0.0
            val = st.text_input(c, "", key=f"num_{c}", help="Numeric")
            try:
                inputs[c] = float(val) if val != "" else 0.0
            except:
                inputs[c] = 0.0

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        if not model_path or not Path(model_path).exists():
            st.error("Please provide a valid model path. Train a model first in the other tab.")
        else:
            try:
                bundle = joblib.load(model_path)
                pipe = bundle["pipeline"]
                df_in = pd.DataFrame([inputs])
                proba = pipe.predict_proba(df_in)[:, 1][0]
                pred = int(proba >= 0.5)
                label = "safe (1)" if pred == 1 else "risky (0)"
                st.metric("Prediction", label)
                st.metric("p(safe)", f"{proba:.3f}")
                st.code(json.dumps(inputs, indent=2), language="json")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
