import sys, json
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import yaml
import numpy as np
from datetime import datetime

# make 'src/' importable
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

# import project code
from train import train as train_fn
from data import load_raw, ensure_target, prepare_labels
from features import get_feature_lists, build_preprocessor, split_X_y
from evaluate import evaluate as evaluate_fn

CFG_PATH_DEFAULT = "configs/config.yaml"


@st.cache_data
def load_cfg(cfg_path: str):
    return yaml.safe_load(Path(cfg_path).read_text())


@st.cache_data
def load_choices(csv_path: str, target_preferred: str):
    df = load_raw(csv_path)
    df, target = ensure_target(df, preferred_target=target_preferred)
    cat_cols, num_cols = get_feature_lists()
    keep = [c for c in cat_cols + num_cols if c in df.columns]
    df_small = df[keep].dropna().head(10_000)  # limit for speed
    choices = {
        c: sorted(map(str, df_small[c].dropna().unique().tolist()))
        for c in cat_cols
        if c in df_small
    }
    return choices, cat_cols, num_cols


def get_available_models():
    """Get list of available models with metadata."""
    arts_dir = ROOT / "artifacts"
    if not arts_dir.exists():
        return []
    
    models = []
    for model_path in sorted(arts_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            bundle = joblib.load(model_path)
            model_type = bundle.get("model_type", "unknown")
            val_score = bundle.get("validation_score", None)
            
            # Parse timestamp from filename
            name_parts = model_path.stem.split('_')
            timestamp_str = name_parts[-1] if name_parts else ""
            
            models.append({
                "path": str(model_path),
                "name": model_path.name,
                "type": model_type,
                "val_score": val_score,
                "timestamp": timestamp_str,
                "size_mb": model_path.stat().st_size / (1024 * 1024)
            })
        except:
            continue
    
    return models


st.set_page_config(page_title="🎯 Loan Safety ML Platform", layout="wide", initial_sidebar_state="expanded")

# Header
st.title("🎯 Loan Safety ML Platform")
st.markdown("**Predict loan risk with Decision Trees and Random Forests**")

with st.sidebar:
    st.header("⚙️ Configuration")
    cfg_path = st.text_input("Config path", CFG_PATH_DEFAULT)
    try:
        cfg = load_cfg(cfg_path)
        st.success("Config loaded ✅")
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        st.stop()

    csv_path = cfg["data"]["raw_csv"]
    target_name = cfg["data"]["target"]
    
    st.markdown("### Dataset Info")
    st.markdown(f"**File:** `{csv_path}`")
    st.markdown(f"**Target:** `{target_name}`")
    st.markdown(f"**Test size:** {cfg['training']['test_size']*100:.0f}%")
    st.markdown(f"**Random seed:** {cfg['training']['random_seed']}")
    
    st.markdown("### Model Configuration")
    current_model_type = cfg.get("model", {}).get("type", "decision_tree")
    st.markdown(f"**Default type:** {current_model_type}")
    
    if current_model_type == "decision_tree":
        params = cfg["model"]["decision_tree"]
        st.markdown(f"**Max depth:** {params['max_depth']}")
    else:
        params = cfg["model"]["random_forest"]
        st.markdown(f"**Trees:** {params['n_estimators']}")
        st.markdown(f"**Max depth:** {params['max_depth']}")

# Main tabs
tab_train, tab_evaluate, tab_predict, tab_compare = st.tabs(
    ["🛠️ Train Models", "📊 Evaluate", "🔮 Predict", "🏆 Compare Models"]
)

with tab_train:
    st.subheader("Train Machine Learning Models")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Model Selection")
        model_choice = st.radio(
            "Choose model type:",
            ["decision_tree", "random_forest"],
            index=1 if current_model_type == "random_forest" else 0,
            format_func=lambda x: "🌲 Decision Tree" if x == "decision_tree" else "🌳 Random Forest"
        )
        
        # Show relevant parameters
        if model_choice == "decision_tree":
            st.markdown("**Decision Tree Parameters:**")
            dt_params = cfg["model"]["decision_tree"]
            st.code(f"max_depth: {dt_params['max_depth']}\n"
                   f"min_samples_split: {dt_params.get('min_samples_split', 2)}\n"
                   f"min_samples_leaf: {dt_params.get('min_samples_leaf', 1)}")
        else:
            st.markdown("**Random Forest Parameters:**")
            rf_params = cfg["model"]["random_forest"]
            st.code(f"n_estimators: {rf_params['n_estimators']}\n"
                   f"max_depth: {rf_params['max_depth']}\n"
                   f"max_features: {rf_params['max_features']}\n"
                   f"min_samples_split: {rf_params.get('min_samples_split', 5)}")
    
    with col2:
        st.markdown("### Training Options")
        use_cv = st.checkbox("Enable cross-validation", value=cfg["training"]["cross_validation"]["enabled"])
        if use_cv:
            cv_folds = st.slider("CV folds", 3, 10, cfg["training"]["cross_validation"]["folds"])
        
        st.markdown("### Start Training")
        if st.button(f"🚀 Train {model_choice.replace('_', ' ').title()}", type="primary"):
            with st.spinner(f"Training {model_choice}... This may take a moment for Random Forest."):
                try:
                    # Update config temporarily for CV
                    if use_cv:
                        cfg["training"]["cross_validation"]["enabled"] = True
                        cfg["training"]["cross_validation"]["folds"] = cv_folds if use_cv else 5
                    
                    # Save temporary config
                    temp_cfg = Path("configs/temp_config.yaml")
                    with open(temp_cfg, 'w') as f:
                        yaml.dump(cfg, f)
                    
                    # Train model
                    model_path = train_fn(str(temp_cfg), model_choice)
                    st.session_state["last_model_path"] = model_path
                    
                    # Load results
                    bundle = joblib.load(model_path)
                    
                    # Display results
                    st.success(f"✅ Model trained successfully!")
                    
                    col1_res, col2_res, col3_res = st.columns(3)
                    with col1_res:
                        st.metric("Model Type", model_choice.replace('_', ' ').title())
                    with col2_res:
                        st.metric("Training Accuracy", f"{bundle.get('training_score', 0):.3f}")
                    with col3_res:
                        st.metric("Validation Accuracy", f"{bundle.get('validation_score', 0):.3f}")
                    
                    st.info(f"Model saved to: `{model_path}`")
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")

with tab_evaluate:
    st.subheader("📊 Model Evaluation")
    
    models = get_available_models()
    
    if not models:
        st.warning("No trained models found. Please train a model first.")
    else:
        # Model selection
        model_names = [f"{m['type']} - {m['timestamp']} (Val: {m['val_score']:.3f})" 
                      if m['val_score'] else f"{m['type']} - {m['timestamp']}"
                      for m in models]
        
        selected_idx = st.selectbox("Select model to evaluate:", 
                                    range(len(models)),