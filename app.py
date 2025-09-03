import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from data import load_raw, ensure_target, prepare_labels
from features import split_X_y

# Page config
st.set_page_config(page_title="Loan Risk Checker", page_icon="💰", layout="centered")

# Title
st.title("💰 Loan Risk Checker")
st.markdown("Check if a loan is **safe** or **risky** in seconds")

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.model_name = None

# Sidebar for model selection
with st.sidebar:
    st.header("Setup")

    # Find available models
    models_dir = Path("artifacts")
    if models_dir.exists():
        models = list(models_dir.glob("*.joblib"))
        if models:
            # Sort by newest first
            models.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Simple model selector
            model_names = [m.name for m in models]
            selected = st.selectbox("Select Model", model_names, help="Choose a trained model")

            if st.button("Load Model"):
                model_path = models_dir / selected
                bundle = joblib.load(model_path)
                st.session_state.model = bundle["pipeline"]
                st.session_state.model_name = selected
                st.success("✅ Model loaded!")
        else:
            st.error("No models found. Train one first!")
    else:
        st.error("No artifacts folder found!")

    if st.session_state.model:
        st.success(f"Using: {st.session_state.model_name}")

# Main interface
if st.session_state.model is None:
    st.info("👈 Please load a model from the sidebar first")
else:
    st.subheader("Enter Loan Details")

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        # Categorical inputs with simple defaults
        grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
        sub_grade = st.selectbox(
            "Sub-grade", [f"{grade}1", f"{grade}2", f"{grade}3", f"{grade}4", f"{grade}5"]
        )
        home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        purpose = st.selectbox(
            "Loan Purpose",
            [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "major_purchase",
                "medical",
                "small_business",
                "other",
            ],
        )
        term = st.selectbox("Term", ["36 months", "60 months"])

    with col2:
        # Numeric inputs with reasonable defaults
        emp_years = st.number_input("Years Employed", min_value=0, max_value=50, value=5)
        dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=100.0, value=15.0)
        revol_util = st.number_input(
            "Credit Utilization %", min_value=0.0, max_value=100.0, value=30.0
        )
        late_fees = st.number_input("Total Late Fees", min_value=0.0, value=0.0)

        # Binary flags (simplified)
        no_delinq = st.checkbox("No recent delinquencies", value=True)
        no_derog = st.checkbox("No derogatory marks", value=True)
        short_emp = st.checkbox("Employed < 1 year", value=False)

    # Predict button
    if st.button("🔍 Check Risk", type="primary", use_container_width=True):
        # Prepare input
        input_data = {
            "grade": grade,
            "sub_grade": sub_grade,
            "home_ownership": home,
            "purpose": purpose,
            "term": term,
            "emp_length_num": float(emp_years),
            "dti": float(dti),
            "revol_util": float(revol_util),
            "total_rec_late_fee": float(late_fees),
            "last_delinq_none": int(no_delinq),
            "last_major_derog_none": int(no_derog),
            "short_emp": int(short_emp),
        }

        # Make prediction
        df = pd.DataFrame([input_data])

        try:
            # Get prediction
            prob = st.session_state.model.predict_proba(df)[0][1]
            prediction = "SAFE" if prob >= 0.5 else "RISKY"

            # Display result with clear visual feedback
            st.markdown("---")

            if prediction == "SAFE":
                st.success(f"### ✅ {prediction} LOAN")
                st.markdown(f"**Confidence:** {prob:.1%} chance of being safe")
            else:
                st.error(f"### ⚠️ {prediction} LOAN")
                st.markdown(f"**Confidence:** {(1-prob):.1%} chance of being risky")

            # Simple risk meter
            st.markdown("**Risk Level:**")
            risk = 1 - prob
            if risk < 0.3:
                st.progress(risk, text="Low Risk")
            elif risk < 0.7:
                st.progress(risk, text="Medium Risk")
            else:
                st.progress(risk, text="High Risk")

        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>", unsafe_allow_html=True)
st.caption("Simple Loan Risk Assessment Tool")
st.markdown("</div>", unsafe_allow_html=True)
