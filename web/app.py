# ============================================================
# DrugLens-DL : Streamlit Web Application
# ============================================================

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
from src.predict import predict_smiles

# ------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------
if "smiles_input" not in st.session_state:
    st.session_state.smiles_input = ""

if "uploader_reset_id" not in st.session_state:
    st.session_state.uploader_reset_id = 0

# ------------------------------------------------------------
# Clear / Reset callback (STREAMLIT-CORRECT)
# ------------------------------------------------------------
def clear_inputs():
    st.session_state.smiles_input = ""
    st.session_state.uploader_reset_id += 1

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="DrugLens-DL",
    page_icon="🧪",
    layout="centered"
)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("🧪 DrugLens-DL")
st.markdown(
    """
**Explainable AI Platform for Drug-Likeness Prediction**

DrugLens-DL integrates molecular fingerprints, physicochemical descriptors,
rule-based screening, and interpretable machine learning to support
early-stage drug discovery.
"""
)

st.divider()

# ------------------------------------------------------------
# Input Section
# ------------------------------------------------------------
st.subheader("Molecular Input")

input_mode = st.radio(
    "Select input mode",
    ["Single SMILES", "Batch SMILES (CSV upload)"],
    horizontal=True
)

smiles_list = []

if input_mode == "Single SMILES":
    smiles = st.text_input(
        "Enter SMILES notation",
        placeholder="CC(=O)NC1=CC=C(C=C1)O",
        key="smiles_input"
    )
    if smiles.strip():
        smiles_list = [smiles.strip()]

else:
    uploaded_file = st.file_uploader(
        "Upload CSV file with a column named 'smiles'",
        type=["csv"],
        key=f"uploaded_file_{st.session_state.uploader_reset_id}"
    )

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        if "smiles" not in df_upload.columns:
            st.error("CSV must contain a column named 'smiles'")
        else:
            smiles_list = df_upload["smiles"].dropna().tolist()
            st.success(f"Loaded {len(smiles_list)} molecules")

# ------------------------------------------------------------
# Action Buttons
# ------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    run_prediction = st.button("🔍 Predict Drug-Likeness")

with col2:
    st.button("🧹 Clear / Reset", on_click=clear_inputs)

# ------------------------------------------------------------
# Prediction Section
# ------------------------------------------------------------
if run_prediction and smiles_list:

    results = []

    with st.spinner("Running DrugLens-DL analysis..."):
        for smi in smiles_list:
            try:
                results.append(predict_smiles(smi))
            except Exception as e:
                results.append({"smiles": smi, "error": str(e)})

    results_df = pd.DataFrame(results)

    st.divider()
    st.subheader("Prediction Results")

    if len(results_df) == 1 and "error" not in results_df.columns:

        r = results_df.iloc[0]

        st.markdown("### AI-Based Prediction")
        st.metric("Drug-Likeness Probability", f"{r['ai_probability']:.3f}")

        st.markdown("### Rule-Based Assessment (Advisory)")
        st.markdown(
            f"""
- **Lipinski Rule of Five:** {r['lipinski_status']}
- **Veber Rule:** {r['veber_status']}
"""
        )

        st.markdown("### Final Interpretation")
        st.success(r["final_verdict"])

        st.markdown("### Molecular Properties")

        prop_df = pd.DataFrame({
            "Property": [
                "Molecular Weight",
                "LogP",
                "Topological Polar Surface Area (TPSA)",
                "Hydrogen Bond Donors (HBD)",
                "Hydrogen Bond Acceptors (HBA)",
                "Rotatable Bonds",
                "Aromatic Rings",
                "Quantitative Estimate of Drug-likeness (QED)"
            ],
            "Value": [
                f"{r['mw']:.2f}",
                f"{r['logp']:.2f}",
                f"{r['tpsa']:.2f}",
                r["hbd"],
                r["hba"],
                r["rotb"],
                r["rings"],
                f"{r['qed']:.2f}"
            ],
            "Unit": [
                "g/mol", "—", "Å²", "count",
                "count", "count", "count", "—"
            ]
        })

        st.table(prop_df)

    else:
        st.dataframe(results_df, width="stretch")
        st.download_button(
            "⬇ Download Batch Results (CSV)",
            results_df.to_csv(index=False).encode("utf-8"),
            "druglens_batch_results.csv",
            "text/csv"
        )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()
st.markdown(
    """
**DrugLens-DL** is an academic research tool intended for scientific use only.  
Predictions should not be interpreted as clinical or regulatory decisions.

**Developer & Correspondence**  
Dr. P. Selvarengan  
📧 psrengan@hotmail.com
"""
)
