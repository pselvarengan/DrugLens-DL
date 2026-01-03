# ============================================================
# DrugLens-DL : Prediction Module (FINAL & CORRECT)
# ============================================================

import os
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, QED

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
MODEL_PATH = os.path.join("models", "druglens_model.pkl")
SCALER_PATH = os.path.join("models", "druglens_scaler.pkl")

# ------------------------------------------------------------
# Load model and scaler
# ------------------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)  # trained on 8 descriptors ONLY

# ------------------------------------------------------------
# Descriptor calculation (8 features)
# ------------------------------------------------------------
def compute_descriptors(mol):
    return np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        QED.qed(mol),
    ]).reshape(1, -1)

# ------------------------------------------------------------
# ECFP4 fingerprint (1024 bits, UNCALED)
# ------------------------------------------------------------
def compute_ecfp4(mol, nBits=1024):
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=nBits
    )
    return np.array(fp).reshape(1, -1)

# ------------------------------------------------------------
# Lipinski Rule of Five
# ------------------------------------------------------------
def lipinski_rule(desc):
    mw, logp, tpsa, hbd, hba, rotb, rings, qed = desc.flatten()
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    return "Pass (0 violations)" if violations == 0 else f"Fail ({violations} violations)"

# ------------------------------------------------------------
# Veber Rule
# ------------------------------------------------------------
def veber_rule(desc):
    _, _, tpsa, _, _, rotb, _, _ = desc.flatten()
    return "Pass" if (rotb <= 10 and tpsa <= 140) else "Fail"

# ------------------------------------------------------------
# Main prediction function
# ------------------------------------------------------------
def predict_druglikeness(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    # Compute features
    desc = compute_descriptors(mol)      # (1, 8)
    desc_scaled = scaler.transform(desc) # scale ONLY descriptors
    fp = compute_ecfp4(mol)              # (1, 1024), unscaled

    # Concatenate → (1, 1032)
    X = np.hstack([fp, desc_scaled])

    # Predict
    prob = model.predict_proba(X)[0, 1]

    # Rule-based assessment
    lipinski_status = lipinski_rule(desc)
    veber_status = veber_rule(desc)

    # Final interpretation
    if prob >= 0.70:
        verdict = "Likely drug-like (High confidence)"
    elif prob >= 0.40:
        verdict = "Moderate drug-likeness (AI cautious)"
    else:
        verdict = "Likely non-drug-like"

    return {
        "smiles": smiles,
        "ai_probability": float(prob),
        "final_verdict": verdict,
        "lipinski_status": lipinski_status,
        "veber_status": veber_status,

        "mw": float(desc[0, 0]),
        "logp": float(desc[0, 1]),
        "tpsa": float(desc[0, 2]),
        "hbd": int(desc[0, 3]),
        "hba": int(desc[0, 4]),
        "rotb": int(desc[0, 5]),
        "rings": int(desc[0, 6]),
        "qed": float(desc[0, 7]),
    }

# ------------------------------------------------------------
# Wrapper for Streamlit
# ------------------------------------------------------------
def predict_smiles(smiles: str):
    return predict_druglikeness(smiles)
