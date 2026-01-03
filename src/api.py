from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_druglikeness

app = FastAPI(
    title="DrugLens-DL API",
    description="AI + Rule-based Drug-likeness Prediction",
    version="1.1"
)

class Molecule(BaseModel):
    smiles: str

@app.post("/predict")
def predict(mol: Molecule):
    result = predict_druglikeness(mol.smiles)

    if result is None:
        return {"error": "Invalid SMILES"}

    return {
        "smiles": mol.smiles,
        "ai_prediction": result["ai_prediction"],
        "ai_probability": result["ai_probability"],
        "lipinski": result["lipinski"],
        "veber": result["veber"]
    }
