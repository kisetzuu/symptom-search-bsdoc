from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Load dataset
dataset_path = r"C:\Users\kulai\OneDrive\Desktop\BSDoc\sympton_prediction_model-i\dataset.csv"
df = pd.read_csv(dataset_path)

# Combine and normalize symptoms into a single list
symptom_columns = [col for col in df.columns if col.startswith("Symptom_")]
df["Symptoms"] = df[symptom_columns].apply(lambda x: [symptom.strip().lower() for symptom in x if pd.notna(symptom)], axis=1)

# FastAPI instance
app = FastAPI()

# Request model
class SymptomRequest(BaseModel):
    symptoms: list[str]

@app.post("/symptom-search")
def symptom_search(request: SymptomRequest):
    # Normalize input
    input_symptoms = [sym.lower().strip() for sym in request.symptoms]

    # Find diseases that match at least one symptom
    matched_diseases = []
    for _, row in df.iterrows():
        disease_symptoms = row["Symptoms"]
        matches = set(input_symptoms) & set(disease_symptoms)
        if matches:
            matched_diseases.append({
                "disease": row["Disease"],
                "matching_symptoms": list(matches)
            })

    if not matched_diseases:
        return {"message": "No common diseases found for the given symptoms."}

    return {
        "input_symptoms": input_symptoms,
        "possible_conditions": matched_diseases
    }

# Run using: uvicorn main:app --reload
