from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
dataset_path = r"C:\Users\kligh\OneDrive\Desktop\BSDoc_Symptom_Search\BSDoc\tf-idf\dataset.csv"
df = pd.read_csv(dataset_path)

symptom_columns = [col for col in df.columns if col.startswith("Symptom_")]
df["symptom_str"] = df[symptom_columns].apply(
    lambda x: " ".join([symptom.strip().lower() for symptom in x if pd.notna(symptom)]),
    axis=1
)

precaution_path = r"C:\Users\kligh\OneDrive\Desktop\BSDoc_Symptom_Search\BSDoc\tf-idf\symptom_precaution.csv"
precautions_df = pd.read_csv(precaution_path)

medication_path = r"C:\Users\kligh\OneDrive\Desktop\BSDoc_Symptom_Search\BSDoc\tf-idf\disease_medication_with_commonality.csv"
medications_df = pd.read_csv(medication_path)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["symptom_str"])

commonality_weights = {
    "Very common": 1.0,
    "Common": 0.8,
    "Common chronic condition": 0.7,
    "Occasional": 0.5,
    "Less common": 0.4,
    "Rare": 0.2,
    "Severe / Emergency": 0.2,
    "Severe / Rare": 0.2,
    "Not classified": 0.3
}

app = FastAPI()

# Add CORS middleware so frontend requests won't be blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SymptomRequest(BaseModel):
    symptoms: list[str]

@app.post("/symptom-info")
def symptom_info(request: SymptomRequest):
    input_symptoms = [sym.lower().strip() for sym in request.symptoms]
    input_str = " ".join(input_symptoms)

    user_vector = vectorizer.transform([input_str])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    disease_scores = {}

    for idx, row in df.iterrows():
        disease_name = row["Disease"]
        disease_symptoms = row["symptom_str"].split()
        matched_count = len(set(input_symptoms) & set(disease_symptoms))
        symptom_match_score = matched_count / len(input_symptoms) if input_symptoms else 0

        med_row = medications_df[medications_df["Disease"].str.lower() == disease_name.lower()]
        commonality = med_row.iloc[0]["Commonality"] if not med_row.empty else "Not classified"
        commonality_score = commonality_weights.get(commonality, 0.3)

        final_score = (0.5 * similarities[idx]) + (0.3 * symptom_match_score) + (0.2 * commonality_score)

        precaution_row = precautions_df[precautions_df['Disease'].str.lower() == disease_name.lower()]
        precautions = precaution_row.iloc[0][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].dropna().tolist() if not precaution_row.empty else []

        medications = med_row.iloc[0]['Informational_Medications'] if not med_row.empty else "No medication info found."

        if disease_name not in disease_scores or disease_scores[disease_name]["final_score"] < final_score:
            disease_scores[disease_name] = {
                "disease": disease_name,
                "commonality": commonality,
                "final_score": final_score,
                "precautions": precautions,
                "informational_medications": medications
            }

    results = list(disease_scores.values())
    results.sort(key=lambda x: x["final_score"], reverse=True)

    likely_common_conditions = [r for r in results if commonality_weights.get(r["commonality"], 0) >= 0.7][:2]
    other_possible_conditions = [r for r in results if r not in likely_common_conditions][:3]

    return {
        "input_symptoms": request.symptoms,
        "recommendation_note": "Based on your symptoms, here are the most likely conditions and other possibilities. This information is for educational purposes only and not a medical diagnosis.",
        "likely_common_conditions": likely_common_conditions,
        "other_possible_conditions": other_possible_conditions,
        "note": "If symptoms persist or worsen, please consult a healthcare provider."
    }

# ✅ Run with:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
