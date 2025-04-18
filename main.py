from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import difflib

# -----------------------------
# File paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "data", "dataset.csv")
precaution_path = os.path.join(BASE_DIR, "data", "symptom_precaution.csv")
medication_path = os.path.join(BASE_DIR, "data", "disease_medication_with_commonality.csv")
synonym_path = os.path.join(BASE_DIR, "data", "symptom_synonyms.json")

# -----------------------------
# Load datasets
# -----------------------------
df = pd.read_csv(dataset_path)
precautions_df = pd.read_csv(precaution_path)
medications_df = pd.read_csv(medication_path)

# Load symptom synonyms
with open(synonym_path, "r") as f:
    SYMPTOM_SYNONYMS = json.load(f)

# Extended mappings
SYMPTOM_SYNONYMS.update({
    "head_pain": "headache",
    "vision_problems": "blurred_vision",
    "ice_craving": "pica",
    "joint_stiffness": "stiff_joints",
    "morning_discomfort": "joint_pain",
    "tingly_spine": "tingling",
    "blue_skin": "cyanosis"
})

# -----------------------------
# Weighting for symptom frequency
# -----------------------------
SYMPTOM_WEIGHTS = {
    "fatigue": 0.6,
    "fever": 0.6,
    "headache": 0.6,
    "nausea": 0.7,
    "cyanosis": 1.2,
    "butterfly_rash": 1.3,
    "joint_pain": 1.0,
    "shortness_of_breath": 1.1,
    "skin_rash": 0.9
}

commonality_weights = {
    "Very common": 1.0,
    "Common": 0.8,
    "Common chronic condition": 0.9,
    "Occasional": 0.5,
    "Less common": 0.4,
    "Rare": 0.2,
    "Severe / Emergency": 0.2,
    "Severe / Rare": 0.2,
    "Not classified": 0.3
}

# -----------------------------
# Build TF-IDF matrix
# -----------------------------
symptom_columns = [col for col in df.columns if col.lower().startswith("symptom")]
df["symptom_str"] = df[symptom_columns].apply(
    lambda x: " ".join([symptom.strip().lower() for symptom in x if pd.notna(symptom)]),
    axis=1
)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["symptom_str"])

# -----------------------------
# App setup
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Input + Normalization
# -----------------------------
def normalize(symptom: str):
    return re.sub(r'[^a-z_ ]+', '', symptom.lower().strip()).replace(" ", "_")

def expand_symptoms(symptoms: list[str]):
    expanded = set()
    known = set(SYMPTOM_SYNONYMS.keys())
    for s in symptoms:
        norm = normalize(s)
        mapped = SYMPTOM_SYNONYMS.get(norm)
        if not mapped:
            closest = difflib.get_close_matches(norm, known, n=1, cutoff=0.85)
            mapped = SYMPTOM_SYNONYMS.get(closest[0], norm) if closest else norm
        expanded.add(mapped)
    return list(expanded)

class SymptomRequest(BaseModel):
    symptoms: list[str]

# -----------------------------
# Endpoint
# -----------------------------
@app.post("/symptom-info")
def symptom_info(request: SymptomRequest):
    input_symptoms = expand_symptoms(request.symptoms)
    input_str = " ".join(input_symptoms)

    user_vector = vectorizer.transform([input_str])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    disease_scores = {}
    matched_symptoms = set()

    for idx, row in df.iterrows():
        disease = row["Disease"]
        disease_symptoms = row["symptom_str"].split()
        overlap = set(input_symptoms) & set(disease_symptoms)
        matched_count = len(overlap)

        # Weighted symptom matching
        weighted_score = sum(SYMPTOM_WEIGHTS.get(sym, 1.0) for sym in overlap)
        symptom_match_score = weighted_score / len(input_symptoms) if input_symptoms else 0

        if symptom_match_score < 0.3 and similarities[idx] < 0.2:
            continue

        # Commonality + severity bonus
        med_row = medications_df[medications_df["Disease"].str.lower() == disease.lower()]
        commonality = med_row.iloc[0]["Commonality"] if not med_row.empty else "Not classified"
        commonality_score = commonality_weights.get(commonality, 0.3)
        emergency_boost = 0.15 if "severe" in commonality.lower() else 0.0

        # Bonus for clusters
        cluster_bonus = 0.05 if {"runny_nose", "continuous_sneezing"} <= overlap else 0.0

        final_score = (
            0.35 * similarities[idx] +
            0.4 * symptom_match_score +
            0.15 * commonality_score +
            emergency_boost +
            cluster_bonus
        )

        # Precautions + Meds
        precaution_row = precautions_df[precautions_df['Disease'].str.lower() == disease.lower()]
        precautions = precaution_row.iloc[0][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].dropna().tolist() if not precaution_row.empty else []
        medications = med_row.iloc[0]['Informational_Medications'] if not med_row.empty else "No medication info found."

        if disease not in disease_scores or disease_scores[disease]["final_score"] < final_score:
            disease_scores[disease] = {
                "disease": disease,
                "commonality": commonality,
                "final_score": final_score,
                "precautions": precautions,
                "informational_medications": medications
            }

        matched_symptoms.update(overlap)

    results = sorted(disease_scores.values(), key=lambda x: x["final_score"], reverse=True)
    unmatched = list(set(input_symptoms) - matched_symptoms)

    top_score = results[0]["final_score"] if results else 0
    matched_ratio = len(matched_symptoms) / len(input_symptoms) if input_symptoms else 0
    fallback = not results or (top_score < 0.35 and matched_ratio < 0.4)

    if fallback:
        return {
            "input_symptoms": request.symptoms,
            "matched_symptoms": list(matched_symptoms),
            "unmatched_symptoms": unmatched,
            "fallback": True,
            "message": "Your symptoms did not strongly match any condition in our database. This may be due to rare symptoms or conditions not included. Please consult a healthcare provider."
        }

    return {
        "input_symptoms": request.symptoms,
        "recommendation_note": "Based on your symptoms, here are the most likely conditions and other possibilities. This information is for educational purposes only and not a medical diagnosis.",
        "likely_common_conditions": results[:2],
        "other_possible_conditions": results[2:5],
        "note": "If symptoms persist or worsen, please consult a healthcare provider.",
        "matched_symptoms": list(matched_symptoms),
        "unmatched_symptoms": unmatched,
        "symptom_coverage": f"{len(matched_symptoms)}/{len(input_symptoms)} matched"
    }
