from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from collections import Counter
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

with open(synonym_path, "r") as f:
    SYMPTOM_SYNONYMS = json.load(f)

# Preprocessing
symptom_columns = [col for col in df.columns if col.lower().startswith("symptom")]
df["symptom_str"] = df[symptom_columns].apply(
    lambda x: " ".join([str(s).strip().lower() for s in x if pd.notna(s)]),
    axis=1
)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["symptom_str"])

# -----------------------------
# Weights and Mappings
# -----------------------------
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

SYMPTOM_SEVERITY = {
    "chest_pain": 1.0, "shortness_of_breath": 1.0, "altered_sensorium": 1.0,
    "vomiting": 0.7, "headache": 0.6, "fever": 0.6, "high_fever": 0.6,
    "fatigue": 0.4, "dizziness": 0.4, "nausea": 0.4, "itching": 0.2,
    "sneezing": 0.2, "runny_nose": 0.2, "cough": 0.3
}

SYMPTOM_CATEGORY = {
    "chest_pain": "cardio", "palpitations": "cardio", "shortness_of_breath": "respiratory",
    "cough": "respiratory", "congestion": "respiratory", "runny_nose": "respiratory",
    "nausea": "digestive", "vomiting": "digestive", "diarrhoea": "digestive",
    "itching": "skin", "skin_rash": "skin", "red_spots_over_body": "skin",
    "headache": "neuro", "dizziness": "neuro", "fatigue": "general",
    "fever": "general", "high_fever": "general"
}

DISEASE_CATEGORY = {
    "Heart attack": "cardio",
    "Tuberculosis": "respiratory",
    "Common Cold": "respiratory",
    "Dengue": "general",
    "Hepatitis A": "digestive",
    "Hepatitis D": "digestive",
    "Hepatitis B": "digestive",
    "GERD": "digestive",
    "Migraine": "neuro",
    "Chicken pox": "skin",
    "Allergy": "respiratory",
    "Hypothyroidism": "general",
    "Hyperthyroidism": "general",
    "Typhoid": "digestive",
    "Malaria": "general",
    "Pneumonia": "respiratory",
    "Paralysis (brain hemorrhage)": "neuro"
}

# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/symptom-info")
def symptom_info(request: SymptomRequest):
    input_symptoms = expand_symptoms(request.symptoms)
    input_str = " ".join(input_symptoms)
    user_vector = vectorizer.transform([input_str])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    severity_score = sum(SYMPTOM_SEVERITY.get(s, 0.3) for s in input_symptoms) / max(len(input_symptoms), 1)

    categories = [SYMPTOM_CATEGORY.get(s) for s in input_symptoms if SYMPTOM_CATEGORY.get(s)]
    most_common_category = Counter(categories).most_common(1)[0][0] if categories else None

    disease_scores = {}

    for idx, row in df.iterrows():
        disease_name = row["Disease"]
        disease_symptoms = row["symptom_str"].split()
        matched = set(input_symptoms) & set(disease_symptoms)
        matched_count = len(matched)

        if matched_count == 0 and similarities[idx] < 0.15:
            continue

        symptom_match_score = matched_count / len(input_symptoms)
        coverage_score = matched_count / len(disease_symptoms) if disease_symptoms else 0

        med_row = medications_df[medications_df["Disease"].str.lower() == disease_name.lower()]
        commonality = med_row.iloc[0]["Commonality"] if not med_row.empty else "Not classified"
        commonality_score = commonality_weights.get(commonality, 0.3)
        is_emergency = "severe" in commonality.lower()
        severity_boost = 0.15 if is_emergency else 0.0

        disease_cat = DISEASE_CATEGORY.get(disease_name)
        category_bonus = 0.05 if disease_cat == most_common_category else 0.0

        final_score = (
            0.3 * similarities[idx] +
            0.25 * symptom_match_score +
            0.1 * commonality_score +
            0.1 * severity_score +
            0.15 * coverage_score +
            category_bonus +
            severity_boost
        )

        if final_score < 0.25:
            continue

        precaution_row = precautions_df[precautions_df['Disease'].str.lower() == disease_name.lower()]
        precautions = precaution_row.iloc[0][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].dropna().tolist() if not precaution_row.empty else []
        medications = med_row.iloc[0]['Informational_Medications'] if not med_row.empty else "No medication info found."

        disease_scores[disease_name] = {
            "disease": disease_name,
            "commonality": commonality,
            "final_score": round(final_score, 3),
            "precautions": precautions,
            "informational_medications": medications,
            "debug_info": {
                "symptom_match_score": round(symptom_match_score, 3),
                "cosine_similarity": round(similarities[idx], 3),
                "coverage_score": round(coverage_score, 3),
                "severity_score": round(severity_score, 3),
                "commonality_score": round(commonality_score, 3),
                "category_bonus": round(category_bonus, 3),
                "severity_boost": round(severity_boost, 3)
            }
        }

    results = list(disease_scores.values())
    results.sort(key=lambda x: x["final_score"], reverse=True)

    if not results:
        return {
            "input_symptoms": request.symptoms,
            "likely_common_conditions": [],
            "other_possible_conditions": [],
            "recommendation_note": "No confident matches found. Consider consulting a doctor or double-checking your symptom terms.",
            "note": "If symptoms persist or worsen, please consult a healthcare provider.",
            "tip": "💡 Tip: You may try simplifying your input or using common phrasing like 'chest pain', 'high fever', or 'rash'."
        }

    if results[0]["final_score"] >= 0.7:
        likely_common_conditions = [results[0]]
        other_possible_conditions = results[1:4]
    else:
        likely_common_conditions = results[:2]
        other_possible_conditions = results[2:5]

    return {
        "input_symptoms": request.symptoms,
        "likely_common_conditions": likely_common_conditions,
        "other_possible_conditions": other_possible_conditions,
        "recommendation_note": "Based on your symptoms, here are the most likely conditions and other possibilities. This information is for educational purposes only and not a medical diagnosis.",
        "note": "If symptoms persist or worsen, please consult a healthcare provider.",
        "tip": "💡 Tip: Rest, hydrate, and avoid self-medicating. Use symptom logs to track recurring patterns for more accurate checkups."
    }
