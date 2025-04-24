import pandas as pd

# Load your dataset
df = pd.read_csv("data/dataset.csv")

# Get all symptom columns (e.g., Symptom_1 to Symptom_n)
symptom_columns = [col for col in df.columns if col.lower().startswith("symptom")]

# Flatten all symptom values into one list
all_symptoms = []
for col in symptom_columns:
    all_symptoms.extend(df[col].dropna().str.strip().str.lower())

# Get unique symptoms
unique_symptoms = sorted(set(all_symptoms))

# Print them
print(f"🧠 Total unique symptoms: {len(unique_symptoms)}")
for symptom in unique_symptoms:
    print("-", symptom)

# Get unique diseases
unique_diseases = sorted(df['Disease'].dropna().str.strip().str.lower().unique())

# Print them
print(f"🩺 Total unique diseases: {len(unique_diseases)}")
for disease in unique_diseases:
    print("-", disease)
