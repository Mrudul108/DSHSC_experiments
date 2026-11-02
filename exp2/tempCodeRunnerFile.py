
# Experiment 2: EHR Data Analysis and Visualization
# CO2: To apply various data analysis and visualization techniques

import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Extract dataset
with zipfile.ZipFile("synthea_sample_data_csv_latest.zip", "r") as z:
    z.extractall("synthea_data")

print("Dataset extracted successfully.")

# 2. Load datasets
path = "synthea_data"
patients = pd.read_csv(os.path.join(path, "patients.csv"))
conditions = pd.read_csv(os.path.join(path, "conditions.csv"))
observations = pd.read_csv(os.path.join(path, "observations.csv"))
encounters = pd.read_csv(os.path.join(path, "encounters.csv"))

print("Datasets loaded successfully.")

# 3. Display dataset information
for name, df in {"Patients": patients, "Conditions": conditions,
                 "Observations": observations, "Encounters": encounters}.items():
    print(f"\n--- {name} ---")
    print("Shape:", df.shape)
    print(df.head())

# 4. Descriptive statistics
print("\nPatient Statistics:")
print(patients[["HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"]].describe())

# 5. Gender and Race analysis
gender = patients["GENDER"].value_counts()
race = patients["RACE"].value_counts().head(10)

print("\nGender Distribution:\n", gender)
print("\nRace Distribution:\n", race)

# 6. Calculate patient age
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
patients["AGE"] = ((pd.Timestamp("2021-11-01") - patients["BIRTHDATE"]).dt.days / 365.25).round(1)

print("\nAge Statistics:\n", patients["AGE"].describe())

# 7. Common medical conditions and encounter types
condition_count = conditions["DESCRIPTION"].value_counts().head(10)
encounter_count = encounters["ENCOUNTERCLASS"].value_counts()

print("\nTop Medical Conditions:\n", condition_count)
print("\nEncounter Types:\n", encounter_count)

# 8. Healthcare expenses statistics
print("\nHealthcare Expenses:\n", patients["HEALTHCARE_EXPENSES"].describe())

# 9. Correlation analysis
correlation = patients[["AGE", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"]].corr()
print("\nCorrelation Matrix:\n", correlation)

# 10. Visualizations
plots = [
    (gender, "bar", "Gender Distribution", "Gender", "Patients"),
    (race, "bar", "Race Distribution", "Race", "Patients"),
    (patients["AGE"].dropna(), "hist", "Age Distribution", "Age", "Patients"),
    (condition_count.sort_values(), "barh", "Top Medical Conditions", "Records", "Condition"),
    (encounter_count, "bar", "Encounter Types", "Type", "Encounters"),
]

for data, kind, title, xlabel, ylabel in plots:
    plt.figure(figsize=(8, 5))
    if kind == "hist":
        plt.hist(data, bins=15)
    else:
        data.plot(kind=kind)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# 11. Healthcare expenses visualizations
plt.figure(figsize=(7, 5))
plt.boxplot(patients["HEALTHCARE_EXPENSES"].dropna())
plt.title("Healthcare Expenses Distribution")
plt.ylabel("Expenses")
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(patients["HEALTHCARE_EXPENSES"], patients["HEALTHCARE_COVERAGE"])
plt.title("Healthcare Expenses vs Coverage")
plt.xlabel("Expenses")
plt.ylabel("Coverage")
plt.tight_layout()
plt.show()

# 12. Correlation heatmap
plt.figure(figsize=(7, 5))
plt.imshow(correlation, interpolation="nearest")
plt.colorbar()
plt.xticks(range(3), correlation.columns, rotation=45, ha="right")
plt.yticks(range(3), correlation.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 13. Summary
print("\n===== EHR DATA ANALYSIS COMPLETED =====")
print("Total Patients:", len(patients))
print("Total Conditions:", len(conditions))
print("Total Observations:", len(observations))
print("Total Encounters:", len(encounters))
print("Most Common Condition:", condition_count.index[0])
print("Most Common Encounter:", encounter_count.index[0])
print("Most Common Gender:", gender.index[0])
