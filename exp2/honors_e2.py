# Experiment 2
# To Apply Various Data Analysis and Visualization Techniques on EHR

import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Extract the dataset

zip_file = "synthea_sample_data_csv_latest.zip"

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall("synthea_data")

print("Dataset extracted successfully.")

for root, dirs, files in os.walk("synthea_data"):
    for file in files:
        if file.endswith(".csv"):
            print(os.path.join(root, file))

# 2. Load important EHR datasets

patients = pd.read_csv("synthea_data/patients.csv")
conditions = pd.read_csv("synthea_data/conditions.csv")
observations = pd.read_csv("synthea_data/observations.csv")
encounters = pd.read_csv("synthea_data/encounters.csv")

print("\nDatasets loaded successfully.")


# 3. Display dataset information

print("\n--- Patients Dataset ---")
print(patients.shape)
print(patients.head())

print("\n--- Conditions Dataset ---")
print(conditions.shape)
print(conditions.head())

print("\n--- Observations Dataset ---")
print(observations.shape)
print(observations.head())

print("\n--- Encounters Dataset ---")
print(encounters.shape)
print(encounters.head())


# 4. Descriptive statistics

print("\n--- Patient Dataset Statistics ---")

print(
    patients[
        ["BIRTHDATE", "HEALTHCARE_EXPENSES",
         "HEALTHCARE_COVERAGE"]
    ].describe()
)


# 5. Gender analysis

gender_count = patients["GENDER"].value_counts()

print("\n--- Gender Distribution ---")
print(gender_count)


# Visualization: Gender distribution

plt.figure(figsize=(7, 5))

gender_count.plot(
    kind="bar"
)

plt.title("Patient Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 6. Race analysis

race_count = patients["RACE"].value_counts().head(10)

print("\n--- Top Race Categories ---")
print(race_count)


# Visualization: Race distribution

plt.figure(figsize=(9, 5))

race_count.plot(
    kind="bar"
)

plt.title("Top Patient Race Categories")
plt.xlabel("Race")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# 7. Calculate patient age

patients["BIRTHDATE"] = pd.to_datetime(
    patients["BIRTHDATE"],
    errors="coerce"
)

reference_date = pd.Timestamp("2021-11-01")

patients["AGE"] = (
    (reference_date - patients["BIRTHDATE"]).dt.days
    / 365.25
)

patients["AGE"] = patients["AGE"].round(1)


print("\n--- Patient Age ---")
print(
    patients[["Id", "BIRTHDATE", "AGE"]].head()
)


# 8. Age distribution

print("\n--- Age Statistics ---")

print(
    patients["AGE"].describe()
)


# Visualization: Age histogram

plt.figure(figsize=(8, 5))

plt.hist(
    patients["AGE"].dropna(),
    bins=15
)

plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()


# 9. Most common medical conditions

condition_count = (
    conditions["DESCRIPTION"]
    .value_counts()
    .head(10)
)

print("\n--- Top 10 Medical Conditions ---")
print(condition_count)


# Visualization: Medical conditions

plt.figure(figsize=(10, 6))

condition_count.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Most Common Medical Conditions")
plt.xlabel("Number of Records")
plt.ylabel("Medical Condition")
plt.tight_layout()
plt.show()


# 10. Encounter type analysis

encounter_count = encounters[
    "ENCOUNTERCLASS"
].value_counts()

print("\n--- Encounter Types ---")
print(encounter_count)


# Visualization: Encounter types

plt.figure(figsize=(8, 5))

encounter_count.plot(
    kind="bar"
)

plt.title("Distribution of Encounter Types")
plt.xlabel("Encounter Type")
plt.ylabel("Number of Encounters")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# 11. Healthcare expenses analysis

print("\n--- Healthcare Expenses Statistics ---")

print(
    patients["HEALTHCARE_EXPENSES"].describe()
)


# Visualization: Healthcare expenses

plt.figure(figsize=(8, 5))

plt.boxplot(
    patients["HEALTHCARE_EXPENSES"].dropna()
)

plt.title("Distribution of Healthcare Expenses")
plt.ylabel("Healthcare Expenses")
plt.tight_layout()
plt.show()


# 12. Healthcare expenses vs coverage

plt.figure(figsize=(8, 5))

plt.scatter(
    patients["HEALTHCARE_EXPENSES"],
    patients["HEALTHCARE_COVERAGE"],
    alpha=0.5
)

plt.title("Healthcare Expenses vs Healthcare Coverage")
plt.xlabel("Healthcare Expenses")
plt.ylabel("Healthcare Coverage")
plt.tight_layout()
plt.show()


# 13. Correlation analysis

numeric_data = patients[
    ["AGE",
     "HEALTHCARE_EXPENSES",
     "HEALTHCARE_COVERAGE"]
]

correlation = numeric_data.corr()

print("\n--- Correlation Matrix ---")
print(correlation)


# Visualization: Correlation heatmap

plt.figure(figsize=(7, 5))

plt.imshow(
    correlation,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# 14. Summary

print("\n========================================")
print("EHR DATA ANALYSIS COMPLETED")
print("========================================")

print("\nNumber of Patients:",
      len(patients))

print("Number of Condition Records:",
      len(conditions))

print("Number of Observations:",
      len(observations))

print("Number of Encounters:",
      len(encounters))

print("\nTop Medical Condition:")
print(condition_count.index[0])

print("\nMost Common Encounter Type:")
print(encounter_count.index[0])

print("\nMost Common Gender:")
print(gender_count.index[0])