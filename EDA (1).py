# =====================================================================
# TASK 1: EXPLORATORY DATA ANALYSIS (EDA)
# Domain: Data Science
# Dataset: Titanic Passenger Survival
# =====================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style for modern-looking plots
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------
# 1. LOAD THE DATASET
# ---------------------------------------------------------------------
print("--- Loading Dataset ---")
# Loading directly from a reliable online repository
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display first few rows in the terminal
print("\nFirst 5 rows of the dataset:")
print(df.head())

# ---------------------------------------------------------------------
# 2. UNDERSTAND THE DATA STRUCTURE
# ---------------------------------------------------------------------
print("\n--- Data Information & Structure ---")
print(df.info())

print("\nMissing values per column before cleaning:")
print(df.isnull().sum())

# ---------------------------------------------------------------------
# 3. DATA CLEANING & IMPUTATION
# ---------------------------------------------------------------------
print("\n--- Cleaning Data ---")

# Fill missing Age values with the median age
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with the most common port (mode)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop the 'Cabin' column as it has too many missing values (>75%)
if "Cabin" in df.columns:
    df.drop(columns=["Cabin"], inplace=True)

print("Missing values after cleaning:")
print(df.isnull().sum())

# ---------------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------
print("\n--- Statistical Summary ---")
print(df.describe())

# ---------------------------------------------------------------------
# 5. DATA VISUALIZATION
# ---------------------------------------------------------------------
print("\n--- Generating Visualizations... ---")

# Figure 1: General Demographic & Survival Distributions
plt.figure(figsize=(15, 10))

# Plot 1: Survival Count
plt.subplot(2, 2, 1)
sns.countplot(data=df, x="Survived", palette="Set2")
plt.title("Distribution of Survival (0 = No, 1 = Yes)")
plt.xlabel("Survived")
plt.ylabel("Count")

# Plot 2: Survival Route by Gender
plt.subplot(2, 2, 2)
sns.countplot(data=df, x="Survived", hue="Sex", palette="Pastel1")
plt.title("Survival Count Split by Gender")
plt.xlabel("Survived")
plt.ylabel("Count")

# Plot 3: Survival Rate by Passenger Class
plt.subplot(2, 2, 3)
sns.countplot(data=df, x="Pclass", hue="Survived", palette="Set1")
plt.title("Survival Distribution by Passenger Class")
plt.xlabel("Ticket Class (1st, 2nd, 3rd)")
plt.ylabel("Count")

# Plot 4: Age Distribution of Passengers
plt.subplot(2, 2, 4)
sns.histplot(data=df, x="Age", kde=True, color="skyblue", bins=30)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()  # This opens the first visualization window

# Figure 2: Correlation Heatmap for Numerical Features
plt.figure(figsize=(8, 6))
numerical_df = df.select_dtypes(include=[np.number])
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()  # This opens the heatmap window

print("\n--- EDA Task Completed Successfully! ---")