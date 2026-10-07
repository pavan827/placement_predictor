"""
Exploratory Data Analysis for the placement dataset.
Saves plots to outputs/
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
sns.set_style("whitegrid")

df = pd.read_csv("data/placement_data.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary stats:\n", df.describe())

# Placement distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Placed", palette="Set2")
plt.title("Placement Distribution")
plt.xticks([0, 1], ["Not Placed", "Placed"])
plt.savefig("outputs/placement_distribution.png", bbox_inches="tight")
plt.close()

# Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png", bbox_inches="tight")
plt.close()

# CGPA vs Placement
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Placed", y="CGPA", palette="Set3")
plt.xticks([0, 1], ["Not Placed", "Placed"])
plt.title("CGPA vs Placement")
plt.savefig("outputs/cgpa_vs_placement.png", bbox_inches="tight")
plt.close()

print("\nEDA plots saved to outputs/")
