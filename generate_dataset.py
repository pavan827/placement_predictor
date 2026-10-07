"""
Generates a synthetic student placement dataset.
Run this first to create data/placement_data.csv
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 600

cgpa = np.round(np.random.normal(7.2, 1.0, n).clip(5.0, 10.0), 2)
attendance = np.round(np.random.normal(78, 10, n).clip(50, 100), 1)
internships = np.random.poisson(1.1, n).clip(0, 4)
projects = np.random.poisson(2.0, n).clip(0, 6)
backlogs = np.random.poisson(0.4, n).clip(0, 5)
communication_skill = np.random.randint(1, 11, n)  # self-rated 1-10
coding_score = np.round(np.random.normal(60, 20, n).clip(0, 100), 1)  # e.g. HackerRank/LeetCode style score
extra_curricular = np.random.randint(0, 2, n)  # 0 = no, 1 = yes

# Placement probability driven by a weighted combination of features
score = (
    cgpa * 1.4
    + attendance * 0.03
    + internships * 1.8
    + projects * 0.9
    + communication_skill * 0.6
    + coding_score * 0.05
    + extra_curricular * 0.8
    - backlogs * 2.2
)

prob = 1 / (1 + np.exp(-(score - score.mean()) / score.std()))
placed = (prob > np.random.uniform(0.3, 0.7, n)).astype(int)

df = pd.DataFrame({
    "CGPA": cgpa,
    "Attendance": attendance,
    "Internships": internships,
    "Projects": projects,
    "Backlogs": backlogs,
    "Communication_Skill": communication_skill,
    "Coding_Score": coding_score,
    "Extra_Curricular": extra_curricular,
    "Placed": placed
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/placement_data.csv", index=False)
print(f"Dataset generated: data/placement_data.csv ({len(df)} rows)")
print(f"Placement rate: {df['Placed'].mean()*100:.1f}%")
