# 🎓 Student Placement Predictor

A machine learning mini project that predicts whether a student is likely to
get placed based on academic and skill-related features.

## Features Used
- CGPA
- Attendance %
- Internships completed
- Projects completed
- Active backlogs
- Communication skill (self-rated)
- Coding score
- Extra-curricular involvement

## Project Structure
```
placement_predictor/
├── generate_dataset.py   # creates synthetic dataset -> data/placement_data.csv
├── eda.py                 # exploratory data analysis, saves plots to outputs/
├── train_model.py         # trains Logistic Regression + Random Forest, saves best model
├── app.py                 # Streamlit web app for live predictions
├── data/                  # generated dataset
├── model/                 # saved trained model (.pkl)
├── outputs/                # EDA plots + confusion matrix
└── requirements.txt
```

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate the dataset:
```bash
python generate_dataset.py
```

3. Run EDA (optional, saves plots to outputs/):
```bash
python eda.py
```

4. Train the model:
```bash
python train_model.py
```

5. Launch the web app:
```bash
streamlit run app.py
```

This opens a browser window where you can input student details and get a
live placement prediction with confidence score.

## Model
Trains both Logistic Regression and Random Forest, automatically picks the
better-performing model based on test accuracy (~85% on this dataset).

## Notes
- Dataset is synthetically generated for demo purposes. Swap in a real
  Kaggle placement dataset (same column names) for a real-world version —
  just replace `data/placement_data.csv`.
- Good talking points for viva: train/test split, feature scaling, why
  Logistic Regression vs Random Forest, confusion matrix, precision/recall.
