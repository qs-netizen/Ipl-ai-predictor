import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# =========================
# CREATE MODELS FOLDER
# =========================

os.makedirs("models", exist_ok=True)

# =========================
# LOAD DATASET
# =========================

matches = pd.read_csv("data/matches.csv")

print("Dataset Loaded Successfully!")

# =========================
# KEEP IMPORTANT COLUMNS
# =========================

matches = matches[[
    'team1',
    'team2',
    'toss_winner',
    'venue',
    'winner'
]]

# =========================
# REMOVE NULL VALUES
# =========================

matches.dropna(inplace=True)

print("Null Values Removed!")

# =========================
# ENCODE CATEGORICAL DATA
# =========================

encoders = {}

for column in matches.columns:
    le = LabelEncoder()

    matches[column] = le.fit_transform(matches[column])

    encoders[column] = le

print("Encoding Completed!")

# =========================
# FEATURES & TARGET
# =========================

X = matches[[
    'team1',
    'team2',
    'toss_winner',
    'venue'
]]

y = matches['winner']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train/Test Split Done!")

# =========================
# XGBOOST MODEL
# =========================

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("Training Model...")

# =========================
# TRAIN MODEL
# =========================

model.fit(X_train, y_train)

print("Model Training Completed!")

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_test)

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "models/ipl_model.pkl")

# =========================
# SAVE ENCODERS
# =========================

joblib.dump(encoders, "models/encoders.pkl")

print("\nModel Saved Successfully!")

print("\nFiles Created:")
print("models/ipl_model.pkl")
print("models/encoders.pkl")
