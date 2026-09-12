import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_PATH = "data/training_data.csv"
MODEL_PATH = "model/employment_model.pkl"


df = pd.read_csv(DATA_PATH)

X = df[
    [
        "attendance",
        "assessment",
        "practical",
        "completion",
        "experience",
    ]
]

y = df["employed"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=6,
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2%}")


os.makedirs("model", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")