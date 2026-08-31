import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data
test_data = pd.read_csv("data/test.csv")

X_test = test_data.drop("target", axis=1)
y_test = test_data["target"]

# Load trained model
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted"),
    "recall": recall_score(y_test, y_pred, average="weighted"),
    "f1_score": f1_score(y_test, y_pred, average="weighted")
}

# Create metrics directory
import os
os.makedirs("metrics", exist_ok=True)

# Save metrics
with open("metrics/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("=" * 40)
print("MODEL EVALUATION")
print("=" * 40)

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nMetrics saved to metrics/metrics.json")