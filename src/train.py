import pandas as pd
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Load training data
train_data = pd.read_csv("data/train.csv")

X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]

# Create model
model = RandomForestClassifier(
    n_estimators=params["model"]["n_estimators"],
    max_depth=params["model"]["max_depth"],
    random_state=params["train"]["random_state"]
)

# Train
model.fit(X_train, y_train)

# Save model
with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model training completed successfully.")