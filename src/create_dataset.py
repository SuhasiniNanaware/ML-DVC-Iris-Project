import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

df.to_csv("data/dataset.csv", index=False)

print("Iris dataset created successfully!")
print("Shape:", df.shape)
print("Saved to: data/dataset.csv")
print("\nFirst 5 rows:")
print(df.head())