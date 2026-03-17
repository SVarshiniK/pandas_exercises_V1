import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset (intentional issue: no file handling)
data = pd.read_csv("data.csv")

# Features & target (intentional issue: no validation)
X = data.drop("target", axis=1)
y = data["target"]

# Model (intentional issue: no scaling, no split)
model = LogisticRegression()

# Fit model
model.fit(X, y)

# Prediction (intentional issue: predicting on same data)
preds = model.predict(X)

# Accuracy (intentional issue: wrong logic)
accuracy = sum(preds == y) / len(y)

print("Accuracy:", accuracy)
