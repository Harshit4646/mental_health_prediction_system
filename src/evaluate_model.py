import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/processed/final_data.csv")
X = df.drop('treatment', axis=1)
y = df['treatment']

model = joblib.load("models/mental_health_model.pkl")

predictions = model.predict(X)

print("Classification Report:\n")
print(classification_report(y, predictions))

print("Confusion Matrix:\n")
print(confusion_matrix(y, predictions))
