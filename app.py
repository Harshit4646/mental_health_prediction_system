from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/mental_health_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(features)[0]

    return jsonify({
        "mental_health_treatment_needed": bool(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)
