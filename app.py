from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# Initialize app
app = Flask(__name__)
CORS(app)

# Load model and columns
model = joblib.load("random_forest_model.pkl")
columns = joblib.load("input_columns.pkl")  # saved from training
selector = joblib.load("selector.pkl")

@app.route("/")
def home():
    return "Sleep Health Random Forest API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])  # all 8 columns
        X_selected = selector.transform(df)
        prediction = model.predict(X_selected)[0]
        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
