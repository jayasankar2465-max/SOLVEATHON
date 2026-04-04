from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = joblib.load("waste_model.pkl")
model_columns = joblib.load("model_columns.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([data])

    # Align columns
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)[0]

    return jsonify({
        "predicted_waste": prediction
    })

@app.route("/")
def home():
    return "Waste Prediction API is running"

if __name__ == "__main__":
    app.run(debug=True)
