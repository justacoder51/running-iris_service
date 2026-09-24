from pathlib import Path
import pickle

from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

app = Flask(__name__)
CORS(app)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl is missing. Run `python train.py` before starting the service."
        )
    with MODEL_PATH.open("rb") as model_file:
        bundle = pickle.load(model_file)
    return bundle["model"], bundle["target_names"]


model, target_names = load_model()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "iris-classifier"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    feature_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    try:
        features = [[float(payload[name]) for name in feature_names]]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide four numeric iris measurements."}), 400

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    return jsonify({
        "prediction": target_names[prediction],
        "confidence": round(confidence, 4),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
