
import os
import joblib
import pandas as pd

from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES


MODEL_DIR = "models"

AVAILABLE_MODELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "knn": "KNN",
}


def predict_sleep_quality(input_data, model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError("Please select a valid model.")

    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained models not found. Run train_models.py first."
        )

    model = joblib.load(model_path)

    # Keep feature names and their order consistent with training.
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    input_df = pd.DataFrame([input_data], columns=features)

    prediction = model.predict(input_df)[0]

    # Probability is optional; some classifiers may not provide it.
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

    return {
        "prediction": str(prediction),
        "confidence": confidence,
    }
