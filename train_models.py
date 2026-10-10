
import json
import os
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from preprocessing import prepare_data
from evaluation import evaluate_model


MODEL_DIR = "models"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        ),
        "svm": SVC(
            class_weight="balanced",
            random_state=42,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }

    all_results = {}

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        # Each model gets its own copy of the preprocessing steps.
        model = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier),
        ])

        model.fit(X_train, y_train)

        results = evaluate_model(model, X_test, y_test)
        all_results[name] = results

        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)

        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['f1_macro']:.4f}")
        print(f"Saved model: {model_path}")

    # Save the evaluation results for comparison on the website.
    with open(
        os.path.join(MODEL_DIR, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(all_results, file, indent=4)

    print("\nTraining complete.")
    print("All models and metrics are saved in the models/ folder.")


if __name__ == "__main__":
    main()
