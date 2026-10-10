
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    results = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_macro": round(
            float(precision_score(
                y_test, predictions, average="macro", zero_division=0
            )), 4
        ),
        "recall_macro": round(
            float(recall_score(
                y_test, predictions, average="macro", zero_division=0
            )), 4
        ),
        "f1_macro": round(
            float(f1_score(
                y_test, predictions, average="macro", zero_division=0
            )), 4
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=["Poor", "Moderate", "Good"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=["Poor", "Moderate", "Good"],
        ).tolist(),
    }

    return results
