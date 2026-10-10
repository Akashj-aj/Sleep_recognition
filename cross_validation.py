
import os
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from preprocessing import prepare_data


def main():
    # Make sure the output directory exists.
    os.makedirs("models", exist_ok=True)

    # Load the dataset and preprocessing pipeline.
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    # Use only the training portion for cross-validation.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        ),
        "SVM": SVC(
            class_weight="balanced",
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    # Keep class proportions similar in all five folds.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    results = []

    for name, classifier in models.items():
        print(f"\nEvaluating {name}...")

        model = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier),
        ])

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        results.append({
            "Model": name,
            "Accuracy Mean": scores["test_accuracy"].mean(),
            "Accuracy Std": scores["test_accuracy"].std(),
            "Macro Precision": scores["test_precision_macro"].mean(),
            "Macro Recall": scores["test_recall_macro"].mean(),
            "Macro F1": scores["test_f1_macro"].mean(),
        })

    df = pd.DataFrame(results)

    # Display results as percentages.
    metric_columns = [
        "Accuracy Mean",
        "Accuracy Std",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
    ]

    df[metric_columns] = df[metric_columns] * 100

    print("\n" + "=" * 75)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 75)
    print(df.round(2).to_string(index=False))

    output_path = os.path.join(
        "models",
        "cross_validation_results.csv",
    )
    df.to_csv(output_path, index=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
