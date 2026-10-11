import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

from prediction import AVAILABLE_MODELS, predict_sleep_quality
from preprocessing import CATEGORIES, NUMERIC_FEATURES, load_and_clean_data

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Sleep_health_and_lifestyle_dataset.csv"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
CV_RESULTS_PATH = BASE_DIR / "models" / "cross_validation_results.csv"
MODEL_DIR = BASE_DIR / "models"

OPTIONS = {
    "genders": ["Male", "Female"],
    "occupations": [
        "Accountant",
        "Doctor",
        "Engineer",
        "Lawyer",
        "Manager",
        "Nurse",
        "Sales Representative",
        "Salesperson",
        "Scientist",
        "Software Engineer",
        "Teacher",
    ],
    "bmi_categories": ["Normal", "Normal Weight", "Overweight", "Obese"],
    "sleep_disorders": ["None", "Insomnia", "Sleep Apnea"],
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": {
        "category": "Linear classification",
        "how": "It gives each input a weight, adds the weighted values, and estimates which sleep category is most likely.",
        "simple": "Fast and relatively easy to interpret.",
    },
    "Decision Tree": {
        "category": "Rule-based classification",
        "how": "It asks a sequence of yes-or-no questions, such as whether a value is above a threshold, until it reaches a category.",
        "simple": "Easy to follow because its decisions look like if-then rules.",
    },
    "Random Forest": {
        "category": "Ensemble tree classification",
        "how": "It builds many decision trees and combines their votes to choose the final sleep-quality category.",
        "simple": "Usually more stable than relying on one decision tree.",
    },
    "SVM": {
        "category": "Margin-based classification",
        "how": "It finds boundaries in the transformed feature space that keep the different sleep categories as separate as possible.",
        "simple": "Useful when categories can be separated by clear boundaries.",
    },
    "KNN": {
        "category": "Neighborhood-based classification",
        "how": "It finds the most similar records in the training data and uses their most common category for the new record.",
        "simple": "Its prediction depends on nearby examples and feature scaling.",
    },
}

PRESETS = {
    "A": {
        "age": 25,
        "gender": "Male",
        "occupation": "Software Engineer",
        "sleep_duration": 7.5,
        "physical_activity": 45,
        "stress_level": 3,
        "bmi_category": "Normal",
        "sleep_disorder": "None",
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "heart_rate": 72,
        "daily_steps": 7000,
        "model": "random_forest",
    },
    "B": {
        "age": 42,
        "gender": "Female",
        "occupation": "Nurse",
        "sleep_duration": 6.1,
        "physical_activity": 30,
        "stress_level": 8,
        "bmi_category": "Overweight",
        "sleep_disorder": "Insomnia",
        "systolic_bp": 135,
        "diastolic_bp": 88,
        "heart_rate": 78,
        "daily_steps": 4500,
        "model": "logistic_regression",
    },
    "C": {
        "age": 50,
        "gender": "Male",
        "occupation": "Doctor",
        "sleep_duration": 5.9,
        "physical_activity": 60,
        "stress_level": 7,
        "bmi_category": "Obese",
        "sleep_disorder": "Sleep Apnea",
        "systolic_bp": 140,
        "diastolic_bp": 90,
        "heart_rate": 85,
        "daily_steps": 3000,
        "model": "svm",
    },
}


def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def load_cv_results():
    if not CV_RESULTS_PATH.exists():
        return []
    return pd.read_csv(CV_RESULTS_PATH).to_dict(orient="records")


def load_dataset():
    raw = pd.read_csv(DATA_PATH)
    prepared = load_and_clean_data()
    prepared["Sleep_Category"] = prepared["Sleep_Category"].astype(str)
    return raw, prepared


def dashboard_context():
    raw, df = load_dataset()
    metrics = load_metrics()
    cv_results = load_cv_results()
    distribution = (
        df["Sleep_Category"]
        .value_counts()
        .reindex(CATEGORIES, fill_value=0)
        .to_dict()
    )
    numeric_summary = df[NUMERIC_FEATURES].describe().round(2).to_dict()
    categorical_counts = {
        column: df[column].fillna("None").value_counts().to_dict()
        for column in ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]
    }
    missing = raw.isna().sum().to_dict()
    missing["Sleep Disorder"] = int(df["Sleep Disorder"].eq("None").sum())
    best = max(
        cv_results, key=lambda row: row.get("Accuracy Mean", 0), default=None
    )

    # Sleep duration distribution bins
    sleep_dur_counts = (
        pd.cut(
            df["Sleep Duration"],
            bins=[0, 6, 7, 8, 24],
            labels=["< 6 hrs", "6-7 hrs", "7-8 hrs", "> 8 hrs"],
        )
        .value_counts()
        .to_dict()
    )
    stress_counts = df["Stress Level"].value_counts().sort_index().to_dict()

    return {
        "raw": raw,
        "df": df,
        "metrics": metrics,
        "cv_results": cv_results,
        "distribution": distribution,
        "numeric_summary": numeric_summary,
        "categorical_counts": categorical_counts,
        "missing": missing,
        "best_cv": best,
        "sleep_dur_counts": sleep_dur_counts,
        "stress_counts": stress_counts,
    }


def form_options():
    return {**OPTIONS, "models": AVAILABLE_MODELS}


def get_feature_importance(model_name):
    model = joblib.load(MODEL_DIR / f"{model_name}.pkl")
    classifier = model.named_steps.get("classifier")
    transformer = model.named_steps.get("preprocessor")
    feature_names = transformer.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        method = "Split-based importance"
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_).mean(axis=0)
        method = "Mean absolute coefficient"
    else:
        return [], "This model does not expose native feature-importance values."

    importance = sorted(
        [
            {
                "feature": name.replace("numeric__", "").replace("categorical__", ""),
                "importance": float(value),
            }
            for name, value in zip(feature_names, values)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[:15]
    return importance, method


def validate_form():
    values = request.form.to_dict()
    model_name = values.get("model", "random_forest")
    if model_name not in AVAILABLE_MODELS:
        raise ValueError("Please select a valid prediction model.")

    def number(name, label):
        val_str = values.get(name, "").strip()
        if not val_str:
            raise ValueError(f"Please provide a value for {label}.")
        try:
            value = float(val_str)
        except ValueError:
            raise ValueError(f"{label} must be a valid number.")
        if not pd.notna(value):
            raise ValueError(f"{label} must be a valid number.")
        return value

    age = number("age", "Age")
    sleep_duration = number("sleep_duration", "Hours of Sleep per Day")
    physical_activity = number("physical_activity", "Physical Activity Level")
    stress_level = number("stress_level", "Perceived Stress Level")
    systolic_bp = number("systolic_bp", "Upper Blood Pressure")
    diastolic_bp = number("diastolic_bp", "Lower Blood Pressure")
    heart_rate = number("heart_rate", "Heart Rate")
    daily_steps = number("daily_steps", "Daily Steps")

    if age < 1 or age > 120 or not age.is_integer():
        raise ValueError("Age must be a whole number between 1 and 120.")
    if not 0 < sleep_duration <= 24:
        raise ValueError(
            "Hours of Sleep per Day must be greater than 0 and at most 24 hours."
        )
    if physical_activity < 0:
        raise ValueError("Physical Activity Level cannot be negative.")
    if not 1 <= stress_level <= 10 or not stress_level.is_integer():
        raise ValueError(
            "Perceived Stress Level must be a whole number from 1 to 10."
        )
    if not 50 <= systolic_bp <= 300 or not 30 <= diastolic_bp <= 200:
        raise ValueError(
            "Blood pressure values are outside valid clinical ranges (Systolic: 50-300, Diastolic: 30-200)."
        )
    if not 0 < heart_rate <= 250:
        raise ValueError("Heart Rate must be between 1 and 250 beats/min.")
    if daily_steps < 0 or not daily_steps.is_integer():
        raise ValueError("Daily Steps must be a non-negative whole number.")

    for field, label, choices in [
        ("gender", "Gender", OPTIONS["genders"]),
        ("occupation", "Occupation", OPTIONS["occupations"]),
        ("bmi_category", "BMI category", OPTIONS["bmi_categories"]),
        ("sleep_disorder", "Sleep disorder", OPTIONS["sleep_disorders"]),
    ]:
        if values.get(field) not in choices:
            raise ValueError(f"Please select a valid {label.lower()}.")

    input_data = {
        "Age": int(age),
        "Sleep Duration": sleep_duration,
        "Physical Activity Level": physical_activity,
        "Stress Level": int(stress_level),
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        "Heart Rate": heart_rate,
        "Daily Steps": int(daily_steps),
        "Gender": values["gender"],
        "Occupation": values["occupation"],
        "BMI Category": values["bmi_category"],
        "Sleep Disorder": values["sleep_disorder"],
    }
    return input_data, model_name


@app.context_processor
def inject_navigation():
    return {"available_models": AVAILABLE_MODELS}


@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html", active_page="dashboard", **dashboard_context()
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error = None
    selected_model = request.form.get("model", "random_forest")
    if request.method == "POST":
        try:
            input_data, selected_model = validate_form()
            result = predict_sleep_quality(input_data, selected_model)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            error = str(exc)
    return render_template(
        "prediction.html",
        active_page="prediction",
        result=result,
        error=error,
        selected_model=selected_model,
        options=form_options(),
        presets=PRESETS,
    )


@app.route("/dataset")
def dataset_analysis():
    context = dashboard_context()
    feature_summary = [
        {
            "feature": feature,
            "mean": context["numeric_summary"][feature]["mean"],
            "median": context["numeric_summary"][feature]["50%"],
            "minimum": context["numeric_summary"][feature]["min"],
            "maximum": context["numeric_summary"][feature]["max"],
            "std_dev": context["numeric_summary"][feature]["std"],
        }
        for feature in NUMERIC_FEATURES
    ]
    selected_feature = request.args.get("feature", "Sleep Duration")
    if selected_feature not in NUMERIC_FEATURES:
        selected_feature = "Sleep Duration"
    selected_model = request.args.get("model", "random_forest")
    if selected_model not in AVAILABLE_MODELS:
        selected_model = "random_forest"

    feature_values = pd.to_numeric(
        context["df"][selected_feature], errors="coerce"
    ).dropna()
    if selected_feature == "Sleep Duration":
        bins = [0, 6, 7, 8, 24]
        distribution_labels = ["< 6 hrs", "6-7 hrs", "7-8 hrs", "> 8 hrs"]
        binned = pd.cut(
            feature_values,
            bins=bins,
            labels=distribution_labels,
            include_lowest=True,
        )
    else:
        bin_count = min(8, max(1, feature_values.nunique()))
        binned = pd.cut(feature_values, bins=bin_count, include_lowest=True)
        distribution_labels = [str(interval) for interval in binned.cat.categories]

    distribution_values = (
        binned.value_counts()
        .reindex(binned.cat.categories, fill_value=0)
        .astype(int)
        .tolist()
    )

    importance, importance_method = get_feature_importance(selected_model)

    preview = (
        context["df"]
        .head(10)
        .drop(columns=["Sleep_Category"])
        .fillna("None")
        .to_dict(orient="records")
    )
    return render_template(
        "dataset.html",
        active_page="dataset",
        preview=preview,
        feature_summary=feature_summary,
        feature_options=NUMERIC_FEATURES,
        selected_feature=selected_feature,
        selected_model=selected_model,
        distribution_labels=distribution_labels,
        distribution_values=distribution_values,
        importance=importance,
        importance_method=importance_method,
        **context,
    )


@app.route("/performance")
def performance():
    context = dashboard_context()
    selected_model = request.args.get("model", "Random Forest")
    model_names = {row["Model"] for row in context["cv_results"]}
    if selected_model not in model_names:
        selected_model = "Random Forest"
    selected_result = next(
        (row for row in context["cv_results"] if row["Model"] == selected_model),
        None,
    )
    return render_template(
        "performance.html",
        active_page="performance",
        selected_model=selected_model,
        selected_result=selected_result,
        model_descriptions=MODEL_DESCRIPTIONS,
        **context,
    )


@app.route("/about")
def about():
    return render_template("about.html", active_page="about")


# Compatibility redirects for former endpoints
@app.route("/patterns")
def patterns():
    return redirect(url_for("dataset_analysis"))


@app.route("/confusion-matrix")
def confusion_matrix_page():
    return redirect(url_for("performance"))


if __name__ == "__main__":
    app.run(debug=True)
