
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


DATA_PATH = "data/Sleep_health_and_lifestyle_dataset.csv"

TARGET_COLUMN = "Sleep_Category"

NUMERIC_FEATURES = [
    "Age",
    "Sleep Duration",
    "Physical Activity Level",
    "Stress Level",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart Rate",
    "Daily Steps",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Occupation",
    "BMI Category",
    "Sleep Disorder",
]

CATEGORIES = ["Poor", "Moderate", "Good"]


def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    # Clean column names without changing their spelling.
    df.columns = df.columns.str.strip()

    # Treat missing sleep-disorder values as "None".
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

    # Split blood pressure, for example "126/83".
    bp = df["Blood Pressure"].astype(str).str.extract(
        r"^\s*(\d+)\s*/\s*(\d+)\s*$"
    )

    df["Systolic_BP"] = pd.to_numeric(bp[0], errors="coerce")
    df["Diastolic_BP"] = pd.to_numeric(bp[1], errors="coerce")

    # Create project-defined labels from the sleep score.
    df[TARGET_COLUMN] = pd.cut(
        df["Quality of Sleep"],
        bins=[0, 5, 7, 10],
        labels=CATEGORIES,
        include_lowest=True,
    )

    # Remove rows without a valid target.
    df = df.dropna(subset=[TARGET_COLUMN])

    return df


def prepare_data():
    df = load_and_clean_data()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(str)

    # Numeric inputs: fill missing values, then scale.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Text inputs: fill missing values, then convert categories to numbers.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    # Keep all three categories represented in the train/test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, preprocessor
