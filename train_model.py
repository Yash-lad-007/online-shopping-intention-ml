# train_model.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

RANDOM_STATE = 42
DATA_PATH = "online_shoppers_intention.csv"
MODEL_PATH = "best_model (2).pkl"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_features_target(df: pd.DataFrame):
    """
    Assumes target column is 'Revenue' (True/False).
    Change if your dataset uses a different name.
    """
    y = df["Revenue"].astype(int)  # True/False -> 1/0
    X = df.drop(columns=["Revenue"])
    return X, y


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("==== Classification Report ====")
    print(classification_report(y_test, y_pred))

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception as e:
        print("Could not compute ROC-AUC:", e)


def main():
    # 1. Load data
    print("Loading data...")
    df = load_data(DATA_PATH)

    # 2. Split features/target
    print("Preparing features and target...")
    X, y = get_features_target(df)

    # 3. Preprocessor
    print("Building preprocessor...")
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # 4. Model
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 5. Pipeline
    pipe = Pipeline(
        steps=[
            ("
