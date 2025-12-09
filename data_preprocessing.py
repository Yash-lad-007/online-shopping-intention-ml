# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

RANDOM_STATE = 42

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def get_features_target(df: pd.DataFrame):
    """
    Assumes target column is 'Revenue' (True/False).
    Adjust if your dataset uses something else.
    """
    y = df['Revenue'].astype(int)  # True/False -> 1/0
    X = df.drop(columns=['Revenue'])
    return X, y

def build_preprocessor(X: pd.DataFrame):
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    numeric_transformer = 'passthrough'
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features

def train_test_split_data(X, y, test_size=0.2):
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception as e:
        print("Could not compute ROC-AUC:", e)
