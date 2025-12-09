# src/train_model.py

import os
import joblib
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import (
    load_data,
    get_features_target,
    build_preprocessor,
    train_test_split_data,
    evaluate_classification_model,
)

DATA_PATH = "data/online_shoppers_intention.csv"
MODEL_PATH = "models/best_model.pkl"

def train():
    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Split into X and y
    X, y = get_features_target(df)

    # 3. Build preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # 4. Define model
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    # 5. Build pipeline
    from sklearn.pipeline import Pipeline
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clf),
    ])

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 7. Fit model
    pipe.fit(X_train, y_train)

    # 8. Evaluate
    print("Evaluating model on test set...")
    evaluate_classification_model(pipe, X_test, y_test)

    # 9. Ensure models dir exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 10. Save model
    joblib.dump(
        {
            "pipeline": pipe,
            "numeric_features": num_cols,
            "categorical_features": cat_cols
        },
        MODEL_PATH
    )
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
