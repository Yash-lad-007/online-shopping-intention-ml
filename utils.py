# src/utils.py

import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"

def load_trained_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"]

def make_prediction(model, input_dict: dict):
    """
    input_dict: dictionary of feature_name -> value
    """
    X = pd.DataFrame([input_dict])
    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)
    return pred, proba
