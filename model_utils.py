# model_utils.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

MODEL_PATHS = {
    "random_forest": os.path.join("models", "rf_model.joblib"),
    "logistic_regression": os.path.join("models", "lr_model.joblib"),
    "linear_regression": os.path.join("models", "linr_model.joblib"),
    "neural_network": os.path.join("models", "nn_model.joblib"),
    "multiclass": os.path.join("models", "multi_model.joblib"),
}
MODEL_NAMES = {
    "random_forest": "Random Forest",
    "logistic_regression": "Regresión Logística",
    "linear_regression": "Regresión Lineal",
    "neural_network": "Red Neuronal",
    "multiclass": "Multiclase",
}

FEATURES = [
    "age",
    "fever",
    "headache",
    "myalgias",
    "jaundice",
    "petechiae",
    "rash",
    "hematocrit",
    "hemoglobin",
    "white_blood_cells",
    "platelets",
    "AST (SGOT)",
    "ALT (SGPT)"
]

TARGET = "diagnosis"

FEATURES_METADATA = {
    "age": {"nombre": "Edad", "min": 0, "max": 120},
    "fever": {"nombre": "Fiebre (°C)", "min": 35.0, "max": 42.0},
    "headache": {"nombre": "Cefalea (escala 0-10)", "min": 0, "max": 10},
    "myalgias": {"nombre": "Mialgias (escala 0-10)", "min": 0, "max": 10},
    "jaundice": {"nombre": "Ictericia (0=no, 1=sí)", "min": 0, "max": 1},
    "petechiae": {"nombre": "Petequias (0=no, 1=sí)", "min": 0, "max": 1},
    "rash": {"nombre": "Erupción (0=no, 1=sí)", "min": 0, "max": 1},
    "hematocrit": {"nombre": "Hematocrito (%)", "min": 20, "max": 60},
    "hemoglobin": {"nombre": "Hemoglobina (g/dL)", "min": 5, "max": 20},
    "white_blood_cells": {"nombre": "Glóbulos Blancos (/µL)", "min": 1000, "max": 20000},
    "platelets": {"nombre": "Plaquetas (/µL)", "min": 10000, "max": 500000},
    "AST (SGOT)": {"nombre": "AST (SGOT) (U/L)", "min": 0, "max": 1000},
    "ALT (SGPT)": {"nombre": "ALT (SGPT) (U/L)", "min": 0, "max": 1000},
}

def load_dataset(path="data/DEMALE-HSJM_2025_data.xlsx"):
    df = pd.read_excel(path)
    return df

def prepare_X_y(df, features=FEATURES, target=TARGET):
    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    y = df[target].copy()
    return X, y

def build_pipeline(X_train):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(X_train)
    scaler.fit(imputer.transform(X_train))
    return imputer, scaler

def train_and_save_model(df, test_size=0.2, random_state=42, model_type="random_forest"):
    X, y = prepare_X_y(df)
    if X.shape[1] == 0:
        raise ValueError("Ninguna de las FEATURES configuradas está en el dataset.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None
    )
    imputer, scaler = build_pipeline(X_train)
    X_train_prep = scaler.transform(imputer.transform(X_train))
    X_test_prep = scaler.transform(imputer.transform(X_test))

    smote = SMOTE(random_state=random_state)
    X_train_prep, y_train = smote.fit_resample(X_train_prep, y_train)

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=500, random_state=random_state)
    elif model_type == "linear_regression":
        clf = LinearRegression()
    elif model_type == "neural_network":
        clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=random_state)
    elif model_type == "multiclass":
        clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state))
    else:
        raise ValueError("Modelo no soportado")

    clf.fit(X_train_prep, y_train)

    if hasattr(clf, "predict_proba"):
        y_pred = clf.predict(X_test_prep)
    else:
        y_pred = np.round(clf.predict(X_test_prep))
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    pipeline = {"imputer": imputer, "scaler": scaler, "model": clf, "features": X.columns.tolist()}
    joblib.dump(pipeline, MODEL_PATHS[model_type])

    return {
        "model_pipeline": pipeline,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }

def load_pipeline(model_type="random_forest"):
    path = MODEL_PATHS.get(model_type, MODEL_PATHS["random_forest"])
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def predict_single(input_dict, pipeline=None):
    if pipeline is None:
        pipeline = load_pipeline()
        if pipeline is None:
            raise FileNotFoundError("No se encontró modelo entrenado.")
    features = pipeline["features"]
    X = pd.DataFrame([input_dict])
    X = X.reindex(columns=features)
    X_prep = pipeline["scaler"].transform(pipeline["imputer"].transform(X))
    model = pipeline["model"]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_prep)[0]
        pred = model.predict(X_prep)[0]
        classes = model.classes_
        proba_dict = {int(classes[i]): float(proba[i]) for i in range(len(classes))}
    else:
        pred = np.round(model.predict(X_prep))[0]
        classes = np.unique(pred)
        proba_dict = {int(pred): 1.0}
    return int(pred), proba_dict

def predict_batch(df, pipeline=None):
    if pipeline is None:
        pipeline = load_pipeline()
        if pipeline is None:
            raise FileNotFoundError("No se encontró modelo entrenado.")
    features = pipeline["features"]
    X = df.reindex(columns=features).copy()
    X_prep = pipeline["scaler"].transform(pipeline["imputer"].transform(X))
    model = pipeline["model"]
    if hasattr(model, "predict_proba"):
        preds = model.predict(X_prep)
        probas = model.predict_proba(X_prep)
        classes = model.classes_
    else:
        preds = np.round(model.predict(X_prep))
        probas = None
        classes = np.unique(preds)
    return preds, probas, classes
