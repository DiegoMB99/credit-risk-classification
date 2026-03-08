from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List
import os

app = FastAPI(
    title="Credit Risk Classification API",
    description="API para predecir riesgo crediticio de clientes bancarios",
    version="1.0.0"
)

# Cargar modelo, scaler y feature names
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print(f"✅ Modelo cargado: {len(feature_names)} features")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None
    scaler = None
    feature_names = []

# Modelo de entrada
class CustomerFeatures(BaseModel):
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [1.0] * 72  # 72 features
            }
        }

# Modelo de respuesta
class PredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    probability_high_risk: float
    probability_low_risk: float

@app.get("/")
def root():
    return {
        "message": "Credit Risk Classification API",
        "status": "active",
        "model": "Logistic Regression + SMOTE",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "num_features": len(feature_names)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validar número de features
        if len(data.features) != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(feature_names)} features, got {len(data.features)}"
            )
        
        # Convertir a array
        features_array = np.array(data.features).reshape(1, -1)
        
        # Escalar
        features_scaled = scaler.transform(features_array)
        
        # Predecir
        prediction = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]
        
        risk_label = "Alto Riesgo" if prediction == 1 else "Bajo Riesgo"
        
        return PredictionResponse(
            prediction=prediction,
            risk_label=risk_label,
            probability_low_risk=round(float(probabilities[0]), 4),
            probability_high_risk=round(float(probabilities[1]), 4)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
def model_info():
    return {
        "model_type": "Logistic Regression",
        "preprocessing": "StandardScaler",
        "balancing_technique": "SMOTE",
        "performance_metrics": {
            "recall": 0.644,
            "f1_score": 0.42,
            "accuracy": 0.644,
            "auc_roc": 0.648
        },
        "num_features": len(feature_names),
        "classes": {
            "0": "Bajo Riesgo",
            "1": "Alto Riesgo"
        }
    }