"""
Heart Disease Prediction API

FastAPI application for serving heart disease predictions.
Includes health checks, prediction endpoint, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import logging
import os
from datetime import datetime
from typing import Optional
import json

# Setup logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML-powered API to predict heart disease risk based on patient health data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
model_metadata = None
startup_time = datetime.now()

# Feature names expected by the model
FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


def load_model():
    """Load model and preprocessor from production directory."""
    global model, preprocessor, model_metadata
    
    model_dir = 'models/production'
    
    try:
        # Load model
        model_path = os.path.join(model_dir, 'model.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.error(f"Model file not found at {model_path}")
            return False
        
        # Load preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_path}")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info("Model metadata loaded")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting Heart Disease Prediction API...")
    success = load_model()
    if success:
        logger.info("API startup complete - model loaded successfully")
    else:
        logger.warning("API started but model not loaded")


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    uptime = (datetime.now() - startup_time).total_seconds()
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info")
def model_info():
    """Get model information and metadata."""
    if model_metadata:
        return {
            "model_type": model_metadata.get("model_info", {}).get("type", "Unknown"),
            "version": model_metadata.get("user_metadata", {}).get("version", "1.0.0"),
            "train_date": model_metadata.get("timestamp", "unknown"),
            "metrics": model_metadata.get("metrics", {}),
            "features": FEATURE_NAMES
        }
    return {"error": "Model metadata not available"}


@app.post("/predict")
def predict(
    age: int = Query(..., ge=0, le=120, description="Age in years"),
    sex: int = Query(..., ge=0, le=1, description="Sex (1=male, 0=female)"),
    cp: int = Query(..., ge=0, le=3, description="Chest pain type (0-3)"),
    trestbps: int = Query(..., ge=0, le=300, description="Resting blood pressure"),
    chol: int = Query(..., ge=0, le=600, description="Serum cholesterol"),
    fbs: int = Query(..., ge=0, le=1, description="Fasting blood sugar > 120"),
    restecg: int = Query(..., ge=0, le=2, description="Resting ECG results"),
    thalach: int = Query(..., ge=0, le=250, description="Max heart rate"),
    exang: int = Query(..., ge=0, le=1, description="Exercise induced angina"),
    oldpeak: float = Query(..., ge=0, le=10, description="ST depression"),
    slope: int = Query(..., ge=0, le=2, description="ST segment slope"),
    ca: int = Query(..., ge=0, le=4, description="Major vessels colored"),
    thal: int = Query(..., ge=0, le=3, description="Thalassemia type")
):
    """
    Make a heart disease prediction.
    
    Returns:
        prediction: 0 (no disease) or 1 (disease present)
        confidence: Probability score (0-1)
        risk_level: Low, Medium, or High
    """
    start_time = datetime.now()
    
    # Check if model is loaded
    if model is None:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create feature array
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]])
        
        # Apply preprocessing if available
        if preprocessor is not None:
            import pandas as pd
            features_df = pd.DataFrame(features, columns=FEATURE_NAMES)
            features_processed = preprocessor.transform(features_df)
        else:
            features_processed = features
        
        # Make prediction
        prediction = int(model.predict(features_processed)[0])
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_processed)[0]
            confidence = float(probabilities[1])  # Probability of disease
        else:
            confidence = float(prediction)
        
        # Determine risk level
        if confidence < 0.3:
            risk_level = "Low"
        elif confidence < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Calculate response time
        duration = (datetime.now() - start_time).total_seconds()
        
        # Log prediction
        logger.info(
            f"Prediction: {prediction}, Confidence: {confidence:.3f}, "
            f"Risk: {risk_level}, Duration: {duration:.3f}s"
        )
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "probability_no_disease": round(1 - confidence, 4),
            "probability_disease": round(confidence, 4),
            "processing_time_ms": round(duration * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
def metrics():
    """
    Prometheus-style metrics endpoint.
    Returns basic application metrics.
    """
    uptime = (datetime.now() - startup_time).total_seconds()
    
    metrics_text = f"""# HELP app_info Application information
# TYPE app_info gauge
app_info{{version="1.0.0",model_loaded="{model is not None}"}} 1

# HELP app_uptime_seconds Application uptime in seconds
# TYPE app_uptime_seconds gauge
app_uptime_seconds {uptime}

# HELP model_loaded Whether the model is loaded
# TYPE model_loaded gauge
model_loaded {1 if model else 0}
"""
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=metrics_text, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
