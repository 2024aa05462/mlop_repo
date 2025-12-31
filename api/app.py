"""
Heart Disease Prediction API

FastAPI application for serving heart disease predictions.
Includes health checks, prediction endpoint, and monitoring.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging
import os
from datetime import datetime
from typing import Optional
import json


# Pydantic model for JSON input
class PredictionRequest(BaseModel):
    """Request model for heart disease prediction with JSON input."""
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=0, le=300, description="Resting blood pressure")
    chol: int = Field(..., ge=0, le=600, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0=normal, 1=fixed defect, 2=reversible defect)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }


class PredictionResponse(BaseModel):
    """Response model for heart disease prediction."""
    prediction: int = Field(..., description="0 = No disease, 1 = Disease present")
    confidence: float = Field(..., description="Probability score (0-1)")
    risk_level: str = Field(..., description="Low, Medium, or High")
    probability_no_disease: float = Field(..., description="Probability of no heart disease")
    probability_disease: float = Field(..., description="Probability of heart disease")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

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

# Metrics counters
prediction_count = 0
prediction_success = 0
prediction_errors = 0
predictions_by_class = {0: 0, 1: 0}
total_latency_ms = 0.0

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


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Make a heart disease prediction.
    
    Accepts JSON input with patient health data and returns:
    - prediction: 0 (no disease) or 1 (disease present)
    - confidence: Probability score (0-1)
    - risk_level: Low, Medium, or High
    
    Example JSON input:
    ```json
    {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }
    ```
    """
    start_time = datetime.now()
    
    # Check if model is loaded
    if model is None:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create feature array from JSON input
        features = np.array([[
            request.age, request.sex, request.cp, request.trestbps,
            request.chol, request.fbs, request.restecg, request.thalach,
            request.exang, request.oldpeak, request.slope, request.ca, request.thal
        ]])
        
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
        duration_ms = duration * 1000
        
        # Update metrics
        global prediction_count, prediction_success, total_latency_ms
        prediction_count += 1
        prediction_success += 1
        total_latency_ms += duration_ms
        predictions_by_class[prediction] = predictions_by_class.get(prediction, 0) + 1
        
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
            "processing_time_ms": round(duration_ms, 2)
        }
        
    except Exception as e:
        global prediction_errors
        prediction_errors += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
def metrics():
    """
    Prometheus-style metrics endpoint.
    Returns comprehensive application metrics for monitoring.
    """
    from fastapi.responses import PlainTextResponse
    
    uptime = (datetime.now() - startup_time).total_seconds()
    avg_latency = total_latency_ms / prediction_count if prediction_count > 0 else 0
    
    metrics_text = f"""# HELP app_info Application information
# TYPE app_info gauge
app_info{{version="1.0.0",model_loaded="{model is not None}"}} 1

# HELP app_uptime_seconds Application uptime in seconds
# TYPE app_uptime_seconds gauge
app_uptime_seconds {uptime}

# HELP model_loaded Whether the model is loaded
# TYPE model_loaded gauge
model_loaded {1 if model else 0}

# HELP predictions_total Total number of predictions made
# TYPE predictions_total counter
predictions_total {prediction_count}

# HELP predictions_success_total Successful predictions
# TYPE predictions_success_total counter
predictions_success_total {prediction_success}

# HELP predictions_errors_total Failed predictions
# TYPE predictions_errors_total counter
predictions_errors_total {prediction_errors}

# HELP predictions_by_class Predictions by class (0=no disease, 1=disease)
# TYPE predictions_by_class counter
predictions_by_class{{class="0"}} {predictions_by_class.get(0, 0)}
predictions_by_class{{class="1"}} {predictions_by_class.get(1, 0)}

# HELP prediction_latency_avg_ms Average prediction latency in milliseconds
# TYPE prediction_latency_avg_ms gauge
prediction_latency_avg_ms {avg_latency:.2f}

# HELP prediction_latency_total_ms Total prediction latency in milliseconds
# TYPE prediction_latency_total_ms counter
prediction_latency_total_ms {total_latency_ms:.2f}
"""
    
    return PlainTextResponse(content=metrics_text, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
