from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np

class HeartDiseaseFeatures(BaseModel):
    """Input features for heart disease prediction"""
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=0, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

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

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([[
            self.age, self.sex, self.cp, self.trestbps, self.chol,
            self.fbs, self.restecg, self.thalach, self.exang,
            self.oldpeak, self.slope, self.ca, self.thal
        ]])

class PredictionRequest(BaseModel):
    """Request body for batch predictions"""
    instances: List[HeartDiseaseFeatures]

class PredictionResponse(BaseModel):
    """Response with prediction and confidence"""
    prediction: int = Field(..., description="Predicted class (0=no disease, 1=disease)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probability_no_disease: float
    probability_disease: float
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")

    @validator('risk_level')
    def validate_risk_level(cls, v):
        if v not in ['Low', 'Medium', 'High']:
            raise ValueError('risk_level must be Low, Medium, or High')
        return v

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse]
    model_version: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float
