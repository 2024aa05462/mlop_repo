import joblib
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class HeartDiseasePredictor:
    """Production-ready heart disease predictor"""

    def __init__(self, model_dir: str = "models/production"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        """Load model, preprocessor, and metadata"""
        try:
            # Load model
            model_path = self.model_dir / "model.pkl"
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")

            # Load preprocessor
            preprocessor_path = self.model_dir / "preprocessor.pkl"
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")

            # Load metadata
            metadata_path = self.model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata loaded: {self.metadata.get('model_info', {}).get('type', 'Unknown')}")
            else:
                 self.metadata = {}

            # Load feature names
            fn_path = self.model_dir / "feature_names.json"
            if fn_path.exists():
                with open(fn_path, 'r') as f:
                    data = json.load(f)
                    self.feature_names = data.get('feature_names')
            
            # If not found, default to standard names 
            if not self.feature_names:
                 self.feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Make prediction

        Args:
            features: Input features as numpy array

        Returns:
            prediction, confidence, probabilities
        """
        # Validate input shape
        if features.shape[1] != 13:
            raise ValueError(f"Expected 13 features, got {features.shape[1]}")
            
        # Convert to DataFrame for preprocessor compatibility (if using names)
        # Assuming current preprocessor pipeline uses ColumnTransformer with named columns
        features_df = pd.DataFrame(features, columns=self.feature_names)

        # Preprocess
        features_processed = self.preprocessor.transform(features_df)

        # Predict
        prediction = self.model.predict(features_processed)[0]
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_processed)[0]
        else:
             # Fallback for models without proba
             probabilities = np.array([1.0-prediction, float(prediction)])
        
        confidence = float(np.max(probabilities))

        return int(prediction), confidence, probabilities

    def get_risk_level(self, probability_disease: float) -> str:
        """Determine risk level based on probability"""
        if probability_disease < 0.3:
            return "Low"
        elif probability_disease < 0.7:
            return "Medium"
        else:
            return "High"

    def get_model_info(self) -> Dict:
        """Get model metadata"""
        info = self.metadata.get('model_info', {})
        return {
            "model_type": info.get('type', 'Unknown'),
            "version": self.metadata.get('version', '1.0.0'),
            "training_date": self.metadata.get('timestamp', 'unknown')
        }
