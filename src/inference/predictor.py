import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.packaging.model_loader import ModelLoader
from typing import Dict
import numpy as np

# We need to import ModelLoader from packaging. 
# Relative imports might be tricky without package installation.
# Assuming standard import path from root.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.packaging.model_loader import ModelLoader

class HeartDiseasePredictor:
    """Production-ready predictor with validation"""

    def __init__(self, model_dir: str = "models/production"):
        loader = ModelLoader(model_dir)
        self.model, self.preprocessor, self.metadata = loader.load_complete_package()
        self.feature_names = self.metadata['feature_names']

    def predict(self, X: np.ndarray, return_proba: bool = False) -> Dict:
        """
        Make predictions with validation

        Args:
            X: Input features (raw, not preprocessed)
            return_proba: Whether to return probabilities

        Returns:
            Dictionary with predictions and metadata
        """
        # Validate input
        self._validate_input(X)

        import pandas as pd
        # Preprocess
        # Convert to DataFrame if using named columns in preprocessor
        if isinstance(X, np.ndarray) and hasattr(self.preprocessor, 'feature_names_in_'):
             # If preprocessor expects specific columns (it does, since it failed with string error)
             # We try to use self.feature_names
             X = pd.DataFrame(X, columns=self.feature_names)
        
        X_processed = self.preprocessor.transform(X)

        # Predict
        predictions = self.model.predict(X_processed)

        result = {
            'predictions': predictions.tolist(),
            'model_version': self.metadata['timestamp']
        }

        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_processed)
            result['probabilities'] = probabilities.tolist()
            result['confidence'] = np.max(probabilities, axis=1).tolist()

        return result

    def _validate_input(self, X: np.ndarray):
        """Validate input shape and features"""
        expected_features = len(self.feature_names)
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {X.shape[1]}"
            )
