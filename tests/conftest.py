"""
Pytest configuration and fixtures for MLOps Heart Disease project.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_heart_data():
    """Create sample heart disease data for testing."""
    return pd.DataFrame({
        'age': [63, 67, 67, 37, 41, 56, 62, 57, 63, 53],
        'sex': [1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
        'cp': [1, 4, 4, 3, 2, 2, 4, 4, 4, 4],
        'trestbps': [145, 160, 120, 130, 130, 130, 160, 120, 140, 140],
        'chol': [233, 286, 229, 250, 204, 256, 164, 354, 187, 203],
        'fbs': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        'restecg': [2, 2, 2, 0, 2, 2, 2, 0, 2, 2],
        'thalach': [150, 108, 129, 187, 172, 142, 145, 163, 144, 155],
        'exang': [0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
        'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4, 0.6, 6.2, 0.6, 4.0, 3.1],
        'slope': [3, 2, 2, 3, 1, 2, 3, 1, 2, 3],
        'ca': [0, 3, 2, 0, 0, 1, 3, 0, 2, 0],
        'thal': [6, 3, 7, 3, 3, 7, 7, 3, 7, 7],
        'target': [0, 1, 1, 0, 0, 1, 1, 0, 1, 1]
    })


@pytest.fixture
def sample_features():
    """Create sample features (without target) for testing."""
    return pd.DataFrame({
        'age': [63, 67, 55],
        'sex': [1, 1, 0],
        'cp': [1, 4, 2],
        'trestbps': [145, 160, 130],
        'chol': [233, 286, 220],
        'fbs': [1, 0, 0],
        'restecg': [2, 2, 1],
        'thalach': [150, 108, 165],
        'exang': [0, 1, 0],
        'oldpeak': [2.3, 1.5, 1.0],
        'slope': [3, 2, 1],
        'ca': [0, 3, 1],
        'thal': [6, 3, 3]
    })


@pytest.fixture
def sample_input_array():
    """Create sample input as numpy array for prediction."""
    return np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])


@pytest.fixture
def valid_prediction_params():
    """Valid parameters for prediction API."""
    return {
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


@pytest.fixture
def feature_names():
    """List of feature names."""
    return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


@pytest.fixture
def numerical_features():
    """List of numerical feature names."""
    return ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


@pytest.fixture
def categorical_features():
    """List of categorical feature names."""
    return ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']


@pytest.fixture
def model_path():
    """Path to production model."""
    return "models/production/model.pkl"


@pytest.fixture
def preprocessor_path():
    """Path to production preprocessor."""
    return "models/production/preprocessor.pkl"
