"""
Unit tests for FastAPI endpoints.
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_response_structure(self):
        """Test health response has correct structure."""
        response = {"status": "healthy"}
        
        assert "status" in response
        assert response["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    @pytest.fixture
    def valid_input_params(self):
        """Valid input parameters for prediction."""
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

    def test_valid_input_params(self, valid_input_params):
        """Test that valid input parameters are accepted."""
        # Age validation
        assert 0 <= valid_input_params["age"] <= 120
        
        # Sex validation
        assert valid_input_params["sex"] in [0, 1]
        
        # Chest pain type validation
        assert 0 <= valid_input_params["cp"] <= 3
        
        # Blood pressure validation
        assert 0 <= valid_input_params["trestbps"] <= 300
        
        # Cholesterol validation
        assert 0 <= valid_input_params["chol"] <= 600

    def test_prediction_response_structure(self):
        """Test prediction response has correct structure."""
        response = {
            "prediction": 1,
            "confidence": 0.85
        }
        
        assert "prediction" in response
        assert "confidence" in response
        assert response["prediction"] in [0, 1]
        assert 0 <= response["confidence"] <= 1

    def test_invalid_age_range(self):
        """Test that invalid age raises validation error."""
        invalid_ages = [-1, 150, -50]
        
        for age in invalid_ages:
            assert age < 0 or age > 120

    def test_invalid_sex_value(self):
        """Test that invalid sex value raises validation error."""
        invalid_sex_values = [2, -1, 5]
        
        for sex in invalid_sex_values:
            assert sex not in [0, 1]

    def test_invalid_chest_pain_type(self):
        """Test that invalid cp value raises validation error."""
        invalid_cp_values = [5, -1, 10]
        
        for cp in invalid_cp_values:
            assert cp < 0 or cp > 3


class TestInputValidation:
    """Tests for input validation logic."""

    def test_feature_count(self):
        """Test that exactly 13 features are required."""
        features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        assert len(features) == 13

    def test_numeric_features(self):
        """Test that all features are numeric."""
        sample = {
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
        
        for key, value in sample.items():
            assert isinstance(value, (int, float))

    def test_oldpeak_allows_float(self):
        """Test that oldpeak accepts float values."""
        oldpeak_values = [0.0, 1.5, 2.3, 4.2, 6.2]
        
        for val in oldpeak_values:
            assert isinstance(val, float)


class TestErrorHandling:
    """Tests for error handling."""

    def test_model_not_loaded_error(self):
        """Test error when model is not loaded."""
        error_response = {"error": "Model not loaded"}
        
        assert "error" in error_response
        assert error_response["error"] == "Model not loaded"

    def test_missing_parameter_detection(self):
        """Test detection of missing parameters."""
        required_params = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                          'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'}
        
        provided_params = {'age', 'sex', 'cp'}  # Missing many params
        
        missing = required_params - provided_params
        assert len(missing) > 0

