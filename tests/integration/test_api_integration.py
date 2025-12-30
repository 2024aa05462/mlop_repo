"""
Integration tests for the Heart Disease Prediction API.
"""
import pytest
import requests
import time
import os

# API base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class TestHealthEndpoint:
    """Integration tests for health endpoint."""

    def test_health_check_returns_200(self):
        """Test that health endpoint returns 200."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

    def test_health_check_returns_healthy_status(self):
        """Test that health check returns healthy status."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")


class TestPredictionEndpoint:
    """Integration tests for prediction endpoint."""

    @pytest.fixture
    def valid_params(self):
        """Valid parameters for prediction."""
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

    def test_predict_returns_200(self, valid_params):
        """Test that predict endpoint returns 200 with valid input."""
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                params=valid_params,
                timeout=30
            )
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

    def test_predict_returns_prediction(self, valid_params):
        """Test that predict returns a prediction value."""
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                params=valid_params,
                timeout=30
            )
            data = response.json()
            assert "prediction" in data
            assert data["prediction"] in [0, 1]
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

    def test_predict_returns_confidence(self, valid_params):
        """Test that predict returns confidence score."""
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                params=valid_params,
                timeout=30
            )
            data = response.json()
            assert "confidence" in data
            assert 0 <= data["confidence"] <= 1
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

    def test_predict_high_risk_patient(self):
        """Test prediction for high-risk patient profile."""
        high_risk_params = {
            "age": 70,
            "sex": 1,
            "cp": 3,
            "trestbps": 180,
            "chol": 300,
            "fbs": 1,
            "restecg": 2,
            "thalach": 100,
            "exang": 1,
            "oldpeak": 4.0,
            "slope": 2,
            "ca": 3,
            "thal": 2
        }
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                params=high_risk_params,
                timeout=30
            )
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

    def test_predict_low_risk_patient(self):
        """Test prediction for low-risk patient profile."""
        low_risk_params = {
            "age": 35,
            "sex": 0,
            "cp": 0,
            "trestbps": 110,
            "chol": 180,
            "fbs": 0,
            "restecg": 0,
            "thalach": 180,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 1,
            "ca": 0,
            "thal": 1
        }
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                params=low_risk_params,
                timeout=30
            )
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")


class TestAPIResponseTime:
    """Tests for API response time."""

    def test_health_response_time(self):
        """Test health endpoint responds within 1 second."""
        try:
            start = time.time()
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            duration = time.time() - start
            
            assert response.status_code == 200
            assert duration < 1.0, f"Health check took {duration:.2f}s, expected < 1s"
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

    def test_predict_response_time(self):
        """Test predict endpoint responds within 5 seconds."""
        params = {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
            "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
            "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        }
        try:
            start = time.time()
            response = requests.post(
                f"{API_BASE_URL}/predict",
                params=params,
                timeout=30
            )
            duration = time.time() - start
            
            assert response.status_code == 200
            assert duration < 5.0, f"Prediction took {duration:.2f}s, expected < 5s"
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")


class TestErrorHandling:
    """Tests for API error handling."""

    def test_missing_parameters_handled(self):
        """Test that missing parameters are handled gracefully."""
        try:
            # Only provide some parameters
            partial_params = {"age": 63, "sex": 1}
            response = requests.post(
                f"{API_BASE_URL}/predict",
                params=partial_params,
                timeout=30
            )
            # Should return error (422 or similar)
            assert response.status_code in [400, 422]
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

    def test_invalid_endpoint_returns_404(self):
        """Test that invalid endpoint returns 404."""
        try:
            response = requests.get(f"{API_BASE_URL}/invalid_endpoint", timeout=10)
            assert response.status_code == 404
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
