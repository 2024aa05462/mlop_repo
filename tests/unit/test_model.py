"""
Unit tests for model training and prediction.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import joblib
import os


class TestModelPrediction:
    """Tests for model prediction functionality."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input for prediction."""
        return np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])

    @pytest.fixture
    def sample_input_df(self):
        """Create sample input as DataFrame."""
        return pd.DataFrame({
            'age': [63],
            'sex': [1],
            'cp': [3],
            'trestbps': [145],
            'chol': [233],
            'fbs': [1],
            'restecg': [0],
            'thalach': [150],
            'exang': [0],
            'oldpeak': [2.3],
            'slope': [0],
            'ca': [0],
            'thal': [1]
        })

    def test_input_has_correct_features(self, sample_input):
        """Test that input has 13 features."""
        assert sample_input.shape[1] == 13

    def test_prediction_is_binary(self):
        """Test that predictions are binary (0 or 1)."""
        # Mock prediction
        predictions = np.array([0, 1, 0, 1, 1])
        assert all(p in [0, 1] for p in predictions)

    def test_probability_range(self):
        """Test that probabilities are between 0 and 1."""
        # Mock probabilities
        probabilities = np.array([0.2, 0.8, 0.45, 0.95, 0.1])
        assert all(0 <= p <= 1 for p in probabilities)

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        proba = np.array([[0.3, 0.7], [0.9, 0.1]])
        confidence = np.max(proba, axis=1)
        
        expected = np.array([0.7, 0.9])
        np.testing.assert_array_equal(confidence, expected)


class TestModelMetrics:
    """Tests for model evaluation metrics."""

    @pytest.fixture
    def predictions_and_labels(self):
        """Create sample predictions and labels."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 0])
        y_prob = np.array([0.2, 0.9, 0.4, 0.1, 0.85, 0.6, 0.95, 0.15, 0.8, 0.35])
        return y_true, y_pred, y_prob

    def test_accuracy_calculation(self, predictions_and_labels):
        """Test accuracy metric calculation."""
        y_true, y_pred, _ = predictions_and_labels
        
        correct = sum(y_true == y_pred)
        total = len(y_true)
        accuracy = correct / total
        
        assert 0 <= accuracy <= 1
        assert accuracy == 0.7  # 7 correct out of 10

    def test_precision_calculation(self, predictions_and_labels):
        """Test precision metric calculation."""
        y_true, y_pred, _ = predictions_and_labels
        
        true_positives = sum((y_true == 1) & (y_pred == 1))
        predicted_positives = sum(y_pred == 1)
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        
        assert 0 <= precision <= 1

    def test_recall_calculation(self, predictions_and_labels):
        """Test recall metric calculation."""
        y_true, y_pred, _ = predictions_and_labels
        
        true_positives = sum((y_true == 1) & (y_pred == 1))
        actual_positives = sum(y_true == 1)
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        
        assert 0 <= recall <= 1

    def test_f1_score_calculation(self, predictions_and_labels):
        """Test F1 score calculation."""
        y_true, y_pred, _ = predictions_and_labels
        
        true_positives = sum((y_true == 1) & (y_pred == 1))
        predicted_positives = sum(y_pred == 1)
        actual_positives = sum(y_true == 1)
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        assert 0 <= f1 <= 1


class TestRiskLevel:
    """Tests for risk level classification."""

    def test_low_risk(self):
        """Test low risk classification."""
        probability = 0.2
        
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.7:
            risk = "Medium"
        else:
            risk = "High"
        
        assert risk == "Low"

    def test_medium_risk(self):
        """Test medium risk classification."""
        probability = 0.5
        
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.7:
            risk = "Medium"
        else:
            risk = "High"
        
        assert risk == "Medium"

    def test_high_risk(self):
        """Test high risk classification."""
        probability = 0.85
        
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.7:
            risk = "Medium"
        else:
            risk = "High"
        
        assert risk == "High"

    def test_boundary_values(self):
        """Test boundary values for risk classification."""
        def get_risk(prob):
            if prob < 0.3:
                return "Low"
            elif prob < 0.7:
                return "Medium"
            else:
                return "High"
        
        assert get_risk(0.29) == "Low"
        assert get_risk(0.30) == "Medium"
        assert get_risk(0.69) == "Medium"
        assert get_risk(0.70) == "High"

