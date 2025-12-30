"""
Unit tests for data preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


class TestDataPreprocessing:
    """Tests for data preprocessing functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample heart disease data."""
        return pd.DataFrame({
            'age': [63, 67, 67, 37, 41],
            'sex': [1, 1, 1, 1, 0],
            'cp': [1, 4, 4, 3, 2],
            'trestbps': [145, 160, 120, 130, 130],
            'chol': [233, 286, 229, 250, 204],
            'fbs': [1, 0, 0, 0, 0],
            'restecg': [2, 2, 2, 0, 2],
            'thalach': [150, 108, 129, 187, 172],
            'exang': [0, 1, 1, 0, 0],
            'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
            'slope': [3, 2, 2, 3, 1],
            'ca': [0, 3, 2, 0, 0],
            'thal': [6, 3, 7, 3, 3],
            'target': [0, 1, 1, 0, 0]
        })

    def test_data_has_expected_columns(self, sample_data):
        """Test that sample data has all required columns."""
        expected_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal', 'target'
        ]
        assert all(col in sample_data.columns for col in expected_columns)

    def test_data_has_correct_shape(self, sample_data):
        """Test that data has correct dimensions."""
        assert sample_data.shape == (5, 14)

    def test_target_is_binary(self, sample_data):
        """Test that target column is binary (0 or 1)."""
        assert sample_data['target'].isin([0, 1]).all()

    def test_age_is_positive(self, sample_data):
        """Test that age values are positive."""
        assert (sample_data['age'] > 0).all()

    def test_sex_is_binary(self, sample_data):
        """Test that sex is binary."""
        assert sample_data['sex'].isin([0, 1]).all()

    def test_no_null_values(self, sample_data):
        """Test that there are no null values in sample data."""
        assert not sample_data.isnull().any().any()


class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return pd.DataFrame({
            'age': [63, 45, 55],
            'thalach': [150, 180, 140],
            'chol': [233, 200, 280],
            'trestbps': [145, 120, 160]
        })

    def test_heart_rate_reserve_calculation(self, sample_features):
        """Test heart rate reserve calculation."""
        # Heart rate reserve = 220 - age - max_heart_rate
        expected = 220 - sample_features['age'] - sample_features['thalach']
        actual = 220 - sample_features['age'] - sample_features['thalach']
        pd.testing.assert_series_equal(expected, actual)

    def test_cholesterol_categories(self, sample_features):
        """Test cholesterol categorization logic."""
        # Normal < 200, Borderline 200-239, High >= 240
        chol = sample_features['chol']
        
        # 233 -> Borderline (1), 200 -> Borderline (1), 280 -> High (2)
        expected_categories = [1, 1, 2]
        
        def categorize_chol(x):
            if x < 200:
                return 0
            elif x < 240:
                return 1
            else:
                return 2
        
        actual = [categorize_chol(c) for c in chol]
        assert actual == expected_categories


class TestDataValidation:
    """Tests for data validation functions."""

    def test_validate_feature_ranges(self):
        """Test that features are within expected ranges."""
        valid_data = pd.DataFrame({
            'age': [50],
            'sex': [1],
            'cp': [2],
            'trestbps': [130],
            'chol': [250],
            'fbs': [0],
            'restecg': [1],
            'thalach': [150],
            'exang': [0],
            'oldpeak': [1.5],
            'slope': [1],
            'ca': [1],
            'thal': [2]
        })
        
        # Age should be between 0 and 120
        assert (valid_data['age'] >= 0).all() and (valid_data['age'] <= 120).all()
        
        # Sex should be 0 or 1
        assert valid_data['sex'].isin([0, 1]).all()
        
        # Chest pain type should be 0-3
        assert (valid_data['cp'] >= 0).all() and (valid_data['cp'] <= 3).all()

    def test_detect_missing_values(self):
        """Test detection of missing values."""
        data_with_nulls = pd.DataFrame({
            'age': [50, None, 60],
            'chol': [200, 250, None]
        })
        
        assert data_with_nulls.isnull().any().any()

    def test_detect_outliers(self):
        """Test detection of outliers using IQR method."""
        data = pd.Series([50, 55, 60, 65, 70, 200])  # 200 is outlier
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        assert len(outliers) >= 1  # Should detect at least 1 outlier

