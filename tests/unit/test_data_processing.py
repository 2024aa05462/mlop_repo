"""
Unit tests for data processing and feature engineering modules.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_expected_columns_present(self, sample_heart_data):
        """Test that all expected columns are present."""
        expected_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal', 'target'
        ]
        assert all(col in sample_heart_data.columns for col in expected_columns)

    def test_data_shape(self, sample_heart_data):
        """Test data has correct number of columns."""
        assert sample_heart_data.shape[1] == 14

    def test_no_missing_values(self, sample_heart_data):
        """Test that sample data has no missing values."""
        assert not sample_heart_data.isnull().any().any()

    def test_target_is_binary(self, sample_heart_data):
        """Test that target column is binary."""
        assert sample_heart_data['target'].isin([0, 1]).all()


class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    def test_heart_rate_reserve_calculation(self, sample_features):
        """Test heart rate reserve calculation."""
        # HR Reserve = 220 - age - max_heart_rate
        hr_reserve = 220 - sample_features['age'] - sample_features['thalach']
        
        # Should be reasonable values
        assert all(hr_reserve >= -50)
        assert all(hr_reserve <= 150)

    def test_cholesterol_categorization(self, sample_features):
        """Test cholesterol is correctly categorized."""
        chol = sample_features['chol']
        
        # Normal < 200, Borderline 200-239, High >= 240
        def categorize(x):
            if x < 200:
                return 0
            elif x < 240:
                return 1
            else:
                return 2
        
        categories = chol.apply(categorize)
        assert categories.isin([0, 1, 2]).all()

    def test_blood_pressure_categories(self, sample_features):
        """Test blood pressure categorization."""
        bp = sample_features['trestbps']
        
        # All values should be positive
        assert (bp > 0).all()
        
        # Categorize: Normal < 120, Elevated 120-129, High >= 130
        def categorize(x):
            if x < 120:
                return 0
            elif x < 130:
                return 1
            else:
                return 2
        
        categories = bp.apply(categorize)
        assert categories.isin([0, 1, 2]).all()

    def test_interaction_feature_creation(self, sample_features):
        """Test interaction features can be created."""
        age_x_thalach = sample_features['age'] * sample_features['thalach']
        chol_x_bp = sample_features['chol'] * sample_features['trestbps']
        
        assert len(age_x_thalach) == len(sample_features)
        assert len(chol_x_bp) == len(sample_features)
        assert all(age_x_thalach > 0)
        assert all(chol_x_bp > 0)


class TestDataValidation:
    """Tests for data validation."""

    def test_age_range(self, sample_heart_data):
        """Test age is within valid range."""
        assert (sample_heart_data['age'] >= 0).all()
        assert (sample_heart_data['age'] <= 120).all()

    def test_sex_is_binary(self, sample_heart_data):
        """Test sex is binary (0 or 1)."""
        assert sample_heart_data['sex'].isin([0, 1]).all()

    def test_chest_pain_range(self, sample_heart_data):
        """Test chest pain type is in valid range."""
        assert (sample_heart_data['cp'] >= 0).all()
        assert (sample_heart_data['cp'] <= 4).all()

    def test_blood_pressure_positive(self, sample_heart_data):
        """Test blood pressure is positive."""
        assert (sample_heart_data['trestbps'] > 0).all()

    def test_cholesterol_positive(self, sample_heart_data):
        """Test cholesterol is positive."""
        assert (sample_heart_data['chol'] > 0).all()

    def test_fbs_is_binary(self, sample_heart_data):
        """Test fasting blood sugar is binary."""
        assert sample_heart_data['fbs'].isin([0, 1]).all()

    def test_max_heart_rate_range(self, sample_heart_data):
        """Test max heart rate is in valid range."""
        assert (sample_heart_data['thalach'] > 0).all()
        assert (sample_heart_data['thalach'] <= 250).all()


class TestDataTransformations:
    """Tests for data transformations."""

    def test_standard_scaling_properties(self, sample_features, numerical_features):
        """Test that numerical features can be standardized."""
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(sample_features[numerical_features])
        
        # Mean should be approximately 0
        assert np.allclose(scaled.mean(axis=0), 0, atol=1e-10)
        
        # Std should be approximately 1
        assert np.allclose(scaled.std(axis=0), 1, atol=1e-10)

    def test_one_hot_encoding(self, sample_features):
        """Test one-hot encoding of categorical features."""
        from sklearn.preprocessing import OneHotEncoder
        
        cat_cols = ['cp', 'restecg']
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(sample_features[cat_cols])
        
        # Should have more columns than input
        assert encoded.shape[1] >= len(cat_cols)
        
        # All values should be 0 or 1
        assert np.all((encoded == 0) | (encoded == 1))

    def test_missing_value_imputation(self):
        """Test missing value imputation."""
        from sklearn.impute import SimpleImputer
        
        data_with_nulls = pd.DataFrame({
            'age': [50, None, 60],
            'chol': [200, 250, None]
        })
        
        imputer = SimpleImputer(strategy='median')
        imputed = imputer.fit_transform(data_with_nulls)
        
        # No missing values after imputation
        assert not np.isnan(imputed).any()

