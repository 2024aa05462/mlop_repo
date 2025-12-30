import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import logging
import os

logger = logging.getLogger(__name__)

def build_preprocessing_pipeline():
    """
    Build the Scikit-Learn preprocessing pipeline.
    """
    # Define features
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    # Note: 'sex', 'fbs', 'exang' are binary but often treated as categorical for OHE or passed through if 0/1.
    # The user plan says:
    # Binary (keep as 0/1): sex, fbs, exang
    # Ordinal (LabelEncoder): slope, ca (Actually ca is 0-4 num vessels, slope is 1-3)
    # Nominal (OneHot): cp, restecg, thal
    
    # Revised strategy based on plan details:
    # Binary features: 'sex', 'fbs', 'exang' -> Passthrough (or SimpleImputer + Passthrough)
    # Numerical: 'age', 'trestbps', 'chol', 'thalach', 'oldpeak' -> Impute Median + Scaling
    # Nominal: 'cp', 'restecg', 'thal', 'slope' -> Impute Mode + OneHot
    # 'ca': The plan says LabelEncoder, but CA is 0-3. It can be treated as numerical or ordinal. 
    # Let's treat 'ca' and 'slope' as numerical or ordinal.
    
    # Let's strictly follow the plan's suggested "SimpleImputer + Encoder" approach.
    
    # Group 1: Numerical (Standard Scaler)
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Group 2: Nominal (OneHot)
    # Plan says: cp, restecg, thal
    cat_nominal_cols = ['cp', 'restecg', 'thal']
    cat_nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Group 3: Binary/Ordinal that we want to keep as is or simple impute
    # Plan says: sex, fbs, exang (binary), slope, ca (ordinal)
    # We can just impute them.
    # Actually, for 'slope' and 'ca', if we want to treat them as Ordinal, we might just pass them through if they are already numeric encoding 
    # but we should ensure no missing values.
    
    passthrough_cols = ['sex', 'fbs', 'exang', 'slope', 'ca']
    passthrough_pipeline = Pipeline([
         ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat_nominal', cat_nominal_pipeline, cat_nominal_cols),
        ('passthrough', passthrough_pipeline, passthrough_cols)
    ])
    
    return preprocessor

def engineer_features(df):
    """
    Create new features.
    """
    df = df.copy()
    # Heart rate reserve
    if 'thalach' in df.columns and 'age' in df.columns:
        df['heart_rate_reserve'] = 220 - df['age'] - df['thalach']
    
    # Example interaction
    if 'chol' in df.columns and 'trestbps' in df.columns:
        df['chol_bp_interaction'] = df['chol'] * df['trestbps']
        
    return df

def preprocess_data(df, train=True, preprocessor=None):
    """
    Apply preprocessing to dataframe.
    """
    X = df.drop(columns=['target'], errors='ignore')
    y = df['target'] if 'target' in df.columns else None
    
    # Feature Engineering
    X = engineer_features(X)
    
    if train:
        preprocessor = build_preprocessing_pipeline()
        X_processed = preprocessor.fit_transform(X)
        return X_processed, y, preprocessor
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor must be provided for testing/inference.")
        X_processed = preprocessor.transform(X)
        return X_processed, y, None

if __name__ == "__main__":
    # Test run
    # Mock data or load real
    pass
