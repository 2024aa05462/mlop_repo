import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import logging

logger = logging.getLogger(__name__)

def create_features(df):
    """
    Create new features based on domain knowledge.
    This function expects a pandas DataFrame.
    """
    df = df.copy()
    
    # 1. Age groups (decades)
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 70, 80, 100], labels=[0, 1, 2, 3, 4, 5, 6])
    
    # 2. Cholesterol categories
    if 'chol' in df.columns:
        # Normal < 200, Borderline 200-239, High >= 240
        df['chol_cat'] = pd.cut(df['chol'], bins=[0, 200, 239, 1000], labels=[0, 1, 2])

    # 3. Blood pressure categories
    if 'trestbps' in df.columns:
        # Normal < 120, Elevated 120-129, High >= 130 (simplified)
        df['bp_cat'] = pd.cut(df['trestbps'], bins=[0, 120, 129, 300], labels=[0, 1, 2])
        
    # 4. Heart rate reserve
    if 'thalach' in df.columns and 'age' in df.columns:
        # Max heart rate approx 220 - age
        df['hr_reserve'] = (220 - df['age']) - df['thalach']
        
    # 5. Interaction features
    if 'age' in df.columns and 'thalach' in df.columns:
        df['age_x_thalach'] = df['age'] * df['thalach']
        
    if 'chol' in df.columns and 'trestbps' in df.columns:
        df['chol_x_bp'] = df['chol'] * df['trestbps']
        
    logger.info("New features created: age_group, chol_cat, bp_cat, hr_reserve, age_x_thalach, chol_x_bp")
    return df

def build_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Build scikit-learn preprocessing pipeline.
    """
    
    # Separate pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # handle_unknown='ignore' is crucial for production to avoid crashing on new categories
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # Combine
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], verbose_feature_names_out=False)
    
    return preprocessor

def get_feature_names(preprocessor, numerical_features, categorical_features):
    """
    Extract feature names from the column transformer.
    """
    # This is tricky with pipelines, but new sklearn has get_feature_names_out
    try:
        return preprocessor.get_feature_names_out()
    except:
        # Fallback if fit hasn't happened or older sklearn
        return numerical_features + categorical_features # Approximate
