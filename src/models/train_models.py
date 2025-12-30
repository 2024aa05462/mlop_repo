"""
Model Training Script with Multiple Models

This script trains both Logistic Regression and Random Forest classifiers
on the Heart Disease dataset and logs experiments to MLflow.

Usage:
    python src/models/train_models.py
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMN_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
OUTPUT_DIR = 'models/production'
EXPERIMENT_NAME = "heart-disease-models"


def load_data():
    """Load and prepare the Heart Disease dataset."""
    logger.info("Loading data...")
    
    # Try to load from local file first
    local_path = 'data/processed/heart_disease_clean.csv'
    if os.path.exists(local_path):
        logger.info(f"Loading from local file: {local_path}")
        df = pd.read_csv(local_path)
    else:
        logger.info(f"Downloading from UCI: {DATA_URL}")
        df = pd.read_csv(DATA_URL, names=COLUMN_NAMES, na_values='?')
        
        # Clean data
        df = df.dropna()
        df = df.astype(float)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv(local_path, index=False)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['target'].value_counts()}")
    
    return df


def build_preprocessor():
    """Build the preprocessing pipeline."""
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, NUMERICAL_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
    ])
    
    return preprocessor


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate and return all evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    return metrics


def train_logistic_regression(X_train, X_test, y_train, y_test, preprocessor):
    """Train Logistic Regression with hyperparameter tuning."""
    logger.info("=" * 50)
    logger.info("Training Logistic Regression...")
    logger.info("=" * 50)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Hyperparameter grid
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'liblinear']
    }
    
    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.set_tag("model_type", "LogisticRegression")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Log to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        
        # Print results
        logger.info(f"Best Parameters: {grid_search.best_params_}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"CV ROC-AUC: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        
        return best_model, metrics


def train_random_forest(X_train, X_test, y_train, y_test, preprocessor):
    """Train Random Forest with hyperparameter tuning."""
    logger.info("=" * 50)
    logger.info("Training Random Forest...")
    logger.info("=" * 50)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.set_tag("model_type", "RandomForest")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Log to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        
        # Print results
        logger.info(f"Best Parameters: {grid_search.best_params_}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"CV ROC-AUC: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        
        return best_model, metrics


def save_production_model(model, metrics, model_name):
    """Save the best model to production directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'model.pkl')
    joblib.dump(model.named_steps['classifier'], model_path)
    logger.info(f"Saved model to: {model_path}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(OUTPUT_DIR, 'preprocessor.pkl')
    joblib.dump(model.named_steps['preprocessor'], preprocessor_path)
    logger.info(f"Saved preprocessor to: {preprocessor_path}")
    
    # Save full pipeline
    pipeline_path = os.path.join(OUTPUT_DIR, 'full_pipeline.pkl')
    joblib.dump(model, pipeline_path)
    logger.info(f"Saved full pipeline to: {pipeline_path}")
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "type": model_name,
            "module": str(type(model.named_steps['classifier']).__module__),
            "sklearn_version": "1.3.0"
        },
        "preprocessor_info": {
            "type": "ColumnTransformer",
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES
        },
        "metrics": metrics,
        "user_metadata": {
            "model_name": "heart_disease_classifier",
            "version": "1.0.0",
            "train_date": datetime.now().strftime("%Y-%m-%d"),
            "dataset": "UCI Heart Disease"
        },
        "reproducibility": {
            "random_seed": 42,
            "train_date": datetime.now().strftime("%Y-%m-%d"),
            "data_version": "v1.0"
        }
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_path}")
    
    # Save feature names
    feature_names = {
        "feature_names": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES
    }
    feature_names_path = os.path.join(OUTPUT_DIR, 'feature_names.json')
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Saved feature names to: {feature_names_path}")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Heart Disease Model Training Pipeline")
    logger.info("=" * 60)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:///" + os.path.abspath("./mlruns").replace("\\", "/"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load data
    df = load_data()
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Build preprocessor
    preprocessor = build_preprocessor()
    
    # Train models
    lr_model, lr_metrics = train_logistic_regression(
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    rf_model, rf_metrics = train_random_forest(
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # Compare and select best model
    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison")
    logger.info("=" * 60)
    logger.info(f"Logistic Regression ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    logger.info(f"Random Forest ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    
    # Select best model based on ROC-AUC
    if rf_metrics['roc_auc'] >= lr_metrics['roc_auc']:
        best_model = rf_model
        best_metrics = rf_metrics
        best_name = "RandomForestClassifier"
        logger.info("\nBest Model: Random Forest")
    else:
        best_model = lr_model
        best_metrics = lr_metrics
        best_name = "LogisticRegression"
        logger.info("\nBest Model: Logistic Regression")
    
    # Save production model
    save_production_model(best_model, best_metrics, best_name)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best Model: {best_name}")
    logger.info(f"Test Accuracy: {best_metrics['accuracy']:.4f}")
    logger.info(f"Test ROC-AUC: {best_metrics['roc_auc']:.4f}")
    logger.info(f"Artifacts saved to: {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

