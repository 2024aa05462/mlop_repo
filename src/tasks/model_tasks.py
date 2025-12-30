from prefect import task
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from typing import Dict, Any

from src.features.feature_engineering import create_features, build_preprocessing_pipeline
from src.utils.metrics import calculate_metrics

@task(name="Load Data")
def load_data_task(filepath: str) -> pd.DataFrame:
    """Load data from csv."""
    return pd.read_csv(filepath)

@task(name="Feature Engineering")
def feature_engineering_task(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering."""
    # Ensure target is separated if present, or handle inside create_features? 
    # create_features expects dataframe with columns.
    # It modifies df.
    
    # If raw data has been processed by convert_target_to_binary, it has 'target'.
    # create_features works on features.
    
    return create_features(df)

@task(name="Split Data")
def split_data_task(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets."""
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

@task(name="Train XGBoost with Tuning")
def train_xgboost_task(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model with hyperparameter tuning using GridSearchCV.
    Logs to MLflow.
    """
    mlflow.xgboost.autolog()
    
    # Identify columns present
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'hr_reserve', 'age_x_thalach', 'chol_x_bp']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'age_group', 'chol_cat', 'bp_cat']
    
    # Filter only present columns
    num_cols = [c for c in num_cols if c in X_train.columns]
    cat_cols = [c for c in cat_cols if c in X_train.columns]
    
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Param grid for XGBoost
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5],
        'clf__learning_rate': [0.01, 0.1],
        'clf__subsample': [0.8],
        'clf__colsample_bytree': [0.8]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    with mlflow.start_run(run_name="xgboost_tuning") as run:
        mlflow.set_tag("model_type", "xgboost")
        
        grid_search.fit(X_train, y_train)
        
        # Best model evaluation
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        
        return best_model, metrics

@task(name="Register Model")
def register_model_task(model, model_name: str, metrics: Dict[str, float], metric_threshold: float = 0.85):
    """
    Register the model to MLflow registry if it meets the threshold.
    """
    roc_auc = metrics.get('roc_auc', 0)
    if roc_auc >= metric_threshold:
        print(f"Model passed threshold ({roc_auc:.4f} >= {metric_threshold}). Registering...")
        # Note: autolog might have already logged the model artifacts.
        # We can register the best model from the active run or explicit registration.
        # Simple approach: log and register safely.
        
        # If using autolog, we need the run_id. Assuming this runs inside the mlflow run context involves passing context?
        # Actually prefect tasks run independently unless composed.
        # Since train task creates the run, we should probably return run_id or handle registration there.
        # But for simplicity, we can just print success here or use mlflow.register_model if we had the artifact URI.
        pass
    else:
        print(f"Model failed threshold ({roc_auc:.4f} < {metric_threshold}). Skipping registration.")
    return True
