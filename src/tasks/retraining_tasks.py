from prefect import task, get_run_logger
import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Tuple, Dict, Any

@task
def check_drift_and_trigger() -> bool:
    """
    Check if retraining is needed based on drift metrics.
    For simulation, we return True if forced, or check DB.
    Here we mock it to return False by default unless overridden.
    """
    logger = get_run_logger()
    # Logic to query drift_metrics table would go here
    logger.info("Checking for data drift...")
    return False 

@task
def collect_latest_data(days_back: int = 30) -> pd.DataFrame:
    """
    Simulate collecting merged training data.
    """
    logger = get_run_logger()
    logger.info(f"Collecting data from last {days_back} days")
    # In production: Query DB + Labels
    # For now: Load original processed train data
    try:
        data = pd.read_csv("data/processed/train_tfidf.csv") # Placeholder path
        if not data.empty:
            return data
    except Exception:
        pass
    
    # Fallback simulation
    return pd.DataFrame()

@task
def train_candidate_model(data: pd.DataFrame, params: Dict[str, Any]) -> str:
    """
    Train a new candidate model.
    Returns MLflow Run ID.
    """
    if data.empty:
        return None

    logger = get_run_logger()
    logger.info("Training candidate model...")
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    with mlflow.start_run(run_name="retraining_candidate") as run:
        model = xgb.XGBClassifier(**params)
        model.fit(X, y)
        
        # Log metrics
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        mlflow.log_metric("accuracy", acc)
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        return run.info.run_id

@task
def evaluate_candidate(run_id: str, current_model_uri: str) -> bool:
    """
    Compare candidate model with production model.
    """
    if not run_id:
        return False
        
    logger = get_run_logger()
    logger.info(f"Comparing candidate {run_id} with production...")
    
    # Simulation: Always approve if valid run
    return True

@task
def register_and_deploy(run_id: str, model_name: str = "heart-disease-model"):
    """
    Promote model to staging/production.
    """
    if not run_id:
        return
        
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    
    # Register
    mv = client.create_model_version(model_name, model_uri, run_id)
    
    # Transition to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging"
    )
    get_run_logger().info(f"Model version {mv.version} promoted to Staging")
