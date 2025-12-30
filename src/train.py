import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prefect import task
import yaml

@task
def train_model(X_train, X_test, y_train, y_test, config_path: str = "config.yaml"):
    """
    Train XGBoost model and log to MLflow.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # MLflow setup
    bg_config = config["model"]
    mlflow.set_experiment(bg_config["experiment_name"])
    
    with mlflow.start_run():
        params = {
            "n_estimators": bg_config["n_estimators"],
            "max_depth": bg_config["max_depth"],
            "learning_rate": bg_config["learning_rate"],
            "objective": "binary:logistic",
            "seed": 42
        }
        
        mlflow.log_params(params)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        print("Model logged to MLflow")
        
    return model
