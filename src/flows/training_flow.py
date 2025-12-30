from prefect import flow
from src.tasks.model_tasks import (
    load_data_task,
    feature_engineering_task,
    split_data_task,
    train_xgboost_task,
    register_model_task
)
import mlflow
import os

@flow(name="Model Training Pipeline")
def training_flow(data_path: str = "data/raw/heart_disease_raw.csv"):
    """
    Flow to train XGBoost model.
    """
    # Setup MLflow
    mlflow.set_tracking_uri("file:///" + os.path.abspath("./mlruns").replace("\\", "/"))
    mlflow.set_experiment("heart_disease_prefect")
    
    # 1. Load Data
    df = load_data_task(data_path)
    
    # 2. Feature Engineering
    df = feature_engineering_task(df)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = split_data_task(df)
    
    # 4. Train
    model, metrics = train_xgboost_task(X_train, y_train, X_test, y_test)
    
    # 5. Register
    register_model_task(model, "heart-disease-xgboost", metrics)

if __name__ == "__main__":
    # Ensure data exists or run ingestion first
    if not os.path.exists("data/raw/heart_disease_raw.csv"):
        print("Data not found. Please run data_ingestion_flow first.")
    else:
        training_flow()
