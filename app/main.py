import pandas as pd
import mlflow.xgboost
from fastapi import FastAPI, HTTPException
from app.schema import HeartDiseaseInput, HeartDiseaseOutput
import yaml
import os

app = FastAPI(title="Heart Disease Prediction API")

config_path = "config.yaml"
model = None

def load_model():
    global model
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Load specific model or latest from the experiment
        experiment_name = config["model"]["experiment_name"]
        
        # This part assumes you have run the training pipeline at least once
        # and that the mlruns directory is accessible.
        # In a real production setup, you might fetch from a remote Model Registry.
        
        # We will search for the latest run in the experiment
        current_experiment = mlflow.get_experiment_by_name(experiment_name)
        if current_experiment is None:
             raise Exception(f"Experiment {experiment_name} not found.")

        runs = mlflow.search_runs(experiment_ids=[current_experiment.experiment_id], 
                                  order_by=["start_time DESC"], 
                                  max_results=1)
        
        if runs.empty:
            raise Exception("No runs found for the experiment.")
        
        run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{run_id}/model"
        
        print(f"Loading model from {model_uri}...")
        model = mlflow.xgboost.load_model(model_uri)
        print("Model loaded successfully.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # For development purposes, we might start without a model if training hasn't run yet
        model = None

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/predict", response_model=HeartDiseaseOutput)
def predict(input_data: HeartDiseaseInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # specific column order for matching the training format
        columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", 
            "restecg", "thalach", "exang", "oldpeak", "slope", 
            "ca", "thal"
        ]
        
        data_df = pd.DataFrame([input_data.dict()], columns=columns)
        
        prediction = model.predict(data_df)[0]
        # XGBoost predict_proba might return (n_samples, n_classes) or just (n_samples,) depending on version/objective
        # For binary, it often just gives Class 1 prob if using predict_proba, but standard predict gives class.
        # Let's see if we can get probability. The generic predict just returns class for Classifier.
        
        # Attempt to get probability
        probability = 0.0
        if hasattr(model, "predict_proba"):
             # returns [[prob_0, prob_1]]
             probs = model.predict_proba(data_df)
             probability = float(probs[0][1]) # probability of class 1
        
        message = "High risk of heart disease" if prediction == 1 else "Low risk of heart disease"
        
        return HeartDiseaseOutput(
            prediction=int(prediction),
            probability=probability,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
