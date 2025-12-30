import pandas as pd
import yaml
from prefect import task

@task
def load_data(config_path: str = "config.yaml"):
    """
    Load data from the raw data path specified in configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    file_path = config["data"]["raw_path"]
    
    # Column names for the UCI Heart Disease dataset
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", 
        "restecg", "thalach", "exang", "oldpeak", "slope", 
        "ca", "thal", "target"
    ]
    
    # Load data, handling '?' as NaN
    df = pd.read_csv(file_path, names=columns, na_values="?")
    
    # Drop rows with missing values (simplest approach for now)
    df = df.dropna()
    
    print(f"Data loaded successfully with shape: {df.shape}")
    return df
