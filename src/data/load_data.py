import pandas as pd
import os
from ucimlrepo import fetch_ucirepo
from datetime import datetime
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_dataset():
    """
    Download the Heart Disease dataset from UCI Repository.
    Returns the dataframe.
    """
    logger.info("Downloading Heart Disease dataset from UCI Repository (ID=45)...")
    try:
        heart_disease = fetch_ucirepo(id=45)
        
        # original features and targets
        X = heart_disease.data.features
        y = heart_disease.data.targets
        
        # Combine into one dataframe
        df = pd.concat([X, y], axis=1)
        
        logger.info(f"Dataset downloaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def validate_data(df):
    """
    Validate the dataset structure.
    """
    expected_columns = 14 # 13 features + 1 target
    if df.shape[1] != expected_columns:
        logger.warning(f"Unexpected number of columns: {df.shape[1]}. Expected: {expected_columns}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError("Dataframe is empty.")
    
    logger.info("Data validation passed.")
    return True

def convert_target_to_binary(df):
    """
    Convert the 'num' target column towards a binary 'target' column.
    0 -> 0 (No disease)
    1, 2, 3, 4 -> 1 (Disease)
    """
    if 'num' in df.columns:
        df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        df.drop(columns=['num'], inplace=True)
        logger.info("Converted 'num' target to binary 'target'.")
    else:
        logger.warning("'num' column not found. Skipping target conversion.")
    return df

def save_raw_data(df, output_dir="data/raw"):
    """
    Save the raw dataframe to CSV with timestamp.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"heart_disease_raw_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    logger.info(f"Raw data saved to {filepath}")
    
    # Also save a 'latest.csv' for easy access
    latest_path = os.path.join(output_dir, "heart_disease_raw.csv")
    df.to_csv(latest_path, index=False)
    logger.info(f"Raw data saved to {latest_path}")
    
    return filepath

def load_data_pipeline():
    """
    Orchestrate the data loading process.
    """
    df = download_dataset()
    validate_data(df)
    df = convert_target_to_binary(df)
    save_raw_data(df)
    return df

if __name__ == "__main__":
    load_data_pipeline()
