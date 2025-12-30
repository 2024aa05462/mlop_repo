import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def stratified_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Perform stratified split into Train, Validation, and Test sets.
    """
    total_test_val_size = test_size + val_size
    
    # First split: Train vs (Test + Val)
    # The size of (Test + Val) relative to total is total_test_val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=total_test_val_size, stratify=y, random_state=random_state
    )
    
    # Second split: Test vs Val
    # Size of Val relative to (Test + Val) is: val_size / total_test_val_size
    relative_val_size = val_size / total_test_val_size
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-relative_val_size, stratify=y_temp, random_state=random_state
    )
    
    logger.info(f"Split completed. Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_splits_to_parquet(X_train, X_val, X_test, y_train, y_val, y_test, output_dir="data/processed"):
    """
    Save splits to Parquet files and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Need to convert numpy array from preprocessing back to DataFrame for Parquet
    # Wait, the output of ColumnTransformer is usually a sparse matrix or numpy array.
    # We should handle that.
    
    def save_df(data, name):
        # If data is numpy array / sparse, convert to DF.
        # Note: We lost column names in preprocessing unless we handle them.
        # For now, we save as generic dataframe or recreate names if possible.
        # The user plan mentions saving feature names separately.
        
        if hasattr(data, "toarray"):
            data = data.toarray()
            
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
            
        filepath = os.path.join(output_dir, f"{name}.parquet")
        df.to_parquet(filepath, compression='snappy')
        logger.info(f"Saved {name} to {filepath}")
        return df.shape

    save_df(X_train, "X_train")
    save_df(X_val, "X_val")
    save_df(X_test, "X_test")
    
    save_df(y_train, "y_train")
    save_df(y_val, "y_val")
    save_df(y_test, "y_test")
    
    # Save Metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "columns": "See feature_names.json" 
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info("Metadata saved.")

if __name__ == "__main__":
    pass
