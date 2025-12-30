#!/usr/bin/env python3
"""
Data Download Script for Heart Disease UCI Dataset

This script downloads the Heart Disease dataset from UCI Machine Learning Repository
and saves it in the data/raw directory.

Usage:
    python scripts/download_data.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset URL
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names for the dataset
COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Output directories
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    logger.info(f"Created directories: {RAW_DATA_DIR}, {PROCESSED_DATA_DIR}")


def download_dataset():
    """Download the Heart Disease dataset from UCI."""
    logger.info(f"Downloading dataset from {UCI_URL}")
    
    try:
        df = pd.read_csv(UCI_URL, names=COLUMN_NAMES, na_values='?')
        logger.info(f"Successfully downloaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        
        # Try alternative method using ucimlrepo if available
        try:
            from ucimlrepo import fetch_ucirepo
            logger.info("Attempting download via ucimlrepo...")
            heart_disease = fetch_ucirepo(id=45)
            X = heart_disease.data.features
            y = heart_disease.data.targets
            df = pd.concat([X, y], axis=1)
            
            # Rename target column if needed
            if 'num' in df.columns:
                df = df.rename(columns={'num': 'target'})
            
            logger.info(f"Successfully downloaded via ucimlrepo with shape: {df.shape}")
            return df
        except ImportError:
            logger.error("ucimlrepo not installed. Run: pip install ucimlrepo")
            raise


def clean_dataset(df):
    """Clean the dataset by handling missing values and converting target."""
    logger.info("Cleaning dataset...")
    
    # Show initial missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.info(f"Missing values found:\n{missing[missing > 0]}")
    
    # Drop rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    logger.info(f"Dropped {dropped_rows} rows with missing values")
    
    # Convert all columns to appropriate types
    df = df.astype(float)
    
    # Convert target to binary (0: no disease, 1: disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    logger.info(f"Converted target to binary. Distribution:\n{df['target'].value_counts()}")
    
    return df


def save_datasets(df):
    """Save raw and cleaned datasets."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw data with timestamp
    raw_path = os.path.join(RAW_DATA_DIR, f"heart_disease_raw_{timestamp}.csv")
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw data to: {raw_path}")
    
    # Save as latest raw data (for easy access)
    latest_raw_path = os.path.join(RAW_DATA_DIR, "heart_disease_raw.csv")
    df.to_csv(latest_raw_path, index=False)
    logger.info(f"Saved latest raw data to: {latest_raw_path}")
    
    # Save cleaned data
    cleaned_path = os.path.join(PROCESSED_DATA_DIR, "heart_disease_clean.csv")
    df.to_csv(cleaned_path, index=False)
    logger.info(f"Saved cleaned data to: {cleaned_path}")
    
    return cleaned_path


def generate_metadata(df, output_path):
    """Generate metadata about the dataset."""
    metadata = {
        "dataset_name": "Heart Disease UCI",
        "source": "UCI Machine Learning Repository",
        "url": UCI_URL,
        "download_date": datetime.now().isoformat(),
        "n_samples": len(df),
        "n_features": len(df.columns) - 1,
        "target_distribution": df['target'].value_counts().to_dict(),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "statistics": {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
            for col in df.columns if df[col].dtype in [np.float64, np.int64]
        }
    }
    
    import json
    metadata_path = os.path.join(PROCESSED_DATA_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_path}")
    
    return metadata


def main():
    """Main function to orchestrate data download and processing."""
    logger.info("=" * 50)
    logger.info("Heart Disease Dataset Download Script")
    logger.info("=" * 50)
    
    try:
        # Create directories
        create_directories()
        
        # Download dataset
        df = download_dataset()
        
        # Clean dataset
        df = clean_dataset(df)
        
        # Save datasets
        save_datasets(df)
        
        # Generate metadata
        generate_metadata(df, PROCESSED_DATA_DIR)
        
        logger.info("=" * 50)
        logger.info("Dataset download and processing complete!")
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

