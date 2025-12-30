from prefect import task
import pandas as pd
import logging
from src.data.load_data import download_dataset, validate_data, convert_target_to_binary, save_raw_data

# Get logger
logger = logging.getLogger(__name__)

@task(name="Download Data", retries=3, retry_delay_seconds=10)
def download_data_task():
    """
    Task to download data from UCI repository.
    """
    logger.info("Starting data download task...")
    try:
        df = download_dataset()
        return df
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

@task(name="Validate Data")
def validate_data_task(df: pd.DataFrame):
    """
    Task to validate the downloaded dataframe.
    """
    logger.info("Validating data...")
    if validate_data(df):
        logger.info("Validation successful")
    return df

@task(name="Process Target")
def process_target_task(df: pd.DataFrame):
    """
    Task to convert target column to binary.
    """
    logger.info("Processing target column...")
    df = convert_target_to_binary(df)
    return df

@task(name="Save Raw Data")
def save_data_task(df: pd.DataFrame, output_dir: str = "data/raw"):
    """
    Task to save raw data to disk.
    """
    logger.info(f"Saving data to {output_dir}...")
    filepath = save_raw_data(df, output_dir)
    return filepath
