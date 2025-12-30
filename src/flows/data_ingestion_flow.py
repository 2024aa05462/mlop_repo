from prefect import flow
from src.tasks.data_tasks import download_data_task, validate_data_task, process_target_task, save_data_task

@flow(name="Data Ingestion Pipeline")
def data_ingestion_flow():
    """
    Flow to download, validate, and save raw data.
    """
    # 1. Download
    df = download_data_task()
    
    # 2. Validate
    df = validate_data_task(df)
    
    # 3. Process Target (Data Cleaning step 1)
    df = process_target_task(df)
    
    # 4. Save
    filepath = save_data_task(df)
    
    return filepath

if __name__ == "__main__":
    data_ingestion_flow()
