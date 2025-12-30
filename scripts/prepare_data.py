import sys
import os
import joblib
import logging

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import load_data_pipeline
from src.data.preprocess import preprocess_data
from src.data.split_data import stratified_split, save_splits_to_parquet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Data Preparation Phase 1...")
    
    # 1. Load Data
    logger.info("Step 1: Loading Data")
    df = load_data_pipeline()
    
    # 2. EDA (Trigger automated report via code or manual)
    # We will skip running the notebook here but we could invoke the profile report generation
    # if we wanted to verify dependencies.
    # from ydata_profiling import ProfileReport
    # profile = ProfileReport(df, title="Heart Disease EDA Report")
    # profile.to_file("reports/eda_report.html")
    
    # 3. Preprocess
    logger.info("Step 2: Preprocessing")
    X_processed, y, preprocessor = preprocess_data(df, train=True)
    
    # Save preprocessor
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    logger.info("Preprocessor saved to models/preprocessor.pkl")
    
    # 4. Split
    logger.info("Step 3: Splitting Data")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X_processed, y)
    
    # 5. Save
    logger.info("Step 4: Saving Artifacts")
    save_splits_to_parquet(X_train, X_val, X_test, y_train, y_val, y_test)
    
    logger.info("Data Preparation Complete!")

if __name__ == "__main__":
    main()
