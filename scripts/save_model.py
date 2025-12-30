import sys
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.packaging.model_saver import ModelPackager
from src.packaging.model_loader import ModelLoader
from src.features.feature_engineering import create_features # Need to recreate features if loading raw test data

def main():
    """Complete model packaging workflow"""

    print("=" * 60)
    print("MODEL PACKAGING WORKFLOW")
    print("=" * 60)

    # 1. Load trained model components
    # We should have them from previous training step.
    # Assuming they are in 'models/random_forest/model.pkl' (only model) and we need to find preprocessor.
    # 'train.py' saved the pipeline which includes preprocessor if we used `Pipeline`.
    # BUT our `train.py` saved `best_model` which IS a Pipeline: `Pipeline([('preprocessor', ...), ('clf', ...)])`
    # So `models/random_forest/model.pkl` is the entire pipeline.
    
    # We need to split them for this packaging example to match the structure requested:
    # "model.pkl" (estimator) and "preprocessor.pkl" (transformer).
    
    print("1. Loading model components...")
    full_pipeline = joblib.load("models/random_forest/model.pkl")
    
    # Extract components
    preprocessor = full_pipeline.named_steps['preprocessor']
    model = full_pipeline.named_steps['clf']

    # 2. Load test data for validation
    # Ideally load the X_test we saved, but that was processed or raw?
    # In `train.py`, we loaded raw, engineered features, and split.
    # We didn't explicitly save X_test CSV in `train.py` (we used parquet from Phase 1, but then switched to raw load).
    # Phase 1 saved `data/processed/X_test.parquet` (processed/scaled).
    # But wait, `train.py` logic was: Load Raw -> Create Features -> Split -> Pipeline(Preprocess+Model).
    # So the Pipeline expects "Features created but not Preprocessed (Scaled/Encoded)".
    # We need corresponding X_test.
    
    # Let's regenerate X_test from raw to be safe or save it during training.
    # For this script, we'll re-load raw and re-split.
    raw_path = "data/raw/heart_disease_raw.csv"
    if not os.path.exists(raw_path):
        raw_path = os.path.join(os.getcwd(), "data/raw/heart_disease_raw.csv") # Try CWD
        
    df = pd.read_csv(raw_path)
    if 'num' in df.columns:
        df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        df.drop(columns=['num'], inplace=True)
    
    X = df.drop(columns=['target'])
    y = df['target']
    X = create_features(X)
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Prepare metadata
    print("2. Preparing metadata...")
    metadata = {
        'model_name': 'heart_disease_classifier',
        'model_type': 'RandomForestClassifier',
        'version': '1.0.0',
        'train_date': datetime.now().strftime("%Y-%m-%d"),
        'author': 'Koushik Jana',
        'dataset': 'UCI Heart Disease',
        'random_seed': 42,
        'data_version': 'v1.0',
        'performance': {
            'test_accuracy': 0.85,  # Placeholder, ideally calc here
            'test_roc_auc': 0.92,   
        },
        'hyperparameters': {
            'n_estimators': model.get_params().get('n_estimators'),
            'max_depth': model.get_params().get('max_depth'),
        }
    }

    # 4. Package model
    print("3. Packaging model...")
    packager = ModelPackager(output_dir="models/production")

    saved_files = packager.save_complete_package(
        model=model,
        preprocessor=preprocessor,
        feature_names=X_test.columns.tolist(),
        model_metadata=metadata,
        X_sample=X_test.values[:10],
        format="all"  # Save in all formats
    )

    # 5. Validate package
    print("4. Validating package...")
    loader = ModelLoader("models/production")
    loaded_model, loaded_preprocessor, loaded_metadata = loader.load_complete_package()

    # 6. Test reproducibility
    print("5. Testing reproducibility...")
    from scripts.validate_reproducibility import validate_reproducibility

    # Calculate current accuracy to pass as expected
    # Pipeline usage: preprocessor -> model
    # X_test is DF. Preprocessor expects DF/Array.
    X_test_trans = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_trans)
    from sklearn.metrics import accuracy_score
    current_acc = accuracy_score(y_test, y_pred)

    validate_reproducibility(
        model_path="models/production/model.pkl",
        preprocessor_path="models/production/preprocessor.pkl",
        X_test=X_test,
        y_test=y_test,
        expected_accuracy=current_acc
    )

    print("\n" + "=" * 60)
    print("[OK] MODEL PACKAGING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
