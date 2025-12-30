import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.inference.predictor import HeartDiseasePredictor

def main():
    print("=" * 50)
    print("TESTING MODEL INFERENCE with Predictor")
    print("=" * 50)

    try:
        # Initialize predictor
        print("\n1. Initializing Predictor...")
        predictor = HeartDiseasePredictor(model_dir="models/production")
        print("   [OK] Predictor initialized successfully.")

        # Create dummy input based on feature names
        feature_names = predictor.feature_names
        print(f"\n2. Feature names ({len(feature_names)}): {feature_names}")

        # Create a dummy valid input (using meaningful values if possible, else 0s)
        # We'll use 0s for simplicity, ensuring shape matches
        X_dummy = np.zeros((1, len(feature_names)))
        
        # If we have real test data, let's try to load a sample
        # Note: In real scenarios, use data/processed/X_test.parquet transformed back to raw or similar
        # For now, we trust the validation in save_model.py for correctness, this is just interface test.
        
        print(f"\n3. Testing prediction on dummy shape {X_dummy.shape}...")
        result = predictor.predict(X_dummy, return_proba=True)
        
        print("   [OK] Prediction successful!")
        print(f"   Result: {result}")
        
    except Exception as e:
        print(f"\n[FAIL] Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
