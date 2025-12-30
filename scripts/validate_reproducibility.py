import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from pathlib import Path

def validate_reproducibility(
    model_path: str,
    preprocessor_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    expected_accuracy: float,
    tolerance: float = 0.001
):
    """
    Validate that loaded model produces expected results
    """
    print("[INFO] Validating model reproducibility...")

    # Load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Transform and predict
    # NOTE: Assuming X_test passed is RAW data that needs preprocessing
    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)

    # Calculate accuracy
    actual_accuracy = accuracy_score(y_test, y_pred)

    # Check if within tolerance
    diff = abs(actual_accuracy - expected_accuracy)

    print(f"   Expected accuracy: {expected_accuracy:.4f}")
    print(f"   Actual accuracy:   {actual_accuracy:.4f}")
    print(f"   Difference:        {diff:.4f}")

    if diff <= tolerance:
        print("[OK] Reproducibility validated!")
        return True
    else:
        print(f"[FAIL] Reproducibility check failed! Difference {diff} exceeds tolerance {tolerance}")
        return False
