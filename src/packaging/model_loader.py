import joblib
import json
import mlflow
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np

class ModelLoader:
    """Load and validate packaged models"""

    def __init__(self, model_dir: str = "models/production"):
        self.model_dir = Path(model_dir)

    def load_complete_package(self) -> Tuple[Any, Any, Dict]:
        """
        Load complete model package

        Returns:
            model, preprocessor, metadata
        """
        # Verify package integrity
        self._verify_package()

        # Load components
        model = joblib.load(self.model_dir / "model.pkl")
        preprocessor = joblib.load(self.model_dir / "preprocessor.pkl")

        # Load metadata
        with open(self.model_dir / "model_metadata.json", 'r') as f:
            metadata = json.load(f)

        # Load feature names
        with open(self.model_dir / "feature_names.json", 'r') as f:
            feature_info = json.load(f)

        metadata['feature_names'] = feature_info['feature_names']

        print(f"[OK] Loaded model: {metadata['model_info']['type']}")
        print(f"   Python: {metadata['python_version']}")
        print(f"   Sklearn: {metadata['model_info']['sklearn_version']}")

        return model, preprocessor, metadata

    def _verify_package(self):
        """Verify package integrity"""
        required_files = ['model.pkl', 'preprocessor.pkl', 'model_metadata.json']
        missing = [f for f in required_files if not (self.model_dir / f).exists()]

        if missing:
            raise FileNotFoundError(f"Missing required files: {missing}")

        # Verify checksums if manifest exists
        manifest_path = self.model_dir / "MANIFEST.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            # TODO: Implement checksum verification matching saved checksums
            pass

    def load_mlflow_model(self, model_uri: str = None):
        """Load model from MLflow format"""
        if model_uri is None:
            model_uri = str(self.model_dir.parent / "mlflow_models")

        model = mlflow.sklearn.load_model(model_uri)
        return model
