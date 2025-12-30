import joblib
import pickle
import json
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import sklearn
import sys

class ModelPackager:
    """
    Comprehensive model packaging utility
    Saves model in multiple formats with full metadata
    """

    def __init__(self, output_dir: str = "models/production"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_complete_package(
        self,
        model,
        preprocessor,
        feature_names: list,
        model_metadata: Dict[str, Any],
        X_sample: np.ndarray = None,
        format: str = "joblib"  # or "pickle", "mlflow", "all"
    ) -> Dict[str, Path]:
        """
        Save complete model package with all artifacts

        Returns:
            Dictionary of saved file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # 1. Save model
        if format in ["joblib", "all"]:
            model_path = self.output_dir / "model.pkl"
            joblib.dump(model, model_path, compress=3)
            saved_files['model'] = model_path

        if format in ["pickle", "all"]:
            model_pickle_path = self.output_dir / "model.pickle"
            with open(model_pickle_path, 'wb') as f:
                pickle.dump(model, f)
            saved_files['model_pickle'] = model_pickle_path

        # 2. Save preprocessor
        preprocessor_path = self.output_dir / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path, compress=3)
        saved_files['preprocessor'] = preprocessor_path

        # 3. Save feature names
        feature_names_path = self.output_dir / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump({
                'feature_names': feature_names,
                'n_features': len(feature_names)
            }, f, indent=2)
        saved_files['feature_names'] = feature_names_path

        # 4. Save comprehensive metadata
        full_metadata = self._create_metadata(model, preprocessor, model_metadata)
        metadata_path = self.output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        saved_files['metadata'] = metadata_path

        # 5. Save model signature (input/output schema)
        if X_sample is not None:
            signature = self._create_signature(model, X_sample, feature_names)
            signature_path = self.output_dir / "model_signature.json"
            with open(signature_path, 'w') as f:
                json.dump(signature, f, indent=2)
            saved_files['signature'] = signature_path

        # 6. Save scaler parameters (if applicable)
        scaler_params = self._extract_scaler_params(preprocessor)
        if scaler_params:
            scaler_path = self.output_dir / "scaler_params.json"
            with open(scaler_path, 'w') as f:
                json.dump(scaler_params, f, indent=2)
            saved_files['scaler_params'] = scaler_path

        # 7. MLflow format (if requested)
        if format in ["mlflow", "all"]:
            mlflow_path = self.save_mlflow_format(
                model, preprocessor, feature_names, X_sample
            )
            saved_files['mlflow'] = mlflow_path

        # 8. Create manifest file
        manifest_path = self._create_manifest(saved_files, timestamp)
        saved_files['manifest'] = manifest_path

        print(f"[OK] Model package saved to: {self.output_dir}")
        return saved_files

    def _create_metadata(self, model, preprocessor, user_metadata: Dict) -> Dict:
        """Create comprehensive metadata"""
        return {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'type': type(model).__name__,
                'module': type(model).__module__,
                'sklearn_version': sklearn.__version__,
            },
            'preprocessor_info': {
                'type': type(preprocessor).__name__,
                'steps': [step[0] for step in preprocessor.steps] if hasattr(preprocessor, 'steps') else []
            },
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'user_metadata': user_metadata,
            'reproducibility': {
                'random_seed': user_metadata.get('random_seed', None),
                'train_date': user_metadata.get('train_date', None),
                'data_version': user_metadata.get('data_version', None)
            }
        }

    def _create_signature(self, model, X_sample: np.ndarray, feature_names: list) -> Dict:
        """Create input/output signature"""
        # Preprocess sample if needed, but model here might be the pipeline or just estimator
        # Assuming 'model' is just the estimator as passed in save_complete_package with separate preprocessor
        # We need to transform X_sample first
        
        # But wait, create_signature usually wants raw input example for serving?
        # The user code snippet uses model.predict(X_sample[:1]). 
        # If 'model' passed to this class (ModelPackager) is just the estimator, it expects processed data.
        # But X_sample passed in is likely raw data (based on calling signatures).
        # We should check.
        
        # Let's assume X_sample is RAW features.
        # We MUST transform it before predicting if 'model' is just the estimator.
        # Ideally, we construct a temp pipeline
        
        try:
             X_processed = preprocessor.transform(X_sample[:1])
             y_pred = model.predict(X_processed)
             y_proba = model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else None
        except:
             # Fallback if model encompasses preprocessing or other issue
             y_pred = model.predict(X_sample[:1])
             y_proba = model.predict_proba(X_sample[:1]) if hasattr(model, 'predict_proba') else None

        return {
            'inputs': {
                'shape': list(X_sample.shape),
                'dtype': str(X_sample.dtype),
                'features': feature_names,
                'sample': X_sample[0].tolist()
            },
            'outputs': {
                'prediction': {
                    'dtype': str(y_pred.dtype),
                    'shape': list(y_pred.shape),
                    'classes': [0, 1]  # Binary classification
                },
                'probabilities': {
                    'dtype': str(y_proba.dtype) if y_proba is not None else None,
                    'shape': list(y_proba.shape) if y_proba is not None else None
                } if y_proba is not None else None
            }
        }

    def _extract_scaler_params(self, preprocessor) -> Optional[Dict]:
        """Extract scaler parameters for reproducibility"""
        try:
            scaler_params = {}
            if hasattr(preprocessor, 'named_transformers_'):
                for name, transformer in preprocessor.named_transformers_.items():
                    if hasattr(transformer, 'named_steps'):
                        for step_name, step in transformer.named_steps.items():
                            if hasattr(step, 'mean_'):
                                scaler_params[f"{name}_{step_name}_mean"] = step.mean_.tolist()
                            if hasattr(step, 'scale_'):
                                scaler_params[f"{name}_{step_name}_scale"] = step.scale_.tolist()
            return scaler_params if scaler_params else None
        except Exception as e:
            print(f"Warning: Could not extract scaler params: {e}")
            return None

    def save_mlflow_format(self, model, preprocessor, feature_names, X_sample):
        """Save in MLflow format"""
        mlflow_path = self.output_dir.parent / "mlflow_models"
        mlflow_path.mkdir(exist_ok=True)

        # Create full pipeline
        from sklearn.pipeline import Pipeline
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Save with MLflow
        from mlflow.models.signature import infer_signature
        
        # Infer signature on raw data -> prediction
        signature = infer_signature(X_sample, full_pipeline.predict(X_sample))

        # We need to delete directory if exists or mlflow will error
        import shutil
        if mlflow_path.exists():
             shutil.rmtree(mlflow_path)
             
        mlflow.sklearn.save_model(
            sk_model=full_pipeline,
            path=str(mlflow_path),
            signature=signature,
            input_example=X_sample[:5]
        )

        return mlflow_path

    def _create_manifest(self, saved_files: Dict[str, Path], timestamp: str) -> Path:
        """Create manifest listing all saved files"""
        manifest = {
            'package_version': '1.0',
            'timestamp': timestamp,
            'files': {k: str(v) for k, v in saved_files.items()},
            'checksums': self._compute_checksums(saved_files)
        }

        manifest_path = self.output_dir / "MANIFEST.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return manifest_path

    def _compute_checksums(self, files: Dict[str, Path]) -> Dict:
        """Compute SHA256 checksums for verification"""
        import hashlib
        checksums = {}
        for name, path in files.items():
            if path.exists() and path.is_file():
                sha256 = hashlib.sha256()
                with open(path, 'rb') as f:
                    sha256.update(f.read())
                checksums[name] = sha256.hexdigest()
        return checksums
