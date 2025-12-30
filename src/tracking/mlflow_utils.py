import mlflow
import mlflow.sklearn
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

class MLflowTracker:
    """Centralized MLflow tracking utilities"""

    def __init__(self, experiment_name: str, tracking_uri: str = "./mlruns"):
        """Initialize MLflow tracker"""
        # Resolve absolute path for tracking URI if it's a local file path
        # MLflow sometimes has issues with relative paths or mixed separators on Windows
        import os
        if tracking_uri.startswith("./") or tracking_uri.startswith(".\\"):
             tracking_uri = "file:///" + os.path.abspath(tracking_uri).replace("\\", "/")
        elif tracking_uri == "file:./mlruns":
             tracking_uri = "file:///" + os.path.abspath("./mlruns").replace("\\", "/")
             
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def start_run(self, run_name: Optional[str] = None, tags: Dict = None):
        """Start MLflow run with context manager"""
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params_dict(self, params: Dict[str, Any]):
        """Log multiple parameters"""
        mlflow.log_params(params)

    def log_metrics_dict(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics"""
        mlflow.log_metrics(metrics, step=step)

    def log_model_with_signature(self, model, artifact_path: str,
                                   signature=None, input_example=None):
        """Log model with input/output signature"""
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example
        )

    def log_figure(self, fig, artifact_name: str):
        """Log matplotlib figure as artifact"""
        mlflow.log_figure(fig, artifact_name)
        plt.close(fig)

    def log_dict_as_json(self, dictionary: Dict, filename: str):
        """Save and log dictionary as JSON artifact"""
        with open(filename, 'w') as f:
            json.dump(dictionary, f, indent=4)
        mlflow.log_artifact(filename)

    def set_tags(self, tags: Dict[str, str]):
        """Set multiple tags"""
        mlflow.set_tags(tags)

    def log_cv_results(self, cv_results: Dict):
        """Log cross-validation results"""
        for key, value in cv_results.items():
            if key.startswith('mean_'):
                mlflow.log_metric(key, value)
            elif key.startswith('std_'):
                mlflow.log_metric(key, value)
