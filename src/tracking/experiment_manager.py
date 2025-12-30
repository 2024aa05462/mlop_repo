import mlflow
import pandas as pd
from .mlflow_utils import MLflowTracker

class ExperimentManager:
    def __init__(self, experiment_name: str, tracking_uri: str = "./mlruns"):
        self.tracker = MLflowTracker(experiment_name, tracking_uri)
        self.experiment_name = experiment_name
        
    def compare_runs(self, metric: str = "roc_auc") -> pd.DataFrame:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None: return pd.DataFrame()
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=[f"metrics.{metric} DESC"])
        data = []
        for r in runs:
            data.append({
                'run_id': r.info.run_id,
                'run_name': r.data.tags.get('mlflow.runName', 'N/A'),
                **r.data.metrics, **r.data.params
            })
        return pd.DataFrame(data)

    def register_best_model(self, model_name: str, metric="roc_auc"):
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(self.experiment_name)
        runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=[f"metrics.{metric} DESC"], max_results=1)
        if not runs: return
        run_id = runs[0].info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", model_name)
