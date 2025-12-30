import mlflow
import pandas as pd
import sys
import os

def compare_runs(experiment_name="heart_disease_classification", metric="roc_auc"):
    """Compare all runs in an experiment"""
    
    # Set tracking uri
    tracking_uri = "file:///" + os.path.abspath("./mlruns").replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment {experiment_name} not found.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )

    # Create comparison DataFrame
    comparison_data = []
    for run in runs:
        comparison_data.append({
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
            'model_type': run.data.tags.get('model_type', 'N/A'),
            **{k: v for k, v in run.data.metrics.items()},
            **{k: v for k, v in run.data.params.items()}
        })

    df = pd.DataFrame(comparison_data)
    
    # Sort if metric exists in columns
    sort_col = f"test_{metric}" if f"test_{metric}" in df.columns else metric
    if sort_col in df.columns:
        df = df.sort_values(by=sort_col, ascending=False)
        
    print(df.head())
    
    # Save to CSV
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/experiment_comparison.csv", index=False)
    print("Comparison saved to reports/experiment_comparison.csv")
    
    return df

if __name__ == "__main__":
    compare_runs()
