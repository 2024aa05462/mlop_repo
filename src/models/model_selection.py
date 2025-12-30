import json
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compare_models(report_dir="reports/model_performance"):
    """
    Read metrics from JSON files and create a comparison report.
    """
    metrics_files = [f for f in os.listdir(report_dir) if f.endswith("_metrics.json")]
    
    results = []
    for f in metrics_files:
        model_name = f.replace("_metrics.json", "")
        with open(os.path.join(report_dir, f), "r") as json_file:
            metrics = json.load(json_file)
            metrics['model'] = model_name
            results.append(metrics)
            
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    df = df[cols]
    
    print("\nModel Comparison:")
    print(df.sort_values(by='f1', ascending=False))
    
    # Save comparison
    df.to_csv(os.path.join(report_dir, "model_comparison.csv"), index=False)
    
    return df

if __name__ == "__main__":
    compare_models()
