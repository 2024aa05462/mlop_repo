import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import pandas as pd
import logging
import json
import os
from src.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, model_name, output_dir="reports/model_performance"):
    """
    Evaluate model and save plots/metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    logger.info(f"Metrics for {model_name}: {metrics}")
    
    # Save metrics
    width_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(width_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics, y_pred, y_prob

def plot_roc_curve(y_test, y_prob, model_name, output_dir="reports/model_performance"):
    if y_prob is None:
        return
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc.png"))
    plt.close()

def plot_confusion_matrix(y_test, y_pred, model_name, output_dir="reports/model_performance"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

def plot_feature_importance(model, feature_names, model_name, output_dir="reports/model_performance"):
    # Check if model has feature importances
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        
    if importances is not None:
        indices = np.argsort(importances)[::-1]
        top_n = min(20, len(indices))
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances - {model_name}")
        plt.bar(range(top_n), importances[indices[:top_n]], align="center")
        
        # If feature names provided
        if feature_names is not None and len(feature_names) == len(importances):
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.png"))
        plt.close()
