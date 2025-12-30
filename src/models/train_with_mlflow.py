import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import yaml
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.features.feature_engineering import create_features, build_preprocessing_pipeline
from src.utils.metrics import calculate_metrics

def load_processed_data(data_dir="data/processed"):
    """Load processed data if available, or raw if not."""
    # For now, let's load raw and re-process to ensure we have the dataframe structure we expect
    # or load the parquet files. The user example uses load_processed_data.
    
    # Let's try to load raw and split, as per previous train.py logic which was safer
    raw_path = "data/raw/heart_disease_raw.csv"
    if not os.path.exists(raw_path):
        # try referencing from root
        raw_path = os.path.join(os.getcwd(), "data/raw/heart_disease_raw.csv")
        
    df = pd.read_csv(raw_path)
    if 'num' in df.columns:
        df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        df.drop(columns=['num'], inplace=True)
        
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Feature Engineering
    X = create_features(X)
    
    return X, y

def log_evaluation_plots(y_test, y_pred, y_pred_proba, model_name):
    """Generate and log plots."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    import seaborn as sns
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    mlflow.log_figure(fig_cm, f"confusion_matrix_{model_name}.png")
    plt.close()
    
    # ROC Curve
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc = plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        mlflow.log_figure(fig_roc, f"roc_curve_{model_name}.png")
        plt.close()

def main():
    # 1. Setup
    # Ensure mlruns directory absolute path
    tracking_uri = "file:///" + os.path.abspath("./mlruns").replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart_disease_classification")
    
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    # 2. Load data
    try:
        X, y = load_processed_data()
    except Exception as e:
        print(f"Error loading data: {e}. Please ensure data/raw exists.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Identify columns
    # We need to construct the pipeline for the model to work on raw-ish features
    # Assume default features lists
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'hr_reserve', 'age_x_thalach', 'chol_x_bp']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'age_group', 'chol_cat', 'bp_cat']
    
    # Filter only present columns
    num_cols = [c for c in num_cols if c in X_train.columns]
    cat_cols = [c for c in cat_cols if c in X_train.columns]
    
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)

    # 3. Define models and parameter grids
    models = {
        'logistic_regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'clf__C': [0.01, 0.1, 1, 10],
                'clf__penalty': ['l2'],
                'clf__solver': ['liblinear', 'saga']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [10, 20, None],
                'clf__min_samples_split': [2, 5]
            }
        }
    }

    # 4. Train each model with hyperparameter tuning
    best_runs = {}

    for model_name, config in models.items():
        with mlflow.start_run(run_name=f"{model_name}_tuning") as parent_run:
            
            # Create Pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('clf', config['model'])
            ])

            # Set tags for parent run
            mlflow.set_tags({
                "model_type": model_name,
                "phase": "hyperparameter_tuning",
                "dataset": "heart_disease_uci"
            })

            # Grid search (autolog handles child runs)
            grid_search = GridSearchCV(
                pipeline,
                config['params'],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)

            # Test set evaluation
            y_pred = grid_search.predict(X_test)
            y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

            # Log test metrics
            test_metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
            # MLflow autolog logs training metrics, let's log test metrics explicitly
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            # Log plots
            log_evaluation_plots(y_test, y_pred, y_pred_proba, model_name)

            # Save best model info
            best_runs[model_name] = {
                'run_id': parent_run.info.run_id,
                'best_params': grid_search.best_params_,
                'test_roc_auc': test_metrics.get('roc_auc', 0)
            }

    # 5. Print summary
    print("\n=== Training Summary ===")
    for model, info in best_runs.items():
        print(f"{model}: ROC-AUC = {info['test_roc_auc']:.4f}")
        print(f"  Best params: {info['best_params']}")
        print(f"  Run ID: {info['run_id']}\n")

if __name__ == "__main__":
    main()
