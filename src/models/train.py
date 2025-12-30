import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os

# 1. Load Data
print("Loading data...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, names=columns)

# Basic Cleaning
df = df.replace('?', np.nan)
df = df.dropna()
df = df.astype(float)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Save processed data
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/heart_disease_clean.csv', index=False)

# 2. Preprocessing
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 3. Model Training & Experiment Tracking
print("Starting MLflow tracking...")
mlflow.set_experiment("heart-disease-mvs")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Best"):
    # Main Model: Random Forest
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Simple Grid Search
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20]
    }
    
    grid = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='roc_auc')
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    # Evaluation
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"Best Params: {grid.best_params_}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    
    # 4. Save Artifacts
    os.makedirs('models/production', exist_ok=True)
    # Save the full pipeline as the model (simplifies inference)
    joblib.dump(best_model.named_steps['classifier'], 'models/production/model.pkl')
    joblib.dump(best_model.named_steps['preprocessor'], 'models/production/preprocessor.pkl')
    
    # Also save full pipeline for convenience if needed later
    joblib.dump(best_model, 'models/production/full_pipeline.pkl')
    
print("Training complete. Artifacts saved to models/production/")
