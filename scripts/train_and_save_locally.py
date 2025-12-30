import sys
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.features.feature_engineering import build_preprocessing_pipeline, create_features

def train_and_save():
    print("Training Random Forest model for packaging...")
    
    # Load raw data
    raw_path = "data/raw/heart_disease_raw.csv"
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    df = pd.read_csv(raw_path)
    
    # Basic target encoding if needed (assuming same as in train scripts)
    if 'num' in df.columns:
        df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        df.drop(columns=['num'], inplace=True)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Feature Engineering (Transformations on DataFrame)
    X_train_eng = create_features(X_train)
    
    # Identify features
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'age_group', 'chol_cat', 'bp_cat']
    # Ensure they exist (some might not be created if columns missing, but here assumes standard dataset)
    categorical_features = [c for c in categorical_features if c in X_train_eng.columns]
    
    numerical_features = [c for c in X_train_eng.columns if c not in categorical_features]
    
    # Build Pipeline
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])
    
    # Train
    model_pipeline.fit(X_train_eng, y_train)
    
    # Save
    output_dir = "models/random_forest"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model_pipeline, output_path)
    
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train_and_save()
