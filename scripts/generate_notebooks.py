import nbformat as nbf
import os

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# 1. Exploratory Data Analysis (EDA) & Preprocessing\nThis notebook covers data loading, cleaning, and visualization."),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuration
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
PROCESSED_DATA_PATH = 'data/processed/heart_disease_clean.csv'

# Ensure directory exists
os.makedirs('data/processed', exist_ok=True)
"""),
        nbf.v4.new_markdown_cell("## Load and Clean Data"),
        nbf.v4.new_code_cell("""# Load Data
df = pd.read_csv(DATA_URL, names=COLUMNS)

# Handling missing values
df = df.replace('?', np.nan)
df = df.dropna()
df = df.astype(float)

# Convert target to binary (0: No Disease, 1: Disease)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Save cleaned data
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"Data saved to {PROCESSED_DATA_PATH}")
df.head()
"""),
        nbf.v4.new_markdown_cell("## Visualizations"),
        nbf.v4.new_code_cell("""# 1. Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title('Class Distribution')
plt.show()
"""),
        nbf.v4.new_code_cell("""# 2. Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
"""),
        nbf.v4.new_code_cell("""# 3. Age Distribution by Target
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='age', hue='target', kde=True, bins=20)
plt.title('Age Distribution by Heart Disease Status')
plt.show()
"""),
        nbf.v4.new_code_cell("""# 4. Pairplot of Key Features
key_features = ['age', 'chol', 'thalach', 'oldpeak', 'target']
sns.pairplot(df[key_features], hue='target', diag_kind='kde')
plt.show()
""")
    ]
    nb['cells'] = cells
    return nb

def create_training_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# 2. Model Training & Evaluation\nThis notebook implements the model training pipeline with MLflow tracking."),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup
os.makedirs('models/production', exist_ok=True)
mlflow.set_experiment("heart_disease_notebook_exp")
"""),
        nbf.v4.new_markdown_cell("## Data Preparation"),
        nbf.v4.new_code_cell("""# Load Processed Data
df = pd.read_csv('data/processed/heart_disease_clean.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Preprocessor
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
"""),
        nbf.v4.new_markdown_cell("## Model Training with MLflow"),
        nbf.v4.new_code_cell("""mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Notebook"):
    # Pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Hyperparameter Tuning
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20]
    }
    
    grid = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    # Evaluation
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Best Params: {grid.best_params_}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")
    
    # Save Artifacts Locally
    joblib.dump(best_model.named_steps['classifier'], 'models/production/model.pkl')
    joblib.dump(best_model.named_steps['preprocessor'], 'models/production/preprocessor.pkl')
    print("Models saved to models/production/")
"""),
        nbf.v4.new_markdown_cell("## Evaluation Plots"),
        nbf.v4.new_code_cell("""# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
""")
    ]
    nb['cells'] = cells
    return nb

def create_validation_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# 3. End-to-End Pipeline Validation\nValidating scripts, API endpoints, and artifacts."),
        nbf.v4.new_markdown_cell("## 1. Validate Artifacts Test"),
        nbf.v4.new_code_cell("""import os
import joblib

def check_artifacts():
    model_path = 'models/production/model.pkl'
    prep_path = 'models/production/preprocessor.pkl'
    
    if os.path.exists(model_path) and os.path.exists(prep_path):
        print("✅ Artifacts found.")
        try:
            model = joblib.load(model_path)
            prep = joblib.load(prep_path)
            print("✅ Artifacts loaded successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to load artifacts: {e}")
            return False
    else:
        print("❌ Artifacts missing.")
        return False

check_artifacts()
"""),
        nbf.v4.new_markdown_cell("## 2. Run Tests using Pytest"),
        nbf.v4.new_code_cell("""!pytest tests/ -v"""),
#         nbf.v4.new_markdown_cell("## 3. Test API (Requires running uvicorn separately)"),
#         nbf.v4.new_code_cell("""import requests

# # Assumes API is running on localhost:8000
# try:
#     # Health Check
#     resp = requests.get("http://localhost:8000/health")
#     print(f"Health Check: {resp.status_code} - {resp.json()}")

#     # Prediction Check
#     payload = {
#         "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, 
#         "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, 
#         "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
#     }
#     resp = requests.post("http://localhost:8000/predict", params=payload)
#     print(f"Prediction: {resp.status_code} - {resp.json()}")
    
# except Exception as e:
#     print(f"Could not connect to API: {e}")
#     print("Make sure to run 'uvicorn api.app:app --reload' in a terminal.")
# """)
    ]
    nb['cells'] = cells
    return nb

def main():
    os.makedirs('notebooks', exist_ok=True)
    
    nb1 = create_eda_notebook()
    with open('notebooks/01_EDA_and_Preprocessing.ipynb', 'w') as f:
        nbf.write(nb1, f)
        
    nb2 = create_training_notebook()
    with open('notebooks/02_Model_Training.ipynb', 'w') as f:
        nbf.write(nb2, f)
        
    nb3 = create_validation_notebook()
    with open('notebooks/03_Pipeline_Validation.ipynb', 'w') as f:
        nbf.write(nb3, f)
        
    print("Notebooks created in notebooks/ directory.")

if __name__ == "__main__":
    main()
