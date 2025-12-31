# Heart Disease Prediction - MLOps Assignment

[![CI/CD Pipeline](https://github.com/shahrukhsaba/mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/shahrukhsaba/mlops/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**BITS Pilani - MLOps Assignment (S1-25_AIMLCZG523)**

A production-ready MLOps pipeline for predicting heart disease risk using the UCI Heart Disease dataset. This project demonstrates end-to-end ML model development, CI/CD, containerization, and cloud deployment.

🚀 **[Quick Start Guide (Docker)](QUICKSTART.md)** | 📄 **[Full Assignment Report](reports/MLOps_Assignment_Report.md)** | 🔗 **[GitHub Repository](https://github.com/shahrukhsaba/mlops)**

---

## 📊 Assignment Tasks Completion Summary

| # | Task | Marks | Status | Section |
|---|------|-------|--------|---------|
| 1 | Data Acquisition & EDA | 5 | ✅ Complete | [Step 1](#step-1-data-acquisition--eda-5-marks) |
| 2 | Feature Engineering & Model Development | 8 | ✅ Complete | [Step 2](#step-2-feature-engineering--model-development-8-marks) |
| 3 | Experiment Tracking | 5 | ✅ Complete | [Step 3](#step-3-experiment-tracking-5-marks) |
| 4 | Model Packaging & Reproducibility | 7 | ✅ Complete | [Step 4](#step-4-model-packaging--reproducibility-7-marks) |
| 5 | CI/CD Pipeline & Automated Testing | 8 | ✅ Complete | [Step 5](#step-5-cicd-pipeline--automated-testing-8-marks) |
| 6 | Model Containerization | 5 | ✅ Complete | [Step 6](#step-6-model-containerization-5-marks) |
| 7 | Production Deployment | 7 | ✅ Complete | [Step 7](#step-7-production-deployment-7-marks) |
| 8 | Monitoring & Logging | 3 | ✅ Complete | [Step 8](#step-8-monitoring--logging-3-marks) |
| 9 | Documentation & Reporting | 2 | ✅ Complete | [Step 9](#step-9-documentation--reporting-2-marks) |
| | **Total** | **50** | ✅ | |

---

## 🚀 Quick Start - Run the Complete Project

```bash
# Step 1: Clone and navigate to the project
git clone https://github.com/shahrukhsaba/mlops.git
cd mlops

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Download and prepare data
python scripts/download_data.py

# Step 4: Run notebooks (generates EDA, training outputs & screenshots)
python scripts/execute_notebooks.py

# Step 5: Train the model for production
python scripts/train_and_save_locally.py

# Step 6: Build Docker image
docker build -t heart-disease-api:latest .

# Step 7: Run Docker container
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest

# Step 8: Test the API
curl http://localhost:8000/health

# Predict with JSON input
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# Step 9: Open API documentation (Swagger UI)
open http://localhost:8000/docs
```

---

## 📋 Table of Contents

### Quick Start & Overview
- [Quick Start (Docker) - Separate Guide](QUICKSTART.md) ⭐
- [Quick Start (In-Page)](#-quick-start---run-the-complete-project)
- [Project Structure](#-project-structure)
- [Deliverables Checklist](#-deliverables-checklist)

### Assignment Steps (50 Marks Total)
- [Step 1: Data Acquisition & EDA (5 marks)](#step-1-data-acquisition--eda-5-marks)
- [Step 2: Feature Engineering & Model Development (8 marks)](#step-2-feature-engineering--model-development-8-marks)
- [Step 3: Experiment Tracking (5 marks)](#step-3-experiment-tracking-5-marks)
- [Step 4: Model Packaging & Reproducibility (7 marks)](#step-4-model-packaging--reproducibility-7-marks)
- [Step 5: CI/CD Pipeline & Automated Testing (8 marks)](#step-5-cicd-pipeline--automated-testing-8-marks)
- [Step 6: Model Containerization (5 marks)](#step-6-model-containerization-5-marks)
- [Step 7: Production Deployment (7 marks)](#step-7-production-deployment-7-marks)
- [Step 8: Monitoring & Logging (3 marks)](#step-8-monitoring--logging-3-marks)
- [Step 9: Documentation & Reporting (2 marks)](#step-9-documentation--reporting-2-marks)

### Additional Resources
- [Generated Screenshots](#-generated-screenshots)
- [Model Performance](#-model-performance)
- [Full Assignment Report](reports/MLOps_Assignment_Report.md)

---

## 📦 Deliverables Checklist

### a) GitHub Repository Contents

| Item | Status | Location |
|------|--------|----------|
| Code (src, api, scripts) | ✅ | `src/`, `api/`, `scripts/` |
| Dockerfile(s) | ✅ | `Dockerfile`, `docker-compose.yml` |
| requirements.txt | ✅ | `requirements.txt` |
| Cleaned dataset & download script | ✅ | `data/`, `scripts/download_data.py` |
| Jupyter notebooks (EDA, training) | ✅ | `notebooks/` (6 notebooks) |
| test/ folder with unit tests | ✅ | `tests/unit/`, `tests/integration/` |
| GitHub Actions workflow YAML | ✅ | `.github/workflows/ci-cd.yml` |
| Deployment manifests | ✅ | `k8s/` (4 YAML files) |
| Screenshot folder | ✅ | `screenshots/` (20 files) |
| Final written report | ✅ | `reports/MLOps_Assignment_Report.md` |

### b) Short Video
| Item | Status | Notes |
|------|--------|-------|
| End-to-end demo video | ⏳ Pending | User to record |

### c) Deployed API URL
| Deployment | URL | Status |
|------------|-----|--------|
| Docker (local) | http://localhost:8000 | ✅ Working |
| Kubernetes (local) | http://localhost:80 | ✅ Working |

---

## 🎯 Overview

This project builds a machine learning classifier to predict heart disease risk based on patient health data per the [BITS Pilani MLOps Assignment (S1-25_AIMLCZG523)](reports/MLOps_Assignment_Report.md).

**Dataset**: UCI Heart Disease Dataset (303 patients, 14 features, binary target)

**Problem**: Build a classifier to predict heart disease risk and deploy as a cloud-ready, monitored API.

---

## 📁 Project Structure

```
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD pipeline
├── api/
│   ├── app.py                     # FastAPI application
│   ├── predictor.py               # Prediction logic
│   ├── schemas.py                 # Pydantic schemas
│   ├── middleware/                # API middleware
│   └── monitoring/                # Prometheus metrics
├── data/
│   ├── raw/                       # Raw data files
│   └── processed/                 # Cleaned data
├── k8s/
│   ├── deployment.yaml            # Kubernetes deployment
│   ├── configmap.yaml             # Configuration
│   ├── ingress.yaml               # Ingress rules
│   └── namespace.yaml             # Namespace definition
├── models/
│   └── production/                # Production model artifacts
├── monitoring/
│   ├── prometheus/                # Prometheus configuration
│   └── grafana/                   # Grafana dashboards
├── notebooks/
│   ├── 01_eda.ipynb                          # Exploratory Data Analysis
│   ├── 02_feature_engineering_modeling.ipynb # Feature Engineering & Model Training
│   └── 03_mlflow_experiments.ipynb           # MLflow Experiment Tracking
├── scripts/
│   ├── download_data.py           # Data download script
│   ├── train_and_save_locally.py  # Local training script
│   └── execute_notebooks.py       # Run all notebooks with outputs
├── screenshots/                    # Generated visualizations
│   ├── 01_*.png                   # EDA screenshots
│   ├── 02_*.png                   # Model training screenshots
│   └── 03_*.png                   # MLflow experiment screenshots
├── src/
│   ├── data/                      # Data loading and preprocessing
│   ├── features/                  # Feature engineering
│   ├── models/                    # Model training
│   ├── monitoring/                # Drift detection
│   └── tracking/                  # MLflow utilities
├── tests/
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
├── Dockerfile                     # Docker image definition
├── docker-compose.yml             # Multi-container setup
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 📝 Detailed Execution Steps

### Prerequisites

- Python 3.9+ or 3.10+
- Docker Desktop (for containerization)
- Git

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/shahrukhsaba/mlops.git
cd mlops

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Data

```bash
python scripts/download_data.py
```

**Expected Output:**
```
Dataset download and processing complete!
Final dataset shape: (297, 14)
Target distribution: {0: 160, 1: 137}
```

### Step 3: Run Notebooks (Optional - for EDA & Experiments)

```bash
# Execute all notebooks and generate outputs/screenshots
python scripts/execute_notebooks.py
```

**Expected Output:**
```
======================================================================
          EXECUTING NOTEBOOKS WITH OUTPUT CAPTURE
======================================================================

📓 Executing: notebooks/01_eda.ipynb
   ✅ Executed and saved: notebooks/01_eda.ipynb

📓 Executing: notebooks/02_feature_engineering_modeling.ipynb
   ✅ Executed and saved: notebooks/02_feature_engineering_modeling.ipynb

📓 Executing: notebooks/03_mlflow_experiments.ipynb
   ✅ Executed and saved: notebooks/03_mlflow_experiments.ipynb
```

This will generate 13 screenshots in the `screenshots/` folder.

### Step 4: Train the Model

```bash
python scripts/train_and_save_locally.py
```

**Expected Output:**
```
Training Random Forest model for packaging...
Model saved to models/random_forest/model.pkl
```

### Step 5: Build Docker Image

```bash
docker build -t heart-disease-api:latest .
```

### Step 6: Run Docker Container

```bash
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest
```

### Step 7: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction with JSON input
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

**Expected Response:**
```json
{
  "prediction": 0,
  "confidence": 0.2737,
  "risk_level": "Low",
  "probability_no_disease": 0.7263,
  "probability_disease": 0.2737,
  "processing_time_ms": 11.93
}
```

### Step 8: View API Documentation

Open in browser: http://localhost:8000/docs

### Step 9: Stop Container (when done)

```bash
docker stop heart-disease-api
docker rm heart-disease-api
```

---

---

## Step 1: Data Acquisition & EDA (5 marks)

> **Assignment Requirement**: Obtain the dataset (provide download script or instructions). Clean and preprocess the data (handle missing values, encode features). Perform EDA with professional visualizations.

### 1.1 Download Dataset

The UCI Heart Disease dataset can be downloaded automatically:

```bash
python scripts/download_data.py
```

This script will:
- Download data from UCI Machine Learning Repository
- Clean and preprocess the data
- Save to `data/processed/heart_disease_clean.csv`
- Generate metadata in `data/processed/metadata.json`

### 1.2 Dataset Description

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numerical |
| sex | Sex (1=male, 0=female) | Binary |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numerical |
| chol | Serum cholesterol (mg/dl) | Numerical |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numerical |
| exang | Exercise induced angina | Binary |
| oldpeak | ST depression induced by exercise | Numerical |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0-3) | Numerical |
| thal | Thalassemia type | Categorical |
| target | Heart disease presence (0/1) | Binary |

### 1.3 EDA Visualizations (Screenshots)

| Screenshot | Description |
|------------|-------------|
| `01_numerical_histograms.png` | Distribution of numerical features |
| `01_correlation_heatmap.png` | Feature correlation matrix |
| `01_class_balance.png` | Target class distribution (46% positive, 54% negative) |
| `01_boxplots_by_target.png` | Box plots for outlier detection |
| `01_categorical_distributions.png` | Categorical feature distributions |

**Notebook**: `notebooks/01_eda.ipynb`

---

## Step 2: Feature Engineering & Model Development (8 marks)

> **Assignment Requirement**: Prepare the final ML features (scaling and encoding). Build and train at least two classification models. Document model selection and tuning process. Evaluate using cross-validation and relevant metrics.

### 2.1 Feature Engineering

**Engineered Features**:
- Age Groups (binned into decades)
- Cholesterol Categories (Normal/Borderline/High)
- Blood Pressure Categories (Normal/Elevated/High)
- Heart Rate Reserve (220 - age - max_heart_rate)
- Interaction Features (age × thalach, chol × trestbps)

### 2.2 Models Trained

| Model | Hyperparameter Grid |
|-------|---------------------|
| **Logistic Regression** | C: [0.01, 0.1, 1, 10], solver: [lbfgs, liblinear] |
| **Random Forest** | n_estimators: [100, 200], max_depth: [10, 20, None] |

### 2.3 Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.00% | 83.33% | 71.43% | 76.92% | 92.86% |
| **Random Forest** | **81.67%** | 81.48% | **78.57%** | **80.00%** | 91.52% |

**Selected Model**: Random Forest (best overall performance)

### 2.4 Model Training Screenshots

| Screenshot | Description |
|------------|-------------|
| `02_model_comparison.png` | Model metrics comparison |
| `02_roc_curve_comparison.png` | ROC curves for both models |
| `02_confusion_matrices.png` | Confusion matrices side-by-side |
| `02_feature_importance.png` | Random Forest feature importance |

**Notebook**: `notebooks/02_feature_engineering_modeling.ipynb`

---

## Step 3: Experiment Tracking (5 marks)

> **Assignment Requirement**: Integrate MLflow (or similar tool) for experiment tracking. Log parameters, metrics, artifacts, and plots for all runs.

### 3.1 MLflow Integration

All experiments are tracked with MLflow:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000
# Open: http://localhost:5000
```

### 3.2 Logged Items

| Item Type | Examples |
|-----------|----------|
| **Parameters** | C, n_estimators, max_depth, solver |
| **Metrics** | accuracy, precision, recall, f1, roc_auc |
| **Artifacts** | model.pkl, preprocessor.pkl, confusion_matrix.png, roc_curve.png |

### 3.3 Experiment Screenshots

| Screenshot | Description |
|------------|-------------|
| `03_lr_confusion_matrix.png` | Logistic Regression confusion matrix |
| `03_lr_roc_curve.png` | Logistic Regression ROC curve |
| `03_rf_confusion_matrix.png` | Random Forest confusion matrix |
| `03_rf_roc_curve.png` | Random Forest ROC curve |

**Notebook**: `notebooks/03_mlflow_experiments.ipynb`  
**MLflow Runs**: `mlruns/` directory (tracked in Git)

---

## Step 4: Model Packaging & Reproducibility (7 marks)

> **Assignment Requirement**: Save the final model in a reusable format. Write a clean requirements.txt. Provide a preprocessing pipeline/transformers to ensure full reproducibility.

### 4.1 Saved Model Artifacts

The following files are included in the repository for full reproducibility:

| File | Description |
|------|-------------|
| `models/production/model.pkl` | Trained RandomForestClassifier (pickle format) |
| `models/production/preprocessor.pkl` | ColumnTransformer for feature preprocessing |
| `models/production/model_metadata.json` | Model version, hyperparameters, performance metrics |
| `models/production/feature_names.json` | List of 19 feature names used by the model |
| `models/production/MODEL_CARD.md` | Model documentation and ethical considerations |
| `mlruns/` | MLflow experiment tracking data (parameters, metrics, artifacts) |
| `requirements.txt` | All Python dependencies with pinned versions |
| `src/features/feature_engineering.py` | Feature creation and preprocessing pipeline |

### 4.2 Preprocessing Pipeline

The preprocessing pipeline (`src/features/feature_engineering.py`) includes:

```python
# Feature Engineering
create_features(df)  # Creates: age_group, chol_cat, bp_cat, hr_reserve, interaction features

# Preprocessing Pipeline
build_preprocessing_pipeline(numerical_features, categorical_features)
# - Numerical: SimpleImputer(median) + StandardScaler
# - Categorical: SimpleImputer(most_frequent) + OneHotEncoder
```

### 4.3 Reproducibility

- **Random Seed**: 42 (fixed for all training)
- **Python Version**: 3.10+
- **scikit-learn Version**: 1.3.0
- **All dependencies pinned** in `requirements.txt`

### 4.4 Load Model for Inference

```python
import joblib

# Load model and preprocessor
model = joblib.load('models/production/model.pkl')
preprocessor = joblib.load('models/production/preprocessor.pkl')

# Make prediction
prediction = model.predict(preprocessed_data)
```

---

## Step 5: CI/CD Pipeline & Automated Testing (8 marks)

> **Assignment Requirement**: Write unit tests for data processing and model code. Create a GitHub Actions pipeline that includes linting, unit testing, and model training steps. Artifacts/logging for each workflow run.

### 5.1 Unit Tests

| Test File | Tests | Description |
|-----------|-------|-------------|
| `tests/unit/test_data_processing.py` | 15 | Data loading, validation, transformations |
| `tests/unit/test_model.py` | 12 | Predictions, metrics, risk classification |
| `tests/unit/test_preprocessing.py` | 11 | Feature engineering, preprocessing |
| `tests/integration/test_api_integration.py` | 8 | Full API integration tests |

**Total: 52+ tests**

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=api --cov-report=html
```

### 5.2 GitHub Actions Pipeline

**Workflow**: `.github/workflows/ci-cd.yml`

| Job | Description | Artifacts |
|-----|-------------|-----------|
| **lint** | flake8, black, isort | - |
| **test** | pytest with coverage | Coverage reports (XML, HTML) |
| **train** | Model training & validation | Model artifacts |
| **docker** | Build & test Docker image | Docker image (.tar.gz) |
| **integration** | API integration tests | Test results |
| **security** | bandit, safety scans | Security reports |
| **docs** | Generate documentation | API docs |

### 5.3 Pre-commit Hook

```bash
# Install pre-commit hook
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Or use pre-commit framework
pip install pre-commit && pre-commit install
```

**View CI/CD Results**: [GitHub Actions](https://github.com/shahrukhsaba/mlops/actions)

---

## Step 6: Model Containerization (5 marks)

> **Assignment Requirement**: Build a Docker container for the model-serving API (Flask or FastAPI recommended). Expose /predict endpoint, accept JSON input, return prediction and confidence. Container must be built and run locally with sample input.

### 6.1 Why FastAPI?

| Feature | FastAPI | Flask |
|---------|---------|-------|
| **Async Support** | ✅ Native | ❌ Requires extensions |
| **Auto Documentation** | ✅ Built-in (`/docs`, `/redoc`) | ❌ Manual setup |
| **Data Validation** | ✅ Pydantic (automatic) | ❌ Manual validation |
| **JSON Schema** | ✅ Auto-generated | ❌ Manual |
| **Performance** | ✅ Faster (async) | ⚠️ Slower |
| **Type Hints** | ✅ Required & validated | ❌ Optional |

### 6.2 API Features

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model status |
| `/predict` | POST | Make prediction (**JSON input**) |
| `/model-info` | GET | Model metadata and metrics |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI documentation |

### 6.3 Dockerfile

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn pandas numpy joblib pydantic
COPY api/ ./api/
COPY models/production/ ./models/production/
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.4 Build & Run Container

```bash
# Build Docker image
docker build -t heart-disease-api:latest .

# Run container
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest

# Or use Docker Compose (includes MLflow)
docker-compose up -d
```

### 6.5 Test Container with JSON Input

```bash
# Health check
curl http://localhost:8000/health

# Prediction with JSON input (required format)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

### Expected Response

```json
{
  "prediction": 0,
  "confidence": 0.2737,
  "risk_level": "Low",
  "probability_no_disease": 0.7263,
  "probability_disease": 0.2737,
  "processing_time_ms": 11.93
}
```

### Test Different Risk Profiles

```bash
# High-risk patient
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":70,"sex":1,"cp":2,"trestbps":180,"chol":300,"fbs":1,"restecg":1,"thalach":100,"exang":1,"oldpeak":4.0,"slope":2,"ca":3,"thal":2}'
# Response: {"prediction": 1, "confidence": 0.785, "risk_level": "High", ...}

# Low-risk patient
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"sex":0,"cp":0,"trestbps":120,"chol":180,"fbs":0,"restecg":0,"thalach":170,"exang":0,"oldpeak":0.0,"slope":1,"ca":0,"thal":1}'
# Response: {"prediction": 0, "confidence": 0.0992, "risk_level": "Low", ...}
```

### Stop Container

```bash
docker stop heart-disease-api
docker rm heart-disease-api
```

---

## Step 7: Production Deployment (7 marks)

> **Assignment Requirement**: Deploy the Dockerized API to a public cloud or local Kubernetes (GKE, EKS, AKS, or Minikube/Docker Desktop). Use a deployment manifest or Helm chart. Expose via Load Balancer or Ingress. Verify endpoints and provide deployment screenshots.

### 7.1 Deployment Options

| Option | Description | Used in This Project |
|--------|-------------|----------------------|
| **Docker Desktop Kubernetes** | Local K8s cluster | ✅ Primary |
| **Minikube** | Local K8s for development | ✅ Supported |
| **GKE/EKS/AKS** | Cloud Kubernetes | ✅ Manifests ready |

### 7.2 Kubernetes Manifests (Deployment Manifests)

The project uses **deployment manifests** (standard Kubernetes YAML):

```
k8s/
├── deployment.yaml    # Deployment + Service (LoadBalancer) + HPA
├── configmap.yaml     # Application configuration
├── ingress.yaml       # Ingress rules (optional, for domain routing)
└── namespace.yaml     # Namespace definition
```

| File | Resources | Description |
|------|-----------|-------------|
| `k8s/deployment.yaml` | Deployment, Service, HPA | Main deployment with LoadBalancer service and auto-scaling |
| `k8s/configmap.yaml` | ConfigMap | Application configuration (paths, ports, settings) |
| `k8s/ingress.yaml` | Ingress | Optional domain-based routing rules |
| `k8s/namespace.yaml` | Namespace | Isolated namespace for the application |

### 7.3 Resources in `deployment.yaml`

| Resource | API Version | Purpose |
|----------|-------------|---------|
| **Deployment** | apps/v1 | Manages 2 replicas with rolling updates |
| **Service** | v1 (**LoadBalancer**) | Exposes API on port 80 → container 8000 |
| **HorizontalPodAutoscaler** | autoscaling/v2 | Auto-scales pods based on CPU/memory |

### 7.4 Deployment Features

| Feature | Configuration |
|---------|---------------|
| **Replicas** | 2 pods (min), 5 pods (max with HPA) |
| **Service Type** | **LoadBalancer** (exposed on `localhost:80`) |
| **Health Checks** | Liveness & Readiness probes on `/health` |
| **Auto-scaling** | HPA based on CPU (70%) and Memory (80%) |
| **Resources** | 256Mi-512Mi memory, 250m-500m CPU |
| **Rolling Updates** | Zero-downtime deployments |
| **Prometheus** | Metrics scraping annotations |
| **Environment** | `ENV=production`, `LOG_LEVEL=INFO` |

### 7.5 Deploy to Docker Desktop Kubernetes

```bash
# Ensure Docker Desktop Kubernetes is enabled
# (Docker Desktop → Settings → Kubernetes → Enable)

# Build Docker image
docker build -t heart-disease-api:latest .

# Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml

# Verify deployment
kubectl get deployments
kubectl get pods
kubectl get svc
kubectl get hpa
```

### 7.6 Deploy to Minikube (Alternative)

```bash
# Start Minikube
minikube start

# Use Minikube's Docker daemon
eval $(minikube docker-env)

# Build image in Minikube
docker build -t heart-disease-api:latest .

# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/ingress.yaml

# Get service URL
minikube service heart-disease-api --url
```

### 7.7 Verify Deployment

```bash
# Check deployment status
kubectl get deployments
# NAME                READY   UP-TO-DATE   AVAILABLE   AGE
# heart-disease-api   2/2     2            2           3h

# Check running pods
kubectl get pods -o wide
# NAME                                 READY   STATUS    RESTARTS   AGE
# heart-disease-api-676668bdb5-877qm   1/1     Running   0          2m
# heart-disease-api-676668bdb5-x6lmb   1/1     Running   0          2m

# Check LoadBalancer service
kubectl get svc
# NAME                TYPE           EXTERNAL-IP   PORT(S)
# heart-disease-api   LoadBalancer   localhost     80:31775/TCP

# Check HorizontalPodAutoscaler
kubectl get hpa
# NAME                    REFERENCE                      MINPODS   MAXPODS   REPLICAS
# heart-disease-api-hpa   Deployment/heart-disease-api   2         5         2
```

### 7.8 Test Endpoints via LoadBalancer (Port 80)

```bash
# Health check
curl http://localhost:80/health
# {"status": "healthy", "model_loaded": true, ...}

# Model info
curl http://localhost:80/model-info
# {"model_type": "RandomForestClassifier", ...}

# Prediction with JSON
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
# {"prediction": 0, "confidence": 0.2737, "risk_level": "Low", ...}
```

### 7.9 Deployment Screenshots

See the `screenshots/` folder for deployment verification:

| Screenshot | Description |
|------------|-------------|
| `07_k8s_deployment_status.txt` | Deployments, pods, services, HPA |
| `07_k8s_api_verification.txt` | API endpoint tests via LoadBalancer |
| `07_k8s_pod_details.txt` | Pod describe and logs |
| `07_docker_container_status.txt` | Docker container status |

---

## Step 8: Monitoring & Logging (3 marks)

> **Assignment Requirement**: Integrate logging of API requests. Demonstrate simple monitoring (Prometheus + Grafana or API metrics/logs dashboard).

### 8.1 API Request Logging

All API requests are logged with structured logging:

```python
# Example log output
2025-12-31 00:30:08 - INFO - Prediction: 0, Confidence: 0.274, Risk: Low, Duration: 0.012s
2025-12-31 00:30:08 - INFO - Prediction: 1, Confidence: 0.785, Risk: High, Duration: 0.011s
```

**Logging features:**
- File logging: `logs/api.log`
- Console logging: stdout (visible in `docker logs`)
- Structured format: timestamp, level, message
- Request details: prediction, confidence, risk level, duration

### 8.2 Prometheus Metrics

The API exposes comprehensive metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

| Metric | Type | Description |
|--------|------|-------------|
| `app_info` | gauge | Application version and model status |
| `app_uptime_seconds` | gauge | API uptime in seconds |
| `model_loaded` | gauge | Model loading status (0/1) |
| `predictions_total` | counter | Total number of predictions |
| `predictions_success_total` | counter | Successful predictions |
| `predictions_errors_total` | counter | Failed predictions |
| `predictions_by_class{class="0"}` | counter | Predictions: No disease |
| `predictions_by_class{class="1"}` | counter | Predictions: Disease |
| `prediction_latency_avg_ms` | gauge | Average latency (ms) |
| `prediction_latency_total_ms` | counter | Total latency (ms) |

### 8.3 Monitoring Stack (Prometheus + Grafana)

```
monitoring/
├── docker-compose-monitoring.yml   # Full monitoring stack
├── prometheus/
│   └── prometheus.yml              # Prometheus scrape config
└── grafana/
    ├── dashboards/
    │   ├── dashboards.yml          # Dashboard provisioning
    │   └── heart-disease-api.json  # Pre-built dashboard
    └── datasources/
        └── prometheus.yml          # Prometheus datasource
```

### 8.4 Start Monitoring Stack

```bash
# Build API image first (if not already built)
docker build -t heart-disease-api:latest .

# Start Prometheus + Grafana + API
cd monitoring
docker-compose -f docker-compose-monitoring.yml up -d
```

### 8.5 Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| **API** | http://localhost:8000 | - |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin / admin |

### 8.6 Grafana Dashboard Features

The pre-built dashboard (`heart-disease-api.json`) includes:

- **Total Predictions** - Counter stat panel
- **Average Latency** - Latency gauge with thresholds
- **Model Status** - Loaded/Not loaded indicator
- **Uptime** - Application uptime
- **Predictions by Class** - Pie chart (Disease vs No Disease)
- **Prediction Counts Over Time** - Time series graph

### 8.7 Grafana Login & Dashboard Access

1. **Open Grafana**: http://localhost:3000
2. **Login**: Username `admin`, Password `admin`
3. **Find Dashboard**: Click ☰ menu → Dashboards → "Heart Disease API Dashboard"

**If dashboard doesn't appear automatically:**

```bash
# Import manually via Grafana UI:
# 1. Click ☰ menu → Dashboards → Import
# 2. Upload: monitoring/grafana/dashboards/heart-disease-api.json
# 3. Click Import
```

### 8.8 Verify Prometheus is Scraping

```bash
# Check targets
curl http://localhost:9090/api/v1/targets

# Query a metric
curl "http://localhost:9090/api/v1/query?query=predictions_total"
```

### 8.9 Monitoring Screenshots

See the `screenshots/` folder:

| Screenshot | Description |
|------------|-------------|
| `08_api_metrics.txt` | Prometheus metrics output |
| `08_api_logs.txt` | API request logs |
| `08_monitoring_stack.txt` | Monitoring stack configuration |

---

## Step 9: Documentation & Reporting (2 marks)

> **Assignment Requirement**: Submit a professional Markdown or PDF report including: Setup/install instructions, EDA and modelling choices, Experiment tracking summary, Architecture diagram, CI/CD and deployment workflow screenshots, Link to code repository.

### 9.1 Full Assignment Report

📄 **[View Full Assignment Report](reports/MLOps_Assignment_Report.md)**

The report includes:
- Setup/install instructions
- EDA and modelling choices  
- Experiment tracking summary
- Architecture diagram
- CI/CD and deployment workflow screenshots
- Link to code repository

### 9.2 Report Contents

| Section | Description |
|---------|-------------|
| 1. Executive Summary | Project overview and key results |
| 2. Data Acquisition & EDA | Dataset, preprocessing, visualizations |
| 3. Feature Engineering & Model Development | Features, models, tuning, metrics |
| 4. Experiment Tracking | MLflow integration and logged items |
| 5. Model Packaging & Reproducibility | Artifacts, requirements, reproducibility |
| 6. CI/CD Pipeline & Automated Testing | Tests, GitHub Actions workflow |
| 7. Model Containerization | Docker, FastAPI, endpoints |
| 8. Production Deployment | Kubernetes architecture, deployment |
| 9. Monitoring & Logging | Prometheus, Grafana, logging |
| 10. Conclusion & Future Work | Summary and improvements |
| 11. References | External resources |
| 12. Appendix | Repository structure, links, screenshots |

### 9.3 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline Architecture                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │  UCI Data   │───►│  Download   │───►│   Clean &   │             │
│   │  Repository │    │   Script    │    │  Preprocess │             │
│   └─────────────┘    └─────────────┘    └──────┬──────┘             │
│                                                 │                     │
│                      ┌──────────────────────────▼──────────────────┐ │
│                      │              MLflow Tracking                 │ │
│                      │  ┌─────────────────┬─────────────────┐      │ │
│                      │  │ Logistic Reg.   │  Random Forest  │      │ │
│                      │  │ Training        │  Training       │      │ │
│                      │  └─────────────────┴─────────────────┘      │ │
│                      └──────────────────────────┬──────────────────┘ │
│                                                 │                     │
│   ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐             │
│   │   GitHub    │───►│   CI/CD     │───►│   Docker    │             │
│   │   Actions   │    │   Pipeline  │    │   Image     │             │
│   └─────────────┘    └─────────────┘    └──────┬──────┘             │
│                                                 │                     │
│   ┌─────────────────────────────────────────────▼──────────────────┐ │
│   │                    Kubernetes Cluster                           │ │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │ │
│   │  │  Pod 1      │  │  Pod 2      │  │  Pod N      │            │ │
│   │  │  FastAPI    │  │  FastAPI    │  │  FastAPI    │            │ │
│   │  └─────────────┘  └─────────────┘  └─────────────┘            │ │
│   │           │              │              │                       │ │
│   │           └──────────────┼──────────────┘                       │ │
│   │                          │                                       │ │
│   │  ┌───────────────────────▼───────────────────────┐             │ │
│   │  │           LoadBalancer Service                │             │ │
│   │  │           (localhost:80)                      │             │ │
│   │  └───────────────────────┬───────────────────────┘             │ │
│   └──────────────────────────┼──────────────────────────────────────┘ │
│                              │                                       │
│   ┌──────────────────────────▼──────────────────────┐               │
│   │                  Monitoring Stack                │               │
│   │  ┌─────────────┐  ┌─────────────────────────┐  │               │
│   │  │ Prometheus  │  │       Grafana           │  │               │
│   │  │  :9090      │──│       :3000             │  │               │
│   │  └─────────────┘  └─────────────────────────┘  │               │
│   └─────────────────────────────────────────────────┘               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.4 Repository Links

| Resource | Link |
|----------|------|
| **GitHub Repository** | https://github.com/shahrukhsaba/mlops |
| **CI/CD Actions** | https://github.com/shahrukhsaba/mlops/actions |
| **Full Report** | [reports/MLOps_Assignment_Report.md](reports/MLOps_Assignment_Report.md) |

---

## 📸 Generated Screenshots

Running `python scripts/execute_notebooks.py` generates all notebook screenshots:

### EDA Screenshots (01_eda.ipynb)

| Screenshot | Description |
|------------|-------------|
| `01_class_balance.png` | Target class distribution |
| `01_numerical_histograms.png` | Distribution of numerical features |
| `01_correlation_heatmap.png` | Feature correlation matrix |
| `01_categorical_distributions.png` | Categorical feature distributions |
| `01_boxplots_by_target.png` | Features by target class |

### Model Training Screenshots (02_feature_engineering_modeling.ipynb)

| Screenshot | Description |
|------------|-------------|
| `02_model_comparison.png` | Model metrics comparison |
| `02_roc_curve_comparison.png` | ROC curves for both models |
| `02_confusion_matrices.png` | Confusion matrices side-by-side |
| `02_feature_importance.png` | Random Forest feature importance |

### MLflow Experiment Screenshots (03_mlflow_experiments.ipynb)

| Screenshot | Description |
|------------|-------------|
| `03_lr_confusion_matrix.png` | Logistic Regression confusion matrix |
| `03_lr_roc_curve.png` | Logistic Regression ROC curve |
| `03_rf_confusion_matrix.png` | Random Forest confusion matrix |
| `03_rf_roc_curve.png` | Random Forest ROC curve |

### Production Deployment Screenshots (Step 7)

| Screenshot | Description |
|------------|-------------|
| `07_k8s_deployment_status.txt` | Kubernetes deployments, pods, services, HPA |
| `07_k8s_api_verification.txt` | API endpoint tests via LoadBalancer |
| `07_k8s_pod_details.txt` | Pod describe and container logs |
| `07_docker_container_status.txt` | Docker container and image status |

---

## 📝 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.00% | 83.33% | 71.43% | 76.92% | **92.86%** |
| Random Forest | **81.67%** | 81.48% | **78.57%** | **80.00%** | 91.52% |

*Note: Logistic Regression achieved higher ROC-AUC while Random Forest had better overall accuracy.*

---

## 🔒 Security

- Input validation with Pydantic schemas
- Health checks for container orchestration
- Dependency vulnerability scanning with safety
- Code security analysis with Bandit

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Authors

- **Sk Shahrukh Saba** - MLOps Assignment

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- BITS Pilani for the MLOps course

