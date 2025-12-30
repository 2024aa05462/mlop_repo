# Heart Disease Prediction - MLOps Assignment

[![CI/CD Pipeline](https://github.com/shahrukhsaba/mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/shahrukhsaba/mlops/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**BITS Pilani - MLOps Assignment (S1-25_AIMLCZG523)**

A production-ready MLOps pipeline for predicting heart disease risk using the UCI Heart Disease dataset. This project demonstrates end-to-end ML model development, CI/CD, containerization, and cloud deployment.

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
curl -X POST "http://localhost:8000/predict?age=63&sex=1&cp=3&trestbps=145&chol=233&fbs=1&restecg=0&thalach=150&exang=0&oldpeak=2.3&slope=0&ca=0&thal=1"

# Step 9: Open API documentation
open http://localhost:8000/docs
```

---

## 📋 Table of Contents

- [Quick Start](#-quick-start---run-the-complete-project)
- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Detailed Execution Steps](#-detailed-execution-steps)
- [Data Acquisition](#-data-acquisition)
- [Model Training](#-model-training)
- [API Usage](#-api-usage)
- [Docker Deployment](#-docker-deployment)
- [Kubernetes Deployment](#-kubernetes-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Monitoring](#-monitoring)
- [Testing](#-testing)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)

---

## 🎯 Overview

This project builds a machine learning classifier to predict heart disease risk based on patient health data. It includes:

- **Data Pipeline**: Automated data ingestion, validation, and preprocessing
- **Model Training**: Logistic Regression and Random Forest with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for logging parameters, metrics, and artifacts
- **API Serving**: FastAPI-based prediction service
- **Containerization**: Docker image for portable deployment
- **Orchestration**: Kubernetes manifests for production deployment
- **CI/CD**: GitHub Actions workflow for automated testing and deployment
- **Monitoring**: Prometheus metrics and logging

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 EDA | Comprehensive exploratory data analysis with visualizations |
| 🔧 Feature Engineering | Automated feature creation and preprocessing pipeline |
| 🤖 Multiple Models | Logistic Regression & Random Forest with GridSearchCV |
| 📈 Experiment Tracking | MLflow logging of parameters, metrics, and artifacts |
| 🧪 Testing | Unit and integration tests with pytest |
| 🐳 Docker | Containerized API for portable deployment |
| ☸️ Kubernetes | Deployment manifests with HPA and health checks |
| 🔄 CI/CD | GitHub Actions for automated pipeline |
| 📡 Monitoring | Prometheus metrics and structured logging |

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

# Make a prediction
curl -X POST "http://localhost:8000/predict?age=63&sex=1&cp=3&trestbps=145&chol=233&fbs=1&restecg=0&thalach=150&exang=0&oldpeak=2.3&slope=0&ca=0&thal=1"
```

**Expected Response:**
```json
{
  "prediction": 0,
  "confidence": 0.2737,
  "risk_level": "Low",
  "probability_no_disease": 0.7263,
  "probability_disease": 0.2737,
  "processing_time_ms": 17.38
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

## 📥 Data Acquisition

### Download Dataset

The UCI Heart Disease dataset can be downloaded automatically:

```bash
python scripts/download_data.py
```

This script will:
- Download data from UCI Machine Learning Repository
- Clean and preprocess the data
- Save to `data/processed/heart_disease_clean.csv`
- Generate metadata in `data/processed/metadata.json`

### Dataset Description

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

---

## 🎓 Model Training

### Train Models

Train both Logistic Regression and Random Forest models:

```bash
python src/models/train_models.py
```

This will:
1. Load and preprocess the data
2. Train Logistic Regression with hyperparameter tuning
3. Train Random Forest with hyperparameter tuning
4. Log experiments to MLflow
5. Save the best model to `models/production/`

### View MLflow Experiments

```bash
mlflow ui --port 5000
```

Open http://localhost:5000 to view experiment tracking.

---

## 🌐 API Usage

### Run API Locally

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/docs` | GET | Swagger documentation |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict?age=63&sex=1&cp=3&trestbps=145&chol=233&fbs=1&restecg=0&thalach=150&exang=0&oldpeak=2.3&slope=0&ca=0&thal=1"
```

### Example Response

```json
{
  "prediction": 1,
  "confidence": 0.85
}
```

### Interactive Documentation

Open http://localhost:8000/docs for Swagger UI.

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t heart-disease-api:latest .
```

### Run Container

```bash
docker run -p 8000:8000 heart-disease-api:latest
```

### Using Docker Compose

```bash
docker-compose up -d
```

This starts both the API and MLflow server.

### Test Container

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict?age=63&sex=1&cp=3&trestbps=145&chol=233&fbs=1&restecg=0&thalach=150&exang=0&oldpeak=2.3&slope=0&ca=0&thal=1"
```

---

## ☸️ Kubernetes Deployment

### Prerequisites

- Minikube or a Kubernetes cluster
- kubectl configured

### Deploy to Minikube

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

### Verify Deployment

```bash
# Check pods
kubectl get pods

# Check services
kubectl get svc

# View logs
kubectl logs -l app=heart-disease-api
```

---

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) includes:

1. **Lint**: Code quality checks with flake8, black, isort
2. **Test**: Unit tests with pytest and coverage
3. **Train**: Model training and artifact generation
4. **Docker**: Build and test Docker image
5. **Integration**: API integration tests
6. **Security**: Bandit and safety vulnerability scans

### Trigger Pipeline

Push to `main` or `develop` branch, or open a pull request.

### View Results

Check the Actions tab in your GitHub repository.

---

## 📊 Monitoring

### Prometheus Metrics

The API exposes metrics at `/metrics`:

- `predictions_total`: Total predictions by class
- `prediction_confidence`: Confidence score distribution
- `prediction_latency_seconds`: Prediction latency
- `api_requests_total`: Total API requests

### Start Monitoring Stack

```bash
cd monitoring
docker-compose -f docker-compose-monitoring.yml up -d
```

### Access Dashboards

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Logs

API logs are written to `logs/api.log`.

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Unit Tests

```bash
pytest tests/unit/ -v
```

### Run Integration Tests

```bash
# Start the API first
uvicorn api.app:app --port 8000 &

# Run integration tests
pytest tests/integration/ -v
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov=api --cov-report=html
```

---

## 📈 MLflow Experiment Tracking

### Features Tracked

- **Parameters**: Hyperparameters, model configuration
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Artifacts**: Model files, preprocessor, plots

### Start MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

### Compare Experiments

View and compare runs at http://localhost:5000.

---

## 📸 Generated Screenshots

Running `python scripts/execute_notebooks.py` generates 13 screenshots:

| Notebook | Screenshot | Description |
|----------|------------|-------------|
| 01_eda | `01_class_balance.png` | Target class distribution |
| 01_eda | `01_numerical_histograms.png` | Distribution of numerical features |
| 01_eda | `01_correlation_heatmap.png` | Feature correlation matrix |
| 01_eda | `01_categorical_distributions.png` | Categorical feature distributions |
| 01_eda | `01_boxplots_by_target.png` | Features by target class |
| 02_modeling | `02_model_comparison.png` | Model metrics comparison |
| 02_modeling | `02_roc_curve_comparison.png` | ROC curves for both models |
| 02_modeling | `02_confusion_matrices.png` | Confusion matrices side-by-side |
| 02_modeling | `02_feature_importance.png` | Random Forest feature importance |
| 03_mlflow | `03_lr_confusion_matrix.png` | Logistic Regression confusion matrix |
| 03_mlflow | `03_lr_roc_curve.png` | Logistic Regression ROC curve |
| 03_mlflow | `03_rf_confusion_matrix.png` | Random Forest confusion matrix |
| 03_mlflow | `03_rf_roc_curve.png` | Random Forest ROC curve |

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

