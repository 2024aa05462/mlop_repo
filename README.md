# Heart Disease Prediction - MLOps Assignment

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/heart-disease-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/heart-disease-mlops/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready MLOps pipeline for predicting heart disease risk using the UCI Heart Disease dataset. This project demonstrates end-to-end ML model development, CI/CD, containerization, and cloud deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Data Acquisition](#data-acquisition)
- [Model Training](#model-training)
- [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)

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
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_mlflow_experiments.ipynb
├── scripts/
│   ├── download_data.py           # Data download script
│   └── train_and_save_locally.py  # Local training script
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

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10+
- Docker (optional, for containerization)
- Minikube/kubectl (optional, for Kubernetes)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-mlops.git
cd heart-disease-mlops
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
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
mlflow ui --port 5000
```

### Compare Experiments

View and compare runs at http://localhost:5000.

---

## 📝 Model Performance

| Model | Accuracy | ROC-AUC | CV Score |
|-------|----------|---------|----------|
| Logistic Regression | 0.82 | 0.88 | 0.85 ± 0.04 |
| Random Forest | 0.85 | 0.92 | 0.88 ± 0.03 |

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

- **Koushik Jana** - MLOps Assignment

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- BITS Pilani for the MLOps course
