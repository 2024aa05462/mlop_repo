# Makefile for Heart Disease MLOps Project
# ==========================================

.PHONY: help install data train test lint docker run clean mlflow all

# Default target
help:
	@echo "Heart Disease MLOps - Available Commands"
	@echo "========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install     - Install all dependencies"
	@echo "  make data        - Download and prepare dataset"
	@echo ""
	@echo "Development:"
	@echo "  make train       - Train models (Logistic Regression + Random Forest)"
	@echo "  make test        - Run all tests"
	@echo "  make lint        - Run linting checks"
	@echo "  make mlflow      - Start MLflow UI"
	@echo ""
	@echo "Deployment:"
	@echo "  make run         - Run API locally"
	@echo "  make docker      - Build Docker image"
	@echo "  make docker-run  - Run Docker container"
	@echo "  make k8s-deploy  - Deploy to Kubernetes (Minikube)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Clean temporary files"
	@echo "  make all         - Full pipeline (install, data, train, test)"

# ==========================================
# Setup Commands
# ==========================================

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Done!"

data:
	@echo "Downloading and preparing dataset..."
	python scripts/download_data.py
	@echo "Done!"

# ==========================================
# Development Commands
# ==========================================

train:
	@echo "Training models..."
	python src/models/train_models.py
	@echo "Training complete! Check models/production/"

train-basic:
	@echo "Training with basic script..."
	python src/models/train.py

test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov=api --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

lint:
	@echo "Running linting..."
	flake8 src/ api/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ api/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics

format:
	@echo "Formatting code..."
	black src/ api/ tests/
	isort src/ api/ tests/

mlflow:
	@echo "Starting MLflow UI at http://localhost:5000"
	mlflow ui --port 5000

# ==========================================
# Deployment Commands
# ==========================================

run:
	@echo "Starting API at http://localhost:8000"
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

run-prod:
	@echo "Starting API in production mode..."
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4

docker:
	@echo "Building Docker image..."
	docker build -t heart-disease-api:latest .
	@echo "Docker image built: heart-disease-api:latest"

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 --name heart-api heart-disease-api:latest

docker-stop:
	@echo "Stopping Docker container..."
	docker stop heart-api || true
	docker rm heart-api || true

docker-compose-up:
	@echo "Starting with docker-compose..."
	docker-compose up -d

docker-compose-down:
	@echo "Stopping docker-compose..."
	docker-compose down

k8s-deploy:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/ingress.yaml
	@echo "Deployment complete!"
	kubectl get pods
	kubectl get svc

k8s-status:
	@echo "Kubernetes status..."
	kubectl get pods
	kubectl get svc
	kubectl get deployments

k8s-logs:
	kubectl logs -l app=heart-disease-api --tail=100

k8s-delete:
	@echo "Deleting Kubernetes resources..."
	kubectl delete -f k8s/ || true

# ==========================================
# Utility Commands
# ==========================================

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml test-results.xml
	@echo "Done!"

clean-all: clean
	@echo "Cleaning all generated files..."
	rm -rf mlruns/
	rm -rf logs/
	rm -rf models/production/*
	@echo "Done!"

# ==========================================
# Full Pipeline
# ==========================================

all: install data train test
	@echo "Full pipeline complete!"

# ==========================================
# Health Checks
# ==========================================

health-check:
	@echo "Checking API health..."
	curl -s http://localhost:8000/health | python -m json.tool

predict-test:
	@echo "Testing prediction..."
	curl -X POST "http://localhost:8000/predict?age=63&sex=1&cp=3&trestbps=145&chol=233&fbs=1&restecg=0&thalach=150&exang=0&oldpeak=2.3&slope=0&ca=0&thal=1" | python -m json.tool
