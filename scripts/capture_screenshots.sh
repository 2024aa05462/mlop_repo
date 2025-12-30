#!/bin/bash

# Automated screenshot capture for reporting (WSL/Linux/Bash)

SCREENSHOT_DIR="docs/screenshots"
mkdir -p $SCREENSHOT_DIR/{eda,mlflow,ci-cd,docker,kubernetes,monitoring}

echo "📸 Capturing data snapshots..."

# 2. Docker
echo "Capturing Docker state..."
docker ps > $SCREENSHOT_DIR/docker/docker_ps.txt
curl -s http://localhost:8000/health | jq '.' > $SCREENSHOT_DIR/docker/health_response.json
# Note: Sample input needs to exist
if [ -f "sample_data/sample_input.json" ]; then
    curl -s http://localhost:8000/predict -X POST -H "Content-Type: application/json" \
      -d @sample_data/sample_input.json | jq '.' > $SCREENSHOT_DIR/docker/predict_response.json
fi

# 3. Kubernetes
echo "Capturing Kubernetes state..."
kubectl get pods -n ml-models -o wide > $SCREENSHOT_DIR/kubernetes/pods.txt
kubectl get svc -n ml-models > $SCREENSHOT_DIR/kubernetes/services.txt
kubectl describe deployment heart-disease-api -n ml-models > $SCREENSHOT_DIR/kubernetes/deployment_describe.txt

# 4. Monitoring
curl -s http://localhost:9090/api/v1/targets | jq '.' > $SCREENSHOT_DIR/monitoring/prometheus_targets.json
curl -s http://localhost:8000/metrics > $SCREENSHOT_DIR/monitoring/metrics_output.txt

echo "✅ Data snapshots captured! Please manually take UI screenshots (Grafana, MLflow) and place them in docs/screenshots/."
