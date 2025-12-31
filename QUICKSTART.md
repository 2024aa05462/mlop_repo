# Quick Start Guide: Run Locally with Docker Desktop

This guide provides step-by-step instructions to clone and run the Heart Disease Prediction API locally using Docker Desktop.

---

## Prerequisites

| Requirement | Version | Installation |
|-------------|---------|--------------|
| **Docker Desktop** | 4.0+ | [Download](https://www.docker.com/products/docker-desktop/) |
| **Git** | 2.0+ | [Download](https://git-scm.com/downloads) |

> **Note**: Ensure Docker Desktop is running before proceeding.

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/shahrukhsaba/mlops.git
cd mlops
```

---

## Step 2: Build Docker Image

```bash
docker build -t heart-disease-api:latest .
```

**Expected Output** (last few lines):
```
Successfully built <image_id>
Successfully tagged heart-disease-api:latest
```

---

## Step 3: Run Docker Container

```bash
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest
```

**Verify container is running:**
```bash
docker ps
```

**Expected Output:**
```
CONTAINER ID   IMAGE                      STATUS         PORTS
<id>           heart-disease-api:latest   Up X seconds   0.0.0.0:8000->8000/tcp
```

---

## Step 4: Test the API

### Health Check
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{"status": "healthy", "model_loaded": true, "uptime_seconds": 10.5}
```

### Make a Prediction
```bash
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

---

## Step 5: View API Documentation

Open in your browser:

| Documentation | URL |
|---------------|-----|
| **Swagger UI** | http://localhost:8000/docs |
| **ReDoc** | http://localhost:8000/redoc |
| **API Info** | http://localhost:8000/ |

---

## Step 6: Stop and Cleanup

```bash
# Stop the container
docker stop heart-disease-api

# Remove the container
docker rm heart-disease-api

# (Optional) Remove the image
docker rmi heart-disease-api:latest
```

---

## Quick Commands Reference

| Action | Command |
|--------|---------|
| Clone repo | `git clone https://github.com/shahrukhsaba/mlops.git && cd mlops` |
| Build image | `docker build -t heart-disease-api:latest .` |
| Run container | `docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest` |
| Check status | `docker ps` |
| View logs | `docker logs heart-disease-api` |
| Health check | `curl http://localhost:8000/health` |
| Predict | `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'` |
| Stop | `docker stop heart-disease-api && docker rm heart-disease-api` |

---

## One-Liner (Copy & Paste)

```bash
# Clone, build, and run in one command
git clone https://github.com/shahrukhsaba/mlops.git && \
cd mlops && \
docker build -t heart-disease-api:latest . && \
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest && \
echo "API running at http://localhost:8000"
```

---

## Troubleshooting

### Port 8000 already in use
```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process or use a different port
docker run -d --name heart-disease-api -p 8080:8000 heart-disease-api:latest
# Access at http://localhost:8080 instead
```

### Container won't start
```bash
# Check logs for errors
docker logs heart-disease-api

# Remove and rebuild
docker rm heart-disease-api
docker rmi heart-disease-api:latest
docker build -t heart-disease-api:latest .
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest
```

### Docker Desktop not running
```
Error: Cannot connect to the Docker daemon
```
**Solution**: Start Docker Desktop application and wait for it to initialize.

---

## Next Steps

- [Full README](README.md) - Complete project documentation
- [Assignment Report](reports/MLOps_Assignment_Report.md) - Detailed report
- [Kubernetes Deployment](README.md#step-7-production-deployment-7-marks) - Deploy to Kubernetes

---

**Repository**: https://github.com/shahrukhsaba/mlops

