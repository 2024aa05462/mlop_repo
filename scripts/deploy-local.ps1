#!/bin/bash

# PowerShell adaptation for local Minikube deployment on Windows

# Start Minikube if not running
if (-not (minikube status | Select-String "Running")) {
    Write-Host "Starting Minikube..."
    minikube start --cpus=4 --memory=8192 --driver=docker
}

# Enable addons
Write-Host "Enabling addons..."
minikube addons enable ingress
minikube addons enable metrics-server

# Build Docker image
Write-Host "Building Docker image..."
minikube image build -t heart-disease-api:v1.0.0 -f docker/Dockerfile .

# Create namespace
Write-Host "Creating namespace..."
kubectl create namespace ml-models --dry-run=client -o yaml | kubectl apply -f -

# Deploy
Write-Host "Deploying application..."
kubectl apply -f k8s/base/ -n ml-models

# Wait
Write-Host "Waiting for pods..."
kubectl wait --for=condition=ready pod -l app=heart-disease-api -n ml-models --timeout=300s

# Info
$ip = minikube ip
$port = kubectl get svc heart-disease-api -n ml-models -o jsonpath='{.spec.ports[0].nodePort}'
Write-Host "Deployment complete!"
Write-Host "API URL: http://$ip:$port"
