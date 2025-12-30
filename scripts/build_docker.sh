#!/bin/bash

echo "Building Docker image..."

docker build \
  -f docker/Dockerfile \
  -t heart-disease-api:latest \
  -t heart-disease-api:v1.0.0 \
  .

if [ $? -eq 0 ]; then
    echo "Docker image built successfully"
    docker images | grep heart-disease-api
else
    echo "Docker build failed"
    exit 1
fi
