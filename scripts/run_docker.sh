#!/bin/bash

echo "Starting Docker container..."

docker run -d \
  --name heart-disease-api \
  -p 8000:8000 \
  --restart unless-stopped \
  heart-disease-api:latest

echo "Waiting for container to be healthy..."
sleep 5

# Check health
docker ps --filter name=heart-disease-api

echo "API is running at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
