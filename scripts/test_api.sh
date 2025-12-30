#!/bin/bash

API_URL="http://localhost:8000"

echo "Testing API endpoints..."

# Test health endpoint
echo -e "\n1. Health Check:"
curl -X GET "$API_URL/health"

# Test single prediction
echo -e "\n2. Single Prediction:"
curl -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d @sample_data/sample_input.json

# Test batch prediction
echo -e "\n3. Batch Prediction:"
curl -X POST "$API_URL/predict/batch" \
  -H "Content-Type: application/json" \
  -d @sample_data/sample_batch_input.json

# Test model info
echo -e "\n4. Model Info:"
curl -X GET "$API_URL/model/info"

echo -e "\nAll tests completed"
