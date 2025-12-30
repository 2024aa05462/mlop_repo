$NAMESPACE="ml-models"
$SERVICE_NAME="heart-disease-api"

Write-Host "Verifying deployment..."

# Check namespace
kubectl get namespace $NAMESPACE

# Check deployment
kubectl get deployment $SERVICE_NAME -n $NAMESPACE

# Check pods
kubectl get pods -l app=$SERVICE_NAME -n $NAMESPACE

# Check service
kubectl get svc $SERVICE_NAME -n $NAMESPACE
