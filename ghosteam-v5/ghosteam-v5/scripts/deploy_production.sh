#!/bin/bash

# Complete Production Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Load deployment environment
if [ -f "deployment.env" ]; then
    source deployment.env
    print_status "Loaded deployment configuration"
else
    print_error "deployment.env not found. Please run ./scripts/prepare_production.sh first"
    exit 1
fi

print_header "🚀 Ghosteam V5 Production Deployment"

# Parse command line arguments
BUILD_IMAGES=true
DEPLOY_INFRASTRUCTURE=true
DEPLOY_APPLICATION=true
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            BUILD_IMAGES=false
            shift
            ;;
        --skip-infrastructure)
            DEPLOY_INFRASTRUCTURE=false
            shift
            ;;
        --skip-application)
            DEPLOY_APPLICATION=false
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --build-only)
            BUILD_IMAGES=true
            DEPLOY_INFRASTRUCTURE=false
            DEPLOY_APPLICATION=false
            shift
            ;;
        --infrastructure-only)
            BUILD_IMAGES=false
            DEPLOY_INFRASTRUCTURE=true
            DEPLOY_APPLICATION=false
            shift
            ;;
        --application-only)
            BUILD_IMAGES=false
            DEPLOY_INFRASTRUCTURE=false
            DEPLOY_APPLICATION=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Usage: $0 [--skip-build] [--skip-infrastructure] [--skip-application] [--skip-tests]"
            echo "       $0 [--build-only] [--infrastructure-only] [--application-only]"
            exit 1
            ;;
    esac
done

# Verify prerequisites
print_header "Verifying Prerequisites"

if ! kubectl cluster-info > /dev/null 2>&1; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

current_context=$(kubectl config current-context)
print_status "Connected to cluster: $current_context"

# Confirm production deployment
if [[ "$current_context" == *"prod"* ]] || [[ "$current_context" == *"production"* ]]; then
    print_warning "⚠️  You are deploying to what appears to be a PRODUCTION cluster: $current_context"
    echo -n "Are you sure you want to continue? (yes/no): "
    read confirmation
    if [ "$confirmation" != "yes" ]; then
        print_status "Deployment cancelled"
        exit 0
    fi
fi

# Build and push images
if [ "$BUILD_IMAGES" = true ]; then
    print_header "Building and Pushing Docker Images"
    
    # Login to registry (assumes you're already authenticated)
    print_status "Building main application image..."
    docker build -t "${DOCKER_REGISTRY}/ghosteam-v5:${IMAGE_TAG}" .
    
    print_status "Building MLflow image..."
    docker build -t "${DOCKER_REGISTRY}/mlflow:${IMAGE_TAG}" -f infrastructure/docker/mlflow/Dockerfile .
    
    print_status "Building Feast image..."
    docker build -t "${DOCKER_REGISTRY}/feast:${IMAGE_TAG}" -f infrastructure/docker/feast/Dockerfile .
    
    print_status "Pushing images to registry..."
    docker push "${DOCKER_REGISTRY}/ghosteam-v5:${IMAGE_TAG}"
    docker push "${DOCKER_REGISTRY}/mlflow:${IMAGE_TAG}"
    docker push "${DOCKER_REGISTRY}/feast:${IMAGE_TAG}"
    
    print_status "✅ Images built and pushed successfully"
fi

# Deploy infrastructure
if [ "$DEPLOY_INFRASTRUCTURE" = true ]; then
    print_header "Deploying Infrastructure Components"
    
    # Add Helm repositories
    print_status "Adding Helm repositories..."
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Deploy PostgreSQL
    print_status "Deploying PostgreSQL..."
    helm upgrade --install postgresql bitnami/postgresql \
        --namespace ghosteam-v5 \
        --set auth.postgresPassword="$DB_PASSWORD" \
        --set auth.database="ghosteam_v5" \
        --set primary.persistence.size="50Gi" \
        --set primary.resources.requests.memory="1Gi" \
        --set primary.resources.requests.cpu="500m" \
        --set primary.resources.limits.memory="2Gi" \
        --set primary.resources.limits.cpu="1000m" \
        --wait --timeout=10m
    
    # Deploy Redis
    print_status "Deploying Redis..."
    helm upgrade --install redis bitnami/redis \
        --namespace ghosteam-v5 \
        --set auth.enabled=false \
        --set master.persistence.size="20Gi" \
        --set master.resources.requests.memory="512Mi" \
        --set master.resources.requests.cpu="250m" \
        --set master.resources.limits.memory="1Gi" \
        --set master.resources.limits.cpu="500m" \
        --wait --timeout=10m
    
    # Deploy monitoring stack
    print_status "Deploying monitoring stack..."
    kubectl create namespace ghosteam-v5-monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace ghosteam-v5-monitoring \
        --set grafana.adminPassword="$GRAFANA_ADMIN_PASSWORD" \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=10Gi \
        --wait --timeout=15m
    
    print_status "✅ Infrastructure deployed successfully"
fi

# Update image tags in manifests
print_status "Updating image tags in manifests..."
find infrastructure/kubernetes -name "*.yaml" -exec sed -i.bak "s|ghosteam/ghosteam-v5:latest|${DOCKER_REGISTRY}/ghosteam-v5:${IMAGE_TAG}|g" {} \;
find infrastructure/kubernetes -name "*.yaml" -exec sed -i.bak "s|ghosteam/mlflow:latest|${DOCKER_REGISTRY}/mlflow:${IMAGE_TAG}|g" {} \;
find infrastructure/kubernetes -name "*.yaml" -exec sed -i.bak "s|ghosteam/feast:latest|${DOCKER_REGISTRY}/feast:${IMAGE_TAG}|g" {} \;
find infrastructure/kubernetes -name "*.yaml" -exec sed -i.bak "s|api.ghosteam.com|api.${DOMAIN}|g" {} \;
find infrastructure/kubernetes -name "*.yaml" -exec sed -i.bak "s|mlflow.ghosteam.com|mlflow.${DOMAIN}|g" {} \;

# Deploy application
if [ "$DEPLOY_APPLICATION" = true ]; then
    print_header "Deploying Application Components"
    
    # Wait for infrastructure to be ready
    print_status "Waiting for infrastructure to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql -n ghosteam-v5 --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n ghosteam-v5 --timeout=300s
    
    # Deploy MLflow
    print_status "Deploying MLflow..."
    kubectl apply -f infrastructure/kubernetes/mlflow/
    kubectl rollout status deployment/mlflow-tracking -n ghosteam-v5 --timeout=300s
    
    # Deploy Feast
    print_status "Deploying Feast feature server..."
    kubectl apply -f infrastructure/kubernetes/feast/
    kubectl rollout status deployment/feast-feature-server -n ghosteam-v5 --timeout=300s
    
    # Deploy main application
    print_status "Deploying main application..."
    kubectl apply -f infrastructure/kubernetes/app/
    kubectl rollout status deployment/ghosteam-v5-api -n ghosteam-v5 --timeout=300s
    
    # Deploy KServe inference services
    if kubectl get crd inferenceservices.serving.kserve.io > /dev/null 2>&1; then
        print_status "Deploying KServe inference services..."
        kubectl apply -f infrastructure/kubernetes/kserve/
    else
        print_warning "KServe not installed, skipping inference services"
    fi
    
    print_status "✅ Application deployed successfully"
fi

# Clean up temporary files
find infrastructure/kubernetes -name "*.yaml.bak" -delete

# Run health checks
print_header "Running Health Checks"

print_status "Checking pod status..."
kubectl get pods -n ghosteam-v5

print_status "Checking service endpoints..."
kubectl get services -n ghosteam-v5

print_status "Checking ingress..."
kubectl get ingress -n ghosteam-v5

# Test API health
print_status "Testing API health..."
API_URL="https://api.${DOMAIN}"
if curl -f "$API_URL/health" > /dev/null 2>&1; then
    print_status "✅ API health check passed"
else
    print_warning "⚠️  API health check failed - may need time to start"
fi

# Run production tests
if [ "$SKIP_TESTS" = false ]; then
    print_header "Running Production Validation Tests"
    
    # Create a test pod to run validation
    kubectl run production-test \
        --image="${DOCKER_REGISTRY}/ghosteam-v5:${IMAGE_TAG}" \
        --rm -i --restart=Never \
        --namespace=ghosteam-v5 \
        -- python -c "
import requests
import sys

try:
    # Test API health
    response = requests.get('http://ghosteam-v5-api-service/health', timeout=10)
    if response.status_code == 200:
        print('✅ Internal API health check passed')
    else:
        print('❌ Internal API health check failed')
        sys.exit(1)
        
    # Test feature store
    response = requests.get('http://feast-feature-server-service:6566/health', timeout=10)
    if response.status_code == 200:
        print('✅ Feature store health check passed')
    else:
        print('❌ Feature store health check failed')
        
    print('✅ Production validation completed')
    
except Exception as e:
    print(f'❌ Production validation failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_status "✅ Production validation tests passed"
    else
        print_warning "⚠️  Some production validation tests failed"
    fi
fi

print_header "🎉 Deployment Complete!"

echo ""
echo "🚀 Ghosteam V5 has been deployed to production!"
echo ""
echo "📊 Access Points:"
echo "   API:              https://api.${DOMAIN}"
echo "   API Docs:         https://api.${DOMAIN}/docs"
echo "   MLflow:           https://mlflow.${DOMAIN}"
echo "   Grafana:          https://grafana.${DOMAIN}"
echo ""
echo "🔍 Monitoring:"
echo "   kubectl get pods -n ghosteam-v5"
echo "   kubectl logs -f deployment/ghosteam-v5-api -n ghosteam-v5"
echo "   ./scripts/monitor_system.sh"
echo ""
echo "📈 Scaling:"
echo "   kubectl scale deployment ghosteam-v5-api --replicas=5 -n ghosteam-v5"
echo ""
echo "🔄 Updates:"
echo "   ./scripts/deploy_production.sh --application-only"
echo ""

print_status "Production deployment successful! 🎉"
