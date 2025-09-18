#!/bin/bash

# Ghosteam V5 Production Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load deployment environment if available
if [ -f "deployment.env" ]; then
    source deployment.env
fi

# Configuration
NAMESPACE="${K8S_NAMESPACE:-ghosteam-v5}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/ghosteam}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
DOMAIN="${DOMAIN:-ghosteam.com}"

# Function to print colored output
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

# Check prerequisites
check_prerequisites() {
    print_header "Checking deployment prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info > /dev/null 2>&1; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check Helm
    if ! command -v helm &> /dev/null; then
        print_error "Helm is not installed. Please install Helm first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    print_status "Prerequisites check passed"
}

# Build and push Docker images
build_and_push_images() {
    print_header "Building and pushing Docker images..."
    
    # Build main application image
    print_status "Building main application image..."
    docker build -t "${DOCKER_REGISTRY}/ghosteam-v5:${IMAGE_TAG}" .
    
    # Build MLflow image
    print_status "Building MLflow image..."
    docker build -t "${DOCKER_REGISTRY}/mlflow:${IMAGE_TAG}" -f infrastructure/docker/mlflow/Dockerfile .
    
    # Build Feast image
    print_status "Building Feast image..."
    docker build -t "${DOCKER_REGISTRY}/feast:${IMAGE_TAG}" -f infrastructure/docker/feast/Dockerfile .
    
    # Push images
    print_status "Pushing images to registry..."
    docker push "${DOCKER_REGISTRY}/ghosteam-v5:${IMAGE_TAG}"
    docker push "${DOCKER_REGISTRY}/mlflow:${IMAGE_TAG}"
    docker push "${DOCKER_REGISTRY}/feast:${IMAGE_TAG}"
    
    print_status "Images built and pushed successfully"
}

# Create Kubernetes secrets
create_secrets() {
    print_header "Creating Kubernetes secrets..."
    
    # Check if secrets already exist
    if kubectl get secret mlflow-secrets -n $NAMESPACE > /dev/null 2>&1; then
        print_warning "MLflow secrets already exist. Skipping creation."
    else
        print_status "Creating MLflow secrets..."
        kubectl create secret generic mlflow-secrets \
            --from-literal=backend-store-uri="${MLFLOW_BACKEND_STORE_URI}" \
            --namespace=$NAMESPACE
    fi
    
    if kubectl get secret aws-credentials -n $NAMESPACE > /dev/null 2>&1; then
        print_warning "AWS credentials already exist. Skipping creation."
    else
        print_status "Creating AWS credentials..."
        kubectl create secret generic aws-credentials \
            --from-literal=access-key-id="${AWS_ACCESS_KEY_ID}" \
            --from-literal=secret-access-key="${AWS_SECRET_ACCESS_KEY}" \
            --namespace=$NAMESPACE
    fi
    
    if kubectl get secret db-credentials -n $NAMESPACE > /dev/null 2>&1; then
        print_warning "Database credentials already exist. Skipping creation."
    else
        print_status "Creating database credentials..."
        kubectl create secret generic db-credentials \
            --from-literal=username="${DB_USERNAME}" \
            --from-literal=password="${DB_PASSWORD}" \
            --namespace=$NAMESPACE
    fi
    
    print_status "Secrets created successfully"
}

# Deploy infrastructure components
deploy_infrastructure() {
    print_header "Deploying infrastructure components..."
    
    # Create namespaces
    print_status "Creating namespaces..."
    kubectl apply -f infrastructure/kubernetes/namespaces.yaml
    
    # Deploy PostgreSQL
    print_status "Deploying PostgreSQL..."
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    helm upgrade --install postgresql bitnami/postgresql \
        --namespace $NAMESPACE \
        --set auth.postgresPassword="${POSTGRES_PASSWORD}" \
        --set auth.database="mlflow" \
        --set primary.persistence.size="20Gi" \
        --wait
    
    # Deploy Redis
    print_status "Deploying Redis..."
    helm upgrade --install redis bitnami/redis \
        --namespace $NAMESPACE \
        --set auth.enabled=false \
        --set master.persistence.size="10Gi" \
        --wait
    
    print_status "Infrastructure components deployed"
}

# Deploy MLflow
deploy_mlflow() {
    print_header "Deploying MLflow..."
    
    # Update image tag in deployment
    sed "s|ghosteam/mlflow:latest|${DOCKER_REGISTRY}/mlflow:${IMAGE_TAG}|g" \
        infrastructure/kubernetes/mlflow/deployment.yaml | kubectl apply -f -
    
    # Wait for deployment
    kubectl rollout status deployment/mlflow-tracking -n $NAMESPACE --timeout=300s
    
    print_status "MLflow deployed successfully"
}

# Deploy Feast
deploy_feast() {
    print_header "Deploying Feast..."
    
    # Create Feast ConfigMap
    kubectl create configmap feast-config \
        --from-file=feast_repo/feature_store.yaml \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Feast feature server
    kubectl apply -f infrastructure/kubernetes/feast/
    
    # Wait for deployment
    kubectl rollout status deployment/feast-feature-server -n $NAMESPACE --timeout=300s
    
    print_status "Feast deployed successfully"
}

# Deploy KServe
deploy_kserve() {
    print_header "Deploying KServe..."
    
    # Check if KServe is installed
    if ! kubectl get crd inferenceservices.serving.kserve.io > /dev/null 2>&1; then
        print_status "Installing KServe..."
        kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml
        
        # Wait for KServe to be ready
        kubectl wait --for=condition=ready pod -l control-plane=kserve-controller-manager -n kserve --timeout=300s
    fi
    
    # Deploy inference services
    kubectl apply -f infrastructure/kubernetes/kserve/
    
    print_status "KServe deployed successfully"
}

# Deploy main application
deploy_application() {
    print_header "Deploying main application..."
    
    # Update image tag in deployment
    sed "s|ghosteam/ghosteam-v5:latest|${DOCKER_REGISTRY}/ghosteam-v5:${IMAGE_TAG}|g" \
        infrastructure/kubernetes/app/deployment.yaml | kubectl apply -f -
    
    # Apply other application resources
    kubectl apply -f infrastructure/kubernetes/app/
    
    # Wait for deployment
    kubectl rollout status deployment/ghosteam-v5-api -n $NAMESPACE --timeout=300s
    
    print_status "Application deployed successfully"
}

# Deploy monitoring
deploy_monitoring() {
    print_header "Deploying monitoring stack..."
    
    # Add Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install Prometheus and Grafana
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace ghosteam-v5-monitoring \
        --create-namespace \
        --set grafana.adminPassword="${GRAFANA_ADMIN_PASSWORD:-admin123}" \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
        --wait --timeout=10m
    
    # Apply custom monitoring configurations
    kubectl apply -f infrastructure/kubernetes/monitoring/
    
    print_status "Monitoring stack deployed successfully"
}

# Run post-deployment checks
run_health_checks() {
    print_header "Running post-deployment health checks..."
    
    # Check MLflow
    print_status "Checking MLflow health..."
    kubectl wait --for=condition=ready pod -l app=mlflow-tracking -n $NAMESPACE --timeout=300s
    
    # Check API
    print_status "Checking API health..."
    kubectl wait --for=condition=ready pod -l app=ghosteam-v5-api -n $NAMESPACE --timeout=300s
    
    # Check Feast
    print_status "Checking Feast health..."
    kubectl wait --for=condition=ready pod -l app=feast-feature-server -n $NAMESPACE --timeout=300s
    
    # Test API endpoint
    API_URL=$(kubectl get service ghosteam-v5-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -n "$API_URL" ]; then
        print_status "Testing API endpoint..."
        if curl -f "http://${API_URL}/health" > /dev/null 2>&1; then
            print_status "API health check passed"
        else
            print_warning "API health check failed"
        fi
    else
        print_warning "Could not determine API URL"
    fi
    
    print_status "Health checks completed"
}

# Print deployment information
print_deployment_info() {
    print_header "🎉 Deployment Complete!"
    
    echo ""
    echo "Deployment Information:"
    echo "Namespace: $NAMESPACE"
    echo "Image Tag: $IMAGE_TAG"
    echo "Environment: $ENVIRONMENT"
    echo ""
    
    echo "Service URLs:"
    
    # Get service URLs
    MLFLOW_URL=$(kubectl get ingress mlflow-ingress -n $NAMESPACE -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "Not configured")
    API_URL=$(kubectl get ingress api-ingress -n $NAMESPACE -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "Not configured")
    GRAFANA_URL=$(kubectl get ingress grafana-ingress -n ghosteam-v5-monitoring -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "Not configured")
    
    echo "📊 MLflow UI:        https://$MLFLOW_URL"
    echo "🚀 API:              https://$API_URL"
    echo "📈 Grafana:          https://$GRAFANA_URL"
    echo ""
    
    echo "Useful commands:"
    echo "View pods:           kubectl get pods -n $NAMESPACE"
    echo "View services:       kubectl get services -n $NAMESPACE"
    echo "View logs:           kubectl logs -f deployment/ghosteam-v5-api -n $NAMESPACE"
    echo "Scale deployment:    kubectl scale deployment ghosteam-v5-api --replicas=3 -n $NAMESPACE"
    echo ""
    
    echo "Monitoring:"
    echo "Prometheus:          kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n ghosteam-v5-monitoring"
    echo "Grafana:             kubectl port-forward svc/prometheus-grafana 3000:80 -n ghosteam-v5-monitoring"
    echo ""
}

# Rollback function
rollback_deployment() {
    print_header "Rolling back deployment..."
    
    # Rollback main application
    kubectl rollout undo deployment/ghosteam-v5-api -n $NAMESPACE
    
    # Rollback MLflow
    kubectl rollout undo deployment/mlflow-tracking -n $NAMESPACE
    
    print_status "Rollback completed"
}

# Main deployment function
main() {
    print_header "🚀 Ghosteam V5 Production Deployment"
    echo ""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --rollback)
                rollback_deployment
                exit 0
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check required environment variables
    if [ -z "$MLFLOW_BACKEND_STORE_URI" ]; then
        print_error "MLFLOW_BACKEND_STORE_URI environment variable is required"
        exit 1
    fi
    
    if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        print_error "AWS credentials are required"
        exit 1
    fi
    
    # Run deployment steps
    check_prerequisites
    
    if [ "$SKIP_BUILD" != "true" ]; then
        build_and_push_images
    fi
    
    create_secrets
    deploy_infrastructure
    deploy_mlflow
    deploy_feast
    deploy_kserve
    deploy_application
    deploy_monitoring
    run_health_checks
    print_deployment_info
}

# Run main function
main "$@"
