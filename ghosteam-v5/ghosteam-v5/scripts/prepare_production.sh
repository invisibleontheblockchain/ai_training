#!/bin/bash

# Production Deployment Preparation Script
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

print_header "🔧 Preparing Ghosteam V5 for Production Deployment"

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the ghosteam-v5 directory"
    exit 1
fi

# Check prerequisites
print_header "Checking Prerequisites"

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if kubectl can connect to cluster
if ! kubectl cluster-info > /dev/null 2>&1; then
    print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    print_status "Make sure you're connected to your production cluster:"
    print_status "  kubectl config current-context"
    print_status "  kubectl config use-context <your-production-context>"
    exit 1
fi

print_status "✅ kubectl connected to: $(kubectl config current-context)"

# Check Helm
if ! command -v helm &> /dev/null; then
    print_error "Helm is not installed. Please install Helm first."
    exit 1
fi

print_status "✅ Helm found: $(helm version --short)"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

print_status "✅ Docker found: $(docker --version)"

# Prompt for required environment variables
print_header "Setting Up Production Environment Variables"

# Function to prompt for variable
prompt_for_var() {
    local var_name=$1
    local description=$2
    local is_secret=${3:-false}
    
    if [ -z "${!var_name}" ]; then
        echo -n "$description: "
        if [ "$is_secret" = true ]; then
            read -s value
            echo ""
        else
            read value
        fi
        export $var_name="$value"
    else
        print_status "$var_name already set"
    fi
}

# Required variables
prompt_for_var "DOMAIN" "Production domain (e.g., ghosteam.com)"
prompt_for_var "S3_BUCKET" "S3 bucket for artifacts"
prompt_for_var "AWS_REGION" "AWS region" 
prompt_for_var "AWS_ACCESS_KEY_ID" "AWS Access Key ID"
prompt_for_var "AWS_SECRET_ACCESS_KEY" "AWS Secret Access Key" true
prompt_for_var "DB_PASSWORD" "Database password" true
prompt_for_var "MLFLOW_DB_PASSWORD" "MLflow database password" true
prompt_for_var "SECRET_KEY" "Application secret key (leave empty to generate)" 

# Generate secret key if not provided
if [ -z "$SECRET_KEY" ]; then
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    print_status "Generated secret key"
fi

prompt_for_var "ALERT_EMAIL" "Alert email address"
prompt_for_var "SLACK_WEBHOOK_URL" "Slack webhook URL (optional)"
prompt_for_var "GRAFANA_API_KEY" "Grafana API key (optional)"

# Docker registry
prompt_for_var "DOCKER_REGISTRY" "Docker registry (e.g., ghcr.io/yourorg)"
prompt_for_var "IMAGE_TAG" "Image tag for deployment (default: latest)"

if [ -z "$IMAGE_TAG" ]; then
    IMAGE_TAG="latest"
fi

# Create production environment file
print_status "Creating production environment file..."
envsubst < .env.production > .env.prod.tmp

# Create Kubernetes namespace
print_header "Setting Up Kubernetes Namespace"
kubectl create namespace ghosteam-v5 --dry-run=client -o yaml | kubectl apply -f -
print_status "✅ Namespace 'ghosteam-v5' ready"

# Create secrets
print_header "Creating Kubernetes Secrets"

# Database credentials
kubectl create secret generic db-credentials \
    --from-literal=username=ghosteam \
    --from-literal=password="$DB_PASSWORD" \
    --namespace=ghosteam-v5 \
    --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic mlflow-db-credentials \
    --from-literal=username=mlflow \
    --from-literal=password="$MLFLOW_DB_PASSWORD" \
    --namespace=ghosteam-v5 \
    --dry-run=client -o yaml | kubectl apply -f -

# AWS credentials
kubectl create secret generic aws-credentials \
    --from-literal=access-key-id="$AWS_ACCESS_KEY_ID" \
    --from-literal=secret-access-key="$AWS_SECRET_ACCESS_KEY" \
    --namespace=ghosteam-v5 \
    --dry-run=client -o yaml | kubectl apply -f -

# Application secrets
kubectl create secret generic app-secrets \
    --from-literal=secret-key="$SECRET_KEY" \
    --from-literal=alert-email="$ALERT_EMAIL" \
    --from-literal=slack-webhook-url="$SLACK_WEBHOOK_URL" \
    --namespace=ghosteam-v5 \
    --dry-run=client -o yaml | kubectl apply -f -

print_status "✅ Kubernetes secrets created"

# Create ConfigMap for environment variables
print_header "Creating Configuration"
kubectl create configmap app-config \
    --from-literal=domain="$DOMAIN" \
    --from-literal=s3-bucket="$S3_BUCKET" \
    --from-literal=aws-region="$AWS_REGION" \
    --from-literal=docker-registry="$DOCKER_REGISTRY" \
    --from-literal=image-tag="$IMAGE_TAG" \
    --namespace=ghosteam-v5 \
    --dry-run=client -o yaml | kubectl apply -f -

print_status "✅ Configuration created"

# Save deployment variables for later use
cat > deployment.env << EOF
export DOMAIN="$DOMAIN"
export S3_BUCKET="$S3_BUCKET"
export AWS_REGION="$AWS_REGION"
export AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
export DOCKER_REGISTRY="$DOCKER_REGISTRY"
export IMAGE_TAG="$IMAGE_TAG"
export DB_PASSWORD="$DB_PASSWORD"
export MLFLOW_DB_PASSWORD="$MLFLOW_DB_PASSWORD"
export SECRET_KEY="$SECRET_KEY"
export ALERT_EMAIL="$ALERT_EMAIL"
export SLACK_WEBHOOK_URL="$SLACK_WEBHOOK_URL"
export GRAFANA_API_KEY="$GRAFANA_API_KEY"
EOF

print_status "✅ Deployment variables saved to deployment.env"

# Verify S3 bucket access
print_header "Verifying AWS S3 Access"
if aws s3 ls "s3://$S3_BUCKET" > /dev/null 2>&1; then
    print_status "✅ S3 bucket access verified"
else
    print_warning "⚠️  Could not verify S3 bucket access. Please ensure:"
    print_warning "   1. Bucket '$S3_BUCKET' exists"
    print_warning "   2. AWS credentials have proper permissions"
    print_warning "   3. AWS CLI is configured"
fi

# Check cluster resources
print_header "Checking Cluster Resources"
nodes=$(kubectl get nodes --no-headers | wc -l)
print_status "✅ Cluster has $nodes nodes"

# Check if cert-manager is installed (for TLS)
if kubectl get crd certificates.cert-manager.io > /dev/null 2>&1; then
    print_status "✅ cert-manager found (TLS certificates will be managed)"
else
    print_warning "⚠️  cert-manager not found. TLS certificates will need manual setup"
fi

# Check if ingress controller is available
if kubectl get ingressclass > /dev/null 2>&1; then
    ingress_class=$(kubectl get ingressclass -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "none")
    if [ "$ingress_class" != "none" ]; then
        print_status "✅ Ingress controller found: $ingress_class"
    else
        print_warning "⚠️  No ingress controller found. External access will need manual setup"
    fi
fi

print_header "🎉 Production Preparation Complete!"

echo ""
echo "✅ Prerequisites verified"
echo "✅ Kubernetes namespace created"
echo "✅ Secrets and configuration created"
echo "✅ Environment variables saved"
echo ""

echo "Next steps:"
echo "1. Build and push images: ./scripts/deploy.sh --build"
echo "2. Deploy infrastructure: ./scripts/deploy.sh --infrastructure"
echo "3. Deploy application: ./scripts/deploy.sh --application"
echo "4. Or run full deployment: ./scripts/deploy.sh"
echo ""

echo "Deployment configuration:"
echo "  Domain: $DOMAIN"
echo "  Registry: $DOCKER_REGISTRY"
echo "  Tag: $IMAGE_TAG"
echo "  S3 Bucket: $S3_BUCKET"
echo "  Namespace: ghosteam-v5"
echo ""

print_status "Ready for production deployment! 🚀"
