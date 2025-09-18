#!/bin/bash

# Ghosteam V5 Environment Setup Script
set -e

echo "🚀 Setting up Ghosteam V5 Learning System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running on supported OS
check_os() {
    print_header "Checking operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Detected Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "Detected macOS"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_status "Docker Compose found: $(docker-compose --version)"
    
    # Check kubectl (optional for local development)
    if command -v kubectl &> /dev/null; then
        print_status "kubectl found: $(kubectl version --client --short 2>/dev/null || echo 'kubectl available')"
    else
        print_warning "kubectl not found. Kubernetes deployment will not be available."
    fi
    
    # Check Helm (optional)
    if command -v helm &> /dev/null; then
        print_status "Helm found: $(helm version --short 2>/dev/null || echo 'helm available')"
    else
        print_warning "Helm not found. Some Kubernetes deployments may not work."
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi
    print_status "Python found: $(python3 --version)"
}

# Setup Python environment
setup_python_env() {
    print_header "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    print_status "Python environment setup complete"
}

# Setup directories
setup_directories() {
    print_header "Setting up directories..."
    
    # Create data directories
    mkdir -p data/raw data/processed data/features data/models
    mkdir -p logs
    mkdir -p feast_repo
    
    # Create .gitkeep files
    touch data/raw/.gitkeep
    touch data/processed/.gitkeep
    touch data/features/.gitkeep
    touch data/models/.gitkeep
    
    print_status "Directories created"
}

# Setup environment file
setup_env_file() {
    print_header "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        
        # Generate secret key
        SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
        
        # Update .env file with generated values
        if [[ "$OS" == "macos" ]]; then
            sed -i '' "s/your-secret-key-here-change-in-production/$SECRET_KEY/" .env
        else
            sed -i "s/your-secret-key-here-change-in-production/$SECRET_KEY/" .env
        fi
        
        print_status ".env file created with generated secret key"
        print_warning "Please review and update .env file with your specific configuration"
    else
        print_status ".env file already exists"
    fi
}

# Setup Feast feature store
setup_feast() {
    print_header "Setting up Feast feature store..."
    
    cd feast_repo
    
    if [ ! -f "feature_store.yaml" ]; then
        print_status "Initializing Feast repository..."
        feast init
        
        # Copy our feature definitions
        cp ../src/data/feature_store/feast_config.py .
        
        print_status "Applying feature definitions..."
        feast apply
    else
        print_status "Feast repository already initialized"
    fi
    
    cd ..
    print_status "Feast setup complete"
}

# Start Docker services
start_docker_services() {
    print_header "Starting Docker services..."
    
    # Build images
    print_status "Building Docker images..."
    docker-compose build
    
    # Start services
    print_status "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    print_status "Checking service health..."
    
    # Check PostgreSQL
    if docker-compose exec -T postgres pg_isready -U mlflow > /dev/null 2>&1; then
        print_status "PostgreSQL is ready"
    else
        print_warning "PostgreSQL may not be ready yet"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        print_status "Redis is ready"
    else
        print_warning "Redis may not be ready yet"
    fi
    
    # Check MLflow
    if curl -f http://localhost:5000/health > /dev/null 2>&1; then
        print_status "MLflow is ready"
    else
        print_warning "MLflow may not be ready yet"
    fi
    
    print_status "Docker services started"
}

# Setup Kubernetes (optional)
setup_kubernetes() {
    if ! command -v kubectl &> /dev/null; then
        print_warning "Skipping Kubernetes setup (kubectl not found)"
        return
    fi
    
    print_header "Setting up Kubernetes environment..."
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info > /dev/null 2>&1; then
        print_warning "No Kubernetes cluster available. Skipping Kubernetes setup."
        return
    fi
    
    # Create namespace
    print_status "Creating Kubernetes namespace..."
    kubectl apply -f infrastructure/kubernetes/namespaces.yaml
    
    # Create secrets (you'll need to update these with real values)
    print_status "Creating Kubernetes secrets..."
    
    # MLflow secrets
    kubectl create secret generic mlflow-secrets \
        --from-literal=backend-store-uri="postgresql://mlflow:password@postgres:5432/mlflow" \
        --namespace=ghosteam-v5 \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # AWS credentials (placeholder)
    kubectl create secret generic aws-credentials \
        --from-literal=access-key-id="your-access-key" \
        --from-literal=secret-access-key="your-secret-key" \
        --namespace=ghosteam-v5 \
        --dry-run=client -o yaml | kubectl apply -f -
    
    print_status "Kubernetes setup complete"
    print_warning "Please update the secrets with real values before deploying to production"
}

# Install monitoring stack
setup_monitoring() {
    if ! command -v helm &> /dev/null; then
        print_warning "Skipping monitoring setup (Helm not found)"
        return
    fi
    
    if ! kubectl cluster-info > /dev/null 2>&1; then
        print_warning "No Kubernetes cluster available. Skipping monitoring setup."
        return
    fi
    
    print_header "Setting up monitoring stack..."
    
    # Add Helm repositories
    print_status "Adding Helm repositories..."
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install Prometheus and Grafana
    print_status "Installing Prometheus and Grafana..."
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace ghosteam-v5-monitoring \
        --create-namespace \
        --set grafana.adminPassword=admin123 \
        --wait --timeout=10m
    
    print_status "Monitoring stack installed"
}

# Run tests
run_tests() {
    print_header "Running tests..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run tests
    print_status "Running unit tests..."
    python -m pytest tests/unit/ -v
    
    print_status "Running integration tests..."
    python -m pytest tests/integration/ -v
    
    print_status "Tests completed"
}

# Print access information
print_access_info() {
    print_header "🎉 Setup Complete!"
    
    echo ""
    echo "Access your services:"
    echo "📊 MLflow UI:        http://localhost:5000"
    echo "📈 Grafana:          http://localhost:3000 (admin/admin123)"
    echo "📉 Prometheus:       http://localhost:9090"
    echo "🚀 API:              http://localhost:8080"
    echo "📓 API Docs:         http://localhost:8080/docs"
    echo "🌸 Flower (Celery):  http://localhost:5555"
    echo "📔 Jupyter:          http://localhost:8888 (token: ghosteam)"
    echo ""
    
    echo "Next steps:"
    echo "1. Review and update .env file with your configuration"
    echo "2. Check service health: docker-compose ps"
    echo "3. View logs: docker-compose logs -f [service_name]"
    echo "4. Run tests: ./scripts/run_tests.sh"
    echo "5. Deploy to production: ./scripts/deploy.sh"
    echo ""
    
    echo "For development:"
    echo "1. Activate Python environment: source venv/bin/activate"
    echo "2. Start API locally: uvicorn src.serving.api.main:app --reload"
    echo "3. Run Jupyter: jupyter lab --port 8889"
    echo ""
}

# Main execution
main() {
    print_header "🚀 Ghosteam V5 Learning System Setup"
    echo ""
    
    check_os
    check_prerequisites
    setup_directories
    setup_env_file
    setup_python_env
    setup_feast
    start_docker_services
    setup_kubernetes
    setup_monitoring
    run_tests
    print_access_info
}

# Run main function
main "$@"
