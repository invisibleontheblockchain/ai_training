#!/bin/bash

# Ghosteam V5 Local Testing Startup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f "$url" > /dev/null 2>&1; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within $((max_attempts * 2)) seconds"
    return 1
}

print_header "🚀 Starting Ghosteam V5 Local Testing Environment"

# Check prerequisites
print_header "Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed"
    exit 1
fi

# Stop any existing containers
print_status "Stopping any existing containers..."
docker-compose down --remove-orphans

# Clean up old volumes if requested
if [ "$1" = "--clean" ]; then
    print_warning "Cleaning up old volumes..."
    docker-compose down -v
    docker system prune -f
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p data/raw data/processed data/features data/models logs feast_repo

# Setup environment file
if [ ! -f ".env" ]; then
    print_status "Creating .env file..."
    cp .env.example .env
    
    # Generate secret key
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i.bak "s/your-secret-key-here-change-in-production/$SECRET_KEY/" .env
    rm .env.bak
fi

# Build and start services
print_status "Building Docker images..."
docker-compose build

print_status "Starting services..."
docker-compose up -d

# Wait for core services
wait_for_service "PostgreSQL" "http://localhost:5432" || true
wait_for_service "Redis" "http://localhost:6379" || true
wait_for_service "MinIO" "http://localhost:9000/minio/health/live"
wait_for_service "MLflow" "http://localhost:5000/health"
wait_for_service "API" "http://localhost:8080/health"

# Check service status
print_header "Service Status Check"
echo ""
echo "Service Status:"
docker-compose ps

echo ""
echo "Service Health:"
services=("postgres:5432" "redis:6379" "mlflow:5000" "api:8080")
for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if docker-compose exec -T $name echo "OK" > /dev/null 2>&1; then
        echo "✅ $name: Running"
    else
        echo "❌ $name: Not responding"
    fi
done

print_header "🎉 Environment Started Successfully!"

echo ""
echo "Access Points:"
echo "📊 MLflow UI:        http://localhost:5000"
echo "📈 Grafana:          http://localhost:3000 (admin/admin123)"
echo "📉 Prometheus:       http://localhost:9090"
echo "🚀 API:              http://localhost:8080"
echo "📓 API Docs:         http://localhost:8080/docs"
echo "🌸 Flower (Celery):  http://localhost:5555"
echo "📔 Jupyter:          http://localhost:8888 (token: ghosteam)"
echo "💾 MinIO Console:    http://localhost:9001 (minioadmin/minioadmin)"
echo ""

echo "Next Steps:"
echo "1. Initialize sample data: ./scripts/setup_sample_data.sh"
echo "2. Run tests: ./scripts/test_continuous_learning.sh"
echo "3. Monitor logs: docker-compose logs -f [service_name]"
echo ""

print_status "Ready for testing! 🚀"
