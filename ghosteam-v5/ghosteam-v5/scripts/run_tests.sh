#!/bin/bash

# Ghosteam V5 Test Runner Script
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

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    print_status "Activating virtual environment..."
    source venv/bin/activate
fi

# Set test environment variables
export ENVIRONMENT=test
export DEBUG=true
export MLFLOW_TRACKING_URI=sqlite:///test_mlflow.db
export REDIS_URL=redis://localhost:6379/1

print_header "🧪 Running Ghosteam V5 Test Suite"

# Run linting
print_header "Running code quality checks..."
print_status "Running flake8..."
flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

print_status "Running black check..."
black --check src/ tests/

# Run unit tests
print_header "Running unit tests..."
pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run integration tests
print_header "Running integration tests..."
pytest tests/integration/ -v

# Run end-to-end tests if services are running
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    print_header "Running end-to-end tests..."
    pytest tests/e2e/ -v
else
    print_warning "Services not running, skipping e2e tests"
fi

# Generate test report
print_header "Generating test report..."
pytest tests/ --html=reports/test_report.html --self-contained-html

print_header "✅ Test suite completed!"
print_status "Coverage report: htmlcov/index.html"
print_status "Test report: reports/test_report.html"
