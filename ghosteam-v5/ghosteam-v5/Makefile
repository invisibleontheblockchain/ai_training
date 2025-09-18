# Ghosteam V5 Makefile

.PHONY: help install test lint format clean build deploy

# Default target
help:
	@echo "Ghosteam V5 Learning System"
	@echo ""
	@echo "Available commands:"
	@echo "  install     Install dependencies and setup environment"
	@echo "  test        Run test suite"
	@echo "  lint        Run code linting"
	@echo "  format      Format code with black"
	@echo "  clean       Clean up temporary files"
	@echo "  build       Build Docker images"
	@echo "  deploy      Deploy to production"
	@echo "  dev         Start development environment"
	@echo "  docs        Generate documentation"

# Install dependencies
install:
	@echo "Installing dependencies..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pre-commit install

# Run tests
test:
	@echo "Running tests..."
	./scripts/run_tests.sh

# Run linting
lint:
	@echo "Running linting..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	black --check src/ tests/

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

# Build Docker images
build:
	@echo "Building Docker images..."
	docker-compose build

# Deploy to production
deploy:
	@echo "Deploying to production..."
	./scripts/deploy.sh

# Start development environment
dev:
	@echo "Starting development environment..."
	docker-compose up -d
	@echo "Services started. Access points:"
	@echo "  API: http://localhost:8080"
	@echo "  MLflow: http://localhost:5000"
	@echo "  Grafana: http://localhost:3000"

# Stop development environment
dev-stop:
	@echo "Stopping development environment..."
	docker-compose down

# Generate documentation
docs:
	@echo "Generating documentation..."
	sphinx-build -b html docs/ docs/_build/

# Setup environment
setup:
	@echo "Setting up environment..."
	./scripts/setup_environment.sh

# Run API locally
api:
	@echo "Starting API locally..."
	uvicorn src.serving.api.main:app --reload --host 0.0.0.0 --port 8080

# Run Jupyter
jupyter:
	@echo "Starting Jupyter..."
	jupyter lab --port 8889 --no-browser

# Database migrations
migrate:
	@echo "Running database migrations..."
	alembic upgrade head

# Backup data
backup:
	@echo "Creating backup..."
	./scripts/backup.sh

# Restore data
restore:
	@echo "Restoring from backup..."
	./scripts/restore.sh

# Security scan
security:
	@echo "Running security scan..."
	bandit -r src/
	safety check

# Performance test
perf:
	@echo "Running performance tests..."
	pytest tests/performance/ -v

# Load test
load:
	@echo "Running load tests..."
	locust -f tests/load/locustfile.py --host=http://localhost:8080
