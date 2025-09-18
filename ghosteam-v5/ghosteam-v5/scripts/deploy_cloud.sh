#!/bin/bash

# Ghosteam V5 Cloud Deployment Script
# This script deploys the complete MLOps system to a cloud instance
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

print_header "🚀 Ghosteam V5 Cloud Deployment"

# Get public IP
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com/ || curl -s http://ipinfo.io/ip || echo "localhost")
print_status "Deploying on instance: $PUBLIC_IP"

# Check prerequisites
print_header "Checking Prerequisites"

if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Installing..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

if ! command -v kubectl &> /dev/null; then
    print_error "kubectl not found. Please ensure K3s is installed."
    exit 1
fi

print_status "✅ Prerequisites verified"

# Setup environment
print_header "Setting Up Environment"

# Create cloud environment configuration
cat > .env.cloud << EOF
# Ghosteam V5 Cloud Environment Configuration
APP_NAME=Ghosteam V5
APP_VERSION=1.0.0
DEBUG=false
ENVIRONMENT=cloud

# Public access configuration
PUBLIC_IP=$PUBLIC_IP
DOMAIN=$PUBLIC_IP

# Database Configuration
DATABASE_URL=postgresql://ghosteam:cloudpassword123@postgres:5432/ghosteam_v5
REDIS_URL=redis://redis:6379

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_ARTIFACT_ROOT=file:///mlflow/artifacts

# Feast Feature Store
FEAST_REPO_PATH=./feast_repo
FEAST_ONLINE_STORE_TYPE=redis
FEAST_OFFLINE_STORE_TYPE=file

# Model Serving
MODEL_SERVING_HOST=0.0.0.0
MODEL_SERVING_PORT=8080
MODEL_CACHE_SIZE=10
MODEL_CACHE_TTL=3600

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Performance
WORKERS=2
MAX_WORKERS=8
WORKER_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
EOF

print_status "✅ Environment configuration created"

# Create cloud-optimized docker-compose
print_header "Creating Cloud Docker Compose Configuration"

cat > docker-compose.cloud.yml << 'EOF'
version: '3.8'

services:
  # PostgreSQL for MLflow backend
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: ghosteam_v5
      POSTGRES_USER: ghosteam
      POSTGRES_PASSWORD: cloudpassword123
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ghosteam"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis for Feast online store and caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MLflow Tracking Server
  mlflow:
    build: 
      context: .
      dockerfile: infrastructure/docker/mlflow/Dockerfile
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://ghosteam:cloudpassword123@postgres:5432/ghosteam_v5
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/mlflow
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Ghosteam V5 API
  api:
    build: .
    depends_on:
      mlflow:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://ghosteam:cloudpassword123@postgres:5432/ghosteam_v5
      - ENVIRONMENT=cloud
      - DEBUG=false
    ports:
      - "8080:8080"
    volumes:
      - ./src:/app/src
      - ./data:/app/data
      - ./feast_repo:/app/feast_repo
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./infrastructure/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
      - '--web.external-url=http://${PUBLIC_IP}:9090'
    restart: unless-stopped

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_ROOT_URL=http://${PUBLIC_IP}:3000
    volumes:
      - grafana_data:/var/lib/grafana
      - ./infrastructure/monitoring/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

  # Feast feature server
  feast:
    build:
      context: .
      dockerfile: infrastructure/docker/feast/Dockerfile
    ports:
      - "6566:6566"
    environment:
      - FEAST_REDIS_URL=redis://redis:6379
    volumes:
      - ./feast_repo:/feast_repo
    depends_on:
      redis:
        condition: service_healthy
    command: feast serve --host 0.0.0.0 --port 6566
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  mlflow_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: ghosteam-v5-cloud-network
EOF

print_status "✅ Cloud Docker Compose configuration created"

# Setup directories and sample data
print_header "Setting Up Data and Directories"

mkdir -p data/raw data/processed data/features data/models logs feast_repo

# Generate sample data for cloud deployment
print_status "Generating sample data..."
python3 << 'EOF'
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate user features
n_samples = 1000
user_ids = [f'user_{i:06d}' for i in range(n_samples)]

user_data = {
    'user_id': user_ids,
    'age': np.random.normal(35, 12, n_samples).astype(int).clip(18, 80),
    'activity_score': np.random.beta(2, 5, n_samples),
    'engagement_rate': np.random.beta(3, 7, n_samples),
    'total_sessions': np.random.poisson(20, n_samples),
    'avg_session_duration': np.random.exponential(300, n_samples),
    'is_premium': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
    'days_since_signup': np.random.exponential(100, n_samples).astype(int),
    'last_activity_score': np.random.beta(2, 5, n_samples),
    'event_timestamp': [
        datetime.now() - timedelta(hours=np.random.randint(0, 24*30))
        for _ in range(n_samples)
    ]
}

user_df = pd.DataFrame(user_data)

# Add correlations
premium_mask = user_df['is_premium']
user_df.loc[premium_mask, 'engagement_rate'] *= 1.5
user_df.loc[premium_mask, 'activity_score'] *= 1.3

# Clip values
user_df['engagement_rate'] = user_df['engagement_rate'].clip(0, 1)
user_df['activity_score'] = user_df['activity_score'].clip(0, 1)
user_df['avg_session_duration'] = user_df['avg_session_duration'].clip(10, 7200)

# Create training data with target
training_data = user_df.sample(800).copy()
target_prob = (
    0.3 +
    0.3 * training_data['engagement_rate'] +
    0.2 * training_data['activity_score'] +
    0.1 * training_data['is_premium'].astype(float) +
    0.1 * np.random.random(len(training_data))
)
training_data['target'] = np.random.binomial(1, target_prob.clip(0, 1))

# Save data
os.makedirs('data/features', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

user_df.to_parquet('data/features/user_features.parquet', index=False)
training_data.to_parquet('data/processed/training_data.parquet', index=False)

print(f"Generated {len(user_df)} user records and {len(training_data)} training samples")
EOF

print_status "✅ Sample data generated"

# Initialize Feast
print_header "Initializing Feast Feature Store"

cd feast_repo

cat > feature_store.yaml << EOF
project: ghosteam_v5
registry: feast_registry.db
provider: local
online_store:
    type: redis
    connection_string: localhost:6379
offline_store:
    type: file
EOF

# Copy feature definitions
cp ../src/data/feature_store/feast_config.py .

cd ..

print_status "✅ Feast configuration ready"

# Build and start services
print_header "Building and Starting Services"

print_status "Building Docker images..."
export PUBLIC_IP=$PUBLIC_IP
docker-compose -f docker-compose.cloud.yml build

print_status "Starting services..."
docker-compose -f docker-compose.cloud.yml up -d

# Wait for services to be ready
print_status "Waiting for services to start..."
sleep 30

# Check service health
print_header "Checking Service Health"

services=("postgres:5432" "redis:6379" "mlflow:5000" "api:8080")
for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -f "http://localhost:$port/health" > /dev/null 2>&1 || nc -z localhost $port; then
        print_status "✅ $name: Healthy"
    else
        print_warning "⚠️  $name: Starting up..."
    fi
done

# Apply Feast features
print_status "Applying Feast features..."
cd feast_repo
docker-compose -f ../docker-compose.cloud.yml exec -T feast feast apply || true
cd ..

# Train initial model
print_header "Training Initial Model"
python3 << 'EOF'
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

try:
    # Load training data
    training_data = pd.read_parquet('data/processed/training_data.parquet')
    
    # Prepare features and target
    feature_cols = ['age', 'activity_score', 'engagement_rate', 'total_sessions', 
                   'avg_session_duration', 'days_since_signup']
    X = training_data[feature_cols]
    y = training_data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    # Save model
    os.makedirs('data/models', exist_ok=True)
    joblib.dump(model, 'data/models/initial_model.pkl')
    
    print(f"✅ Model trained successfully!")
    print(f"   Training accuracy: {train_acc:.3f}")
    print(f"   Test accuracy: {test_acc:.3f}")
    
except Exception as e:
    print(f"❌ Model training failed: {e}")
EOF

print_header "🎉 Ghosteam V5 Cloud Deployment Complete!"

echo ""
echo "🌐 Access Your MLOps System:"
echo "   🚀 API:              http://$PUBLIC_IP:8080"
echo "   📓 API Docs:         http://$PUBLIC_IP:8080/docs"
echo "   📊 MLflow:           http://$PUBLIC_IP:5000"
echo "   📈 Grafana:          http://$PUBLIC_IP:3000 (admin/admin123)"
echo "   📉 Prometheus:       http://$PUBLIC_IP:9090"
echo "   🍃 Feast:            http://$PUBLIC_IP:6566"
echo ""

echo "🧪 Test Your Deployment:"
echo "   curl http://$PUBLIC_IP:8080/health"
echo "   curl http://$PUBLIC_IP:8080/models"
echo ""

echo "📊 Monitor Your System:"
echo "   docker-compose -f docker-compose.cloud.yml ps"
echo "   docker-compose -f docker-compose.cloud.yml logs -f api"
echo ""

echo "🔄 Manage Your Deployment:"
echo "   Stop:    docker-compose -f docker-compose.cloud.yml down"
echo "   Restart: docker-compose -f docker-compose.cloud.yml restart"
echo "   Update:  docker-compose -f docker-compose.cloud.yml pull && docker-compose -f docker-compose.cloud.yml up -d"
echo ""

# Save deployment info
cat > cloud-deployment-info.txt << EOF
# Ghosteam V5 Cloud Deployment Information
Deployed on: $(date)
Public IP: $PUBLIC_IP

# Access URLs:
API: http://$PUBLIC_IP:8080
API Docs: http://$PUBLIC_IP:8080/docs
MLflow: http://$PUBLIC_IP:5000
Grafana: http://$PUBLIC_IP:3000 (admin/admin123)
Prometheus: http://$PUBLIC_IP:9090
Feast: http://$PUBLIC_IP:6566

# Management Commands:
Status: docker-compose -f docker-compose.cloud.yml ps
Logs: docker-compose -f docker-compose.cloud.yml logs -f [service]
Stop: docker-compose -f docker-compose.cloud.yml down
Restart: docker-compose -f docker-compose.cloud.yml restart
EOF

print_status "✅ Deployment info saved to cloud-deployment-info.txt"
print_status "🚀 Your Ghosteam V5 MLOps system is now live in the cloud!"
