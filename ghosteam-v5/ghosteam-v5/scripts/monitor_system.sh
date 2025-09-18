#!/bin/bash

# System Monitoring Script
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check service health
check_service_health() {
    local service_name=$1
    local url=$2
    
    if curl -f "$url" > /dev/null 2>&1; then
        echo "✅ $service_name: Healthy"
    else
        echo "❌ $service_name: Unhealthy"
    fi
}

# Function to get service metrics
get_service_metrics() {
    local service_name=$1
    local container_name=$2
    
    echo "📊 $service_name Metrics:"
    
    # CPU and Memory usage
    stats=$(docker stats --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}" $container_name 2>/dev/null || echo "N/A\tN/A")
    echo "   CPU: $(echo $stats | awk '{print $1}')"
    echo "   Memory: $(echo $stats | awk '{print $2}')"
}

print_header "🔍 Ghosteam V5 System Monitoring Dashboard"

# Service Health Checks
print_header "Service Health Status"
check_service_health "API" "http://localhost:8080/health"
check_service_health "MLflow" "http://localhost:5000/health"
check_service_health "Grafana" "http://localhost:3000/api/health"
check_service_health "Prometheus" "http://localhost:9090/-/healthy"
check_service_health "MinIO" "http://localhost:9000/minio/health/live"

echo ""

# Container Status
print_header "Container Status"
docker-compose ps

echo ""

# Resource Usage
print_header "Resource Usage"
get_service_metrics "API" "ghosteam-v5-api-1"
get_service_metrics "MLflow" "ghosteam-v5-mlflow-1"
get_service_metrics "PostgreSQL" "ghosteam-v5-postgres-1"
get_service_metrics "Redis" "ghosteam-v5-redis-1"

echo ""

# API Metrics
print_header "API Metrics"
api_metrics=$(curl -s "http://localhost:8080/metrics" 2>/dev/null || echo '{"error": "API not responding"}')
if echo "$api_metrics" | grep -q "total_predictions"; then
    echo "📈 Total Predictions: $(echo $api_metrics | jq -r '.total_predictions // 0')"
    echo "⏱️  Average Latency: $(echo $api_metrics | jq -r '.average_latency_ms // 0')ms"
    echo "❌ Error Rate: $(echo $api_metrics | jq -r '.error_rate // 0')%"
else
    echo "⚠️  API metrics not available"
fi

echo ""

# MLflow Models
print_header "MLflow Models"
python3 << 'EOF'
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    models = client.search_registered_models()
    
    if models:
        print(f"📦 Registered Models: {len(models)}")
        for model in models[:3]:  # Show first 3
            latest_version = model.latest_versions[0] if model.latest_versions else None
            if latest_version:
                print(f"   • {model.name}: v{latest_version.version} ({latest_version.current_stage})")
    else:
        print("📦 No registered models found")
        
except Exception as e:
    print(f"⚠️  MLflow connection failed: {e}")
EOF

echo ""

# Disk Usage
print_header "Storage Usage"
echo "💾 Docker Volumes:"
docker system df

echo ""
echo "📁 Data Directories:"
du -sh data/* 2>/dev/null || echo "No data directories found"

echo ""

# Recent Logs (last 10 lines from each service)
print_header "Recent Activity (Last 10 Log Lines)"

services=("api" "mlflow" "postgres" "redis")
for service in "${services[@]}"; do
    echo ""
    echo "📋 $service logs:"
    docker-compose logs --tail=5 $service 2>/dev/null | tail -5 || echo "   No logs available"
done

echo ""

# System Recommendations
print_header "System Recommendations"

# Check if any containers are using too much memory
high_memory=$(docker stats --no-stream --format "table {{.Name}}\t{{.MemPerc}}" | awk 'NR>1 && $2+0 > 80 {print $1}')
if [ -n "$high_memory" ]; then
    print_warning "High memory usage detected in: $high_memory"
    echo "   Consider scaling or optimizing these services"
fi

# Check disk space
disk_usage=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$disk_usage" -gt 80 ]; then
    print_warning "Disk usage is high: ${disk_usage}%"
    echo "   Consider cleaning up old data or expanding storage"
fi

echo ""
echo "🔗 Quick Access Links:"
echo "   📊 MLflow UI:        http://localhost:5000"
echo "   📈 Grafana:          http://localhost:3000"
echo "   📉 Prometheus:       http://localhost:9090"
echo "   🚀 API Docs:         http://localhost:8080/docs"
echo "   🌸 Flower:           http://localhost:5555"
echo ""

echo "🔄 To refresh this dashboard, run: ./scripts/monitor_system.sh"
echo "📊 For detailed monitoring, visit Grafana: http://localhost:3000"
