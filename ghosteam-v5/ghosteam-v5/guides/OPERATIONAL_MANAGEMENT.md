# 🔧 OPERATIONAL MANAGEMENT GUIDE

## 🎯 **OBJECTIVE**
Comprehensive guide for monitoring, maintaining, and troubleshooting your Ghosteam V5 MLOps system in production.

---

## 📊 **MONITORING & OBSERVABILITY**

### **1. System Health Monitoring**

#### **Health Check Dashboard**
```python
# src/monitoring/health.py
import psutil
import time
import requests
from typing import Dict, Any
import mlflow
from sqlalchemy import create_engine, text
import redis

class SystemHealthMonitor:
    def __init__(self, database_url: str, redis_url: str, mlflow_url: str):
        self.database_url = database_url
        self.redis_url = redis_url
        self.mlflow_url = mlflow_url
        
    def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        health_status = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # System resources
        health_status["components"]["system"] = self._check_system_resources()
        
        # Database health
        health_status["components"]["database"] = self._check_database_health()
        
        # Redis health
        health_status["components"]["redis"] = self._check_redis_health()
        
        # MLflow health
        health_status["components"]["mlflow"] = self._check_mlflow_health()
        
        # API health
        health_status["components"]["api"] = self._check_api_health()
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in health_status["components"].values()]
        if "critical" in component_statuses:
            health_status["overall_status"] = "critical"
        elif "warning" in component_statuses:
            health_status["overall_status"] = "warning"
        
        return health_status
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            if cpu_percent > 80 or memory.percent > 80 or disk.percent > 80:
                status = "warning"
            if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
                status = "critical"
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "available_memory_gb": memory.available / (1024**3)
            }
        except Exception as e:
            return {"status": "critical", "error": str(e)}
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            engine = create_engine(self.database_url)
            start_time = time.time()
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            response_time = time.time() - start_time
            
            status = "healthy"
            if response_time > 1.0:
                status = "warning"
            if response_time > 5.0:
                status = "critical"
            
            return {
                "status": status,
                "response_time_ms": response_time * 1000,
                "connection": "active"
            }
        except Exception as e:
            return {"status": "critical", "error": str(e)}
    
    def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        try:
            r = redis.from_url(self.redis_url)
            start_time = time.time()
            
            r.ping()
            
            response_time = time.time() - start_time
            
            status = "healthy"
            if response_time > 0.1:
                status = "warning"
            if response_time > 1.0:
                status = "critical"
            
            return {
                "status": status,
                "response_time_ms": response_time * 1000,
                "connection": "active"
            }
        except Exception as e:
            return {"status": "critical", "error": str(e)}
    
    def _check_mlflow_health(self) -> Dict[str, Any]:
        """Check MLflow server health"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.mlflow_url}/health", timeout=5)
            response_time = time.time() - start_time
            
            status = "healthy" if response.status_code == 200 else "critical"
            
            return {
                "status": status,
                "response_time_ms": response_time * 1000,
                "status_code": response.status_code
            }
        except Exception as e:
            return {"status": "critical", "error": str(e)}
    
    def _check_api_health(self) -> Dict[str, Any]:
        """Check API endpoint health"""
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8081/health", timeout=5)
            response_time = time.time() - start_time
            
            status = "healthy" if response.status_code == 200 else "critical"
            
            return {
                "status": status,
                "response_time_ms": response_time * 1000,
                "status_code": response.status_code
            }
        except Exception as e:
            return {"status": "critical", "error": str(e)}
```

#### **Performance Metrics Collection**
```python
# src/monitoring/metrics.py
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from functools import wraps
import mlflow

# Prometheus metrics
REGISTRY = CollectorRegistry()

# API Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'], registry=REGISTRY)
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'], registry=REGISTRY)
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made', ['model_name'], registry=REGISTRY)

# System Metrics
CPU_USAGE = Gauge('system_cpu_percent', 'CPU usage percentage', registry=REGISTRY)
MEMORY_USAGE = Gauge('system_memory_percent', 'Memory usage percentage', registry=REGISTRY)
DISK_USAGE = Gauge('system_disk_percent', 'Disk usage percentage', registry=REGISTRY)

# Model Metrics
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy', ['model_name', 'version'], registry=REGISTRY)
MODEL_LATENCY = Histogram('model_prediction_duration_seconds', 'Model prediction latency', ['model_name'], registry=REGISTRY)

def track_api_metrics(endpoint: str):
    """Decorator to track API metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_COUNT.labels(method="GET", endpoint=endpoint, status=status).inc()
                REQUEST_DURATION.labels(method="GET", endpoint=endpoint).observe(duration)
        
        return wrapper
    return decorator

def update_system_metrics():
    """Update system metrics"""
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)

def get_metrics():
    """Get Prometheus metrics"""
    update_system_metrics()
    return generate_latest(REGISTRY)
```

### **2. Logging Strategy**

#### **Structured Logging Setup**
```python
# src/logging/config.py
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'model_name'):
            log_entry['model_name'] = record.model_name
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup structured logging configuration"""
    
    # Create formatter
    formatter = JSONFormatter()
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        format='%(message)s'
    )
    
    # Configure specific loggers
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

# Usage in main.py
setup_logging(log_level="INFO", log_file="logs/ghosteam-v5.log")
logger = logging.getLogger(__name__)
```

### **3. Alerting Mechanisms**

#### **Alert Manager**
```python
# src/monitoring/alerts.py
import smtplib
import requests
import json
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def send_alert(self, alert_type: str, message: str, severity: str = "warning", 
                   metadata: Dict[str, Any] = None):
        """Send alert through configured channels"""
        
        alert_data = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Send email alert
        if self.config.get("email", {}).get("enabled"):
            self._send_email_alert(alert_data)
        
        # Send Slack alert
        if self.config.get("slack", {}).get("enabled"):
            self._send_slack_alert(alert_data)
        
        # Send PagerDuty alert for critical issues
        if severity == "critical" and self.config.get("pagerduty", {}).get("enabled"):
            self._send_pagerduty_alert(alert_data)
        
        logger.info(f"Alert sent: {alert_type} - {message}", extra={"alert_data": alert_data})
    
    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send email alert"""
        try:
            email_config = self.config["email"]
            
            msg = MimeMultipart()
            msg['From'] = email_config["from"]
            msg['To'] = ", ".join(email_config["to"])
            msg['Subject'] = f"[{alert_data['severity'].upper()}] Ghosteam V5 Alert: {alert_data['type']}"
            
            body = f"""
            Alert Type: {alert_data['type']}
            Severity: {alert_data['severity']}
            Message: {alert_data['message']}
            Timestamp: {datetime.fromtimestamp(alert_data['timestamp'])}
            
            Metadata: {json.dumps(alert_data['metadata'], indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Send Slack alert"""
        try:
            slack_config = self.config["slack"]
            
            color = {
                "info": "#36a64f",
                "warning": "#ff9900",
                "critical": "#ff0000"
            }.get(alert_data["severity"], "#36a64f")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"Ghosteam V5 Alert: {alert_data['type']}",
                    "text": alert_data["message"],
                    "fields": [
                        {"title": "Severity", "value": alert_data["severity"], "short": True},
                        {"title": "Timestamp", "value": datetime.fromtimestamp(alert_data["timestamp"]), "short": True}
                    ]
                }]
            }
            
            response = requests.post(slack_config["webhook_url"], json=payload)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_pagerduty_alert(self, alert_data: Dict[str, Any]):
        """Send PagerDuty alert for critical issues"""
        try:
            pagerduty_config = self.config["pagerduty"]
            
            payload = {
                "routing_key": pagerduty_config["integration_key"],
                "event_action": "trigger",
                "payload": {
                    "summary": f"Ghosteam V5 Critical Alert: {alert_data['type']}",
                    "source": "ghosteam-v5",
                    "severity": "critical",
                    "custom_details": alert_data["metadata"]
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
```

---

## 🚨 **TROUBLESHOOTING GUIDE**

### **Common Issues & Solutions**

#### **1. High Memory Usage**
```bash
# Symptoms: Memory usage > 80%
# Diagnosis:
ps aux --sort=-%mem | head -10
free -h
cat /proc/meminfo

# Solutions:
# 1. Restart services
sudo systemctl restart ghosteam-v5-api
sudo systemctl restart mlflow-server

# 2. Clear cache
redis-cli FLUSHALL

# 3. Optimize model loading
# Use model caching and lazy loading
```

#### **2. Database Connection Issues**
```bash
# Symptoms: Database connection errors
# Diagnosis:
pg_isready -h localhost -p 5432
psql -h localhost -U username -d dbname -c "SELECT 1;"

# Solutions:
# 1. Check connection pool
# 2. Restart database service
# 3. Check network connectivity
# 4. Verify credentials
```

#### **3. Model Prediction Errors**
```python
# Symptoms: Prediction endpoint returning errors
# Diagnosis script:
def diagnose_model_issues():
    try:
        # Check model loading
        model = mlflow.sklearn.load_model("models:/model_name/latest")
        print("✅ Model loaded successfully")
        
        # Check prediction with sample data
        sample_data = np.random.randn(1, 10)
        prediction = model.predict(sample_data)
        print("✅ Model prediction successful")
        
    except Exception as e:
        print(f"❌ Model issue: {e}")
        
        # Common fixes:
        # 1. Check model registry
        # 2. Verify feature schema
        # 3. Check model dependencies
        # 4. Retrain model if necessary
```

### **Performance Optimization**

#### **1. API Response Time Optimization**
```python
# src/optimization/performance.py
import asyncio
import aioredis
from functools import lru_cache
import pickle

class PerformanceOptimizer:
    def __init__(self):
        self.redis_pool = None
        self.model_cache = {}
    
    async def init_redis_pool(self):
        """Initialize Redis connection pool"""
        self.redis_pool = aioredis.ConnectionPool.from_url(
            "redis://localhost:6379", 
            max_connections=20
        )
    
    @lru_cache(maxsize=100)
    def get_cached_prediction(self, input_hash: str):
        """Cache predictions for identical inputs"""
        # Implementation for prediction caching
        pass
    
    async def preload_models(self):
        """Preload frequently used models"""
        # Load models into memory at startup
        pass
    
    def optimize_batch_predictions(self, inputs: List):
        """Optimize batch prediction processing"""
        # Batch processing for multiple predictions
        pass
```

#### **2. Database Query Optimization**
```sql
-- Common optimization queries
-- 1. Add indexes for frequently queried columns
CREATE INDEX idx_model_name ON predictions(model_name);
CREATE INDEX idx_timestamp ON predictions(created_at);

-- 2. Analyze query performance
EXPLAIN ANALYZE SELECT * FROM predictions WHERE model_name = 'model_v1';

-- 3. Optimize connection pooling
-- Set appropriate pool sizes in database configuration
```

---

## 📋 **MAINTENANCE PROCEDURES**

### **Daily Maintenance**
```bash
#!/bin/bash
# daily_maintenance.sh

echo "🔍 Daily Maintenance - $(date)"

# 1. Check system health
curl -s http://localhost:8081/health | jq .

# 2. Check disk space
df -h

# 3. Check log file sizes
du -sh logs/*

# 4. Backup database
pg_dump ghosteam_v5 > backups/daily_backup_$(date +%Y%m%d).sql

# 5. Clean old logs (keep 7 days)
find logs/ -name "*.log" -mtime +7 -delete

# 6. Check model performance
python scripts/check_model_performance.py

echo "✅ Daily maintenance completed"
```

### **Weekly Maintenance**
```bash
#!/bin/bash
# weekly_maintenance.sh

echo "🔍 Weekly Maintenance - $(date)"

# 1. Update system packages
sudo apt update && sudo apt upgrade -y

# 2. Restart services
sudo systemctl restart ghosteam-v5-api
sudo systemctl restart mlflow-server

# 3. Database maintenance
psql -d ghosteam_v5 -c "VACUUM ANALYZE;"

# 4. Check model drift
python scripts/check_model_drift.py

# 5. Performance report
python scripts/generate_performance_report.py

echo "✅ Weekly maintenance completed"
```

### **Monthly Maintenance**
```bash
#!/bin/bash
# monthly_maintenance.sh

echo "🔍 Monthly Maintenance - $(date)"

# 1. Full system backup
tar -czf backups/monthly_backup_$(date +%Y%m).tar.gz \
    data/ models/ logs/ configs/

# 2. Security updates
sudo apt update && sudo apt upgrade -y

# 3. Certificate renewal (if using Let's Encrypt)
sudo certbot renew

# 4. Performance optimization review
python scripts/performance_optimization_review.py

# 5. Model retraining evaluation
python scripts/evaluate_retraining_needs.py

echo "✅ Monthly maintenance completed"
```

---

## 🎯 **OPERATIONAL KPIS**

### **System Performance**
- **Uptime**: >99.9%
- **Response Time**: <200ms (95th percentile)
- **Error Rate**: <0.1%
- **Throughput**: >1000 requests/minute

### **Model Performance**
- **Prediction Accuracy**: >90%
- **Model Latency**: <100ms
- **Data Drift Detection**: <24 hours
- **Model Retraining**: Weekly evaluation

### **Operational Efficiency**
- **Mean Time to Detection (MTTD)**: <5 minutes
- **Mean Time to Resolution (MTTR)**: <15 minutes
- **Deployment Frequency**: Daily
- **Change Failure Rate**: <5%

**🚀 Your operational management system is now production-ready!**
