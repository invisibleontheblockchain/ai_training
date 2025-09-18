# 🔴 PHASE 1: IMMEDIATE PRODUCTION DEPLOYMENT

## 🎯 **OBJECTIVE**
Deploy your fully operational Ghosteam V5 MLOps system to production cloud environment within 1-2 weeks.

## ⏰ **TIMELINE: 1-2 WEEKS**

---

## 📋 **WEEK 1: CLOUD DEPLOYMENT**

### **Day 1-2: Platform Selection & Initial Deployment**

#### **Option A: Railway Deployment (FASTEST - Recommended for MVP)**

**Why Railway?**
- ✅ Already configured and tested
- ✅ Managed databases included
- ✅ Zero-config deployments
- ✅ Built-in HTTPS
- ✅ Cost-effective for startups

**Deployment Steps:**
```bash
# 1. Ensure you're in the ghosteam-v5 directory
cd ghosteam-v5

# 2. Verify Railway connection
railway whoami
railway status

# 3. Deploy to production
railway up --service ghosteam-v5

# 4. Add managed databases
railway add --database postgres
railway add --database redis

# 5. Configure production environment variables
railway variables set ENVIRONMENT=production
railway variables set DEBUG=false
railway variables set SECRET_KEY=$(openssl rand -hex 32)

# 6. Get your production URL
railway domain
```

**Expected Result**: Live system at `https://your-app.railway.app`

#### **Option B: AWS ECS Deployment (SCALABLE - Recommended for Growth)**

**Prerequisites:**
```bash
# Install AWS CLI and configure
aws configure
# Install ECS CLI
curl -Lo ecs-cli https://amazon-ecs-cli.s3.amazonaws.com/ecs-cli-linux-amd64-latest
```

**Deployment Steps:**
```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name ghosteam-v5

# 2. Build and push Docker image
$(aws ecr get-login --no-include-email)
docker build -f Dockerfile.railway -t ghosteam-v5 .
docker tag ghosteam-v5:latest $ECR_URI:latest
docker push $ECR_URI:latest

# 3. Create ECS cluster
ecs-cli up --cluster-config ghosteam-v5 --ecs-profile ghosteam-v5

# 4. Deploy service
ecs-cli compose --file docker-compose.aws.yml service up
```

### **Day 3-4: Database Migration & Configuration**

#### **Database Setup**

**Railway (Automatic):**
```bash
# Databases are automatically provisioned
railway variables  # Check DATABASE_URL and REDIS_URL
```

**AWS (Manual Setup):**
```bash
# Create RDS PostgreSQL instance
aws rds create-db-instance \
  --db-instance-identifier ghosteam-v5-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username admin \
  --master-user-password $(openssl rand -base64 32)

# Create ElastiCache Redis cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id ghosteam-v5-redis \
  --cache-node-type cache.t3.micro \
  --engine redis
```

#### **Environment Configuration**

Create production environment file:
```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:pass@host:5432/dbname
REDIS_URL=redis://host:6379
MLFLOW_TRACKING_URI=http://localhost:5000
LOG_LEVEL=INFO
```

### **Day 5-7: Security Implementation**

#### **1. HTTPS/TLS Setup**

**Railway (Automatic):**
- ✅ HTTPS automatically enabled
- ✅ SSL certificates managed

**AWS (Manual):**
```bash
# Create Application Load Balancer with SSL
aws elbv2 create-load-balancer \
  --name ghosteam-v5-alb \
  --subnets subnet-12345 subnet-67890 \
  --security-groups sg-12345

# Add SSL certificate (use ACM)
aws acm request-certificate \
  --domain-name yourdomain.com \
  --validation-method DNS
```

#### **2. API Authentication**

Add API key authentication:
```python
# src/auth.py
from fastapi import HTTPException, Depends, Header
import os

API_KEY = os.getenv("API_KEY", "your-secure-api-key")

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Update main.py
from .auth import verify_api_key

@app.get("/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    # ... existing code
```

#### **3. Secrets Management**

**Railway:**
```bash
# Set secrets as environment variables
railway variables set API_KEY=$(openssl rand -hex 32)
railway variables set SECRET_KEY=$(openssl rand -hex 32)
```

**AWS:**
```bash
# Use AWS Secrets Manager
aws secretsmanager create-secret \
  --name ghosteam-v5/api-key \
  --secret-string '{"api_key":"'$(openssl rand -hex 32)'"}'
```

---

## 📋 **WEEK 2: MONITORING & VALIDATION**

### **Day 8-10: Basic Monitoring Setup**

#### **1. Health Check Enhancement**

Update health check endpoint:
```python
# src/minimal_app.py
import psutil
import time

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "mlflow_version": mlflow.__version__,
        "services": {
            "api": "running",
            "mlflow": "available",
            "database": await check_database(),
            "redis": await check_redis()
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }
```

#### **2. Uptime Monitoring**

**UptimeRobot Setup:**
```bash
# Sign up at uptimerobot.com
# Add HTTP(s) monitor for your production URL
# Configure alerts via email/Slack
```

**Pingdom Alternative:**
```bash
# Sign up at pingdom.com
# Create uptime check
# Set up alert contacts
```

#### **3. Error Tracking**

**Sentry Integration:**
```python
# Install Sentry
pip install sentry-sdk[fastapi]

# Add to main.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
)
```

### **Day 11-12: Performance Optimization**

#### **1. Database Connection Pooling**

```python
# src/database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

#### **2. Redis Caching**

```python
# src/cache.py
import redis
import json
from functools import wraps

redis_client = redis.from_url(REDIS_URL)

def cache_result(expiration=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator

# Use in endpoints
@app.get("/models")
@cache_result(expiration=60)  # Cache for 1 minute
async def list_models():
    # ... existing code
```

### **Day 13-14: Production Testing & Validation**

#### **1. Load Testing**

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test API endpoints
ab -n 1000 -c 10 https://your-app.railway.app/health
ab -n 500 -c 5 https://your-app.railway.app/models

# Install and use wrk for advanced testing
wrk -t12 -c400 -d30s https://your-app.railway.app/health
```

#### **2. End-to-End Testing**

```python
# tests/test_production.py
import requests
import pytest

PRODUCTION_URL = "https://your-app.railway.app"

def test_production_health():
    response = requests.get(f"{PRODUCTION_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_production_models():
    headers = {"X-API-Key": "your-api-key"}
    response = requests.get(f"{PRODUCTION_URL}/models", headers=headers)
    assert response.status_code == 200

def test_production_prediction():
    headers = {"X-API-Key": "your-api-key"}
    data = {"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    response = requests.post(f"{PRODUCTION_URL}/predict", json=data, headers=headers)
    assert response.status_code == 200
```

#### **3. Production Verification Script**

```python
# production_verification.py
import requests
import time
import sys

def verify_production_deployment(base_url, api_key):
    """Comprehensive production verification"""
    
    headers = {"X-API-Key": api_key} if api_key else {}
    
    tests = [
        ("Health Check", "GET", "/health", None),
        ("Models List", "GET", "/models", headers),
        ("Prediction", "POST", "/predict", headers, {"data": [1,2,3,4,5,6,7,8,9,10]})
    ]
    
    print(f"🔍 Verifying production deployment at {base_url}")
    
    for test_name, method, endpoint, test_headers, payload in tests:
        try:
            url = f"{base_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, headers=test_headers or {})
            else:
                response = requests.post(url, json=payload, headers=test_headers or {})
            
            if response.status_code == 200:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED ({response.status_code})")
                return False
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            return False
    
    print("🎉 Production deployment verified successfully!")
    return True

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://your-app.railway.app"
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = verify_production_deployment(base_url, api_key)
    sys.exit(0 if success else 1)
```

---

## ✅ **PHASE 1 COMPLETION CHECKLIST**

### **Deployment**
- [ ] Cloud platform selected and configured
- [ ] Application deployed and accessible via HTTPS
- [ ] Managed databases (PostgreSQL, Redis) configured
- [ ] Environment variables set for production

### **Security**
- [ ] HTTPS/TLS certificates configured
- [ ] API key authentication implemented
- [ ] Secrets properly managed (not in code)
- [ ] Basic firewall rules configured

### **Monitoring**
- [ ] Enhanced health check endpoint
- [ ] Uptime monitoring configured
- [ ] Error tracking (Sentry) implemented
- [ ] Basic performance metrics collected

### **Testing**
- [ ] Load testing completed
- [ ] End-to-end production tests passing
- [ ] Performance benchmarks established
- [ ] Rollback procedures documented

---

## 🎯 **SUCCESS CRITERIA**

At the end of Phase 1, you should have:

1. **✅ Live Production System**: Accessible via HTTPS with custom domain
2. **✅ 99%+ Uptime**: Monitored and alerting on downtime
3. **✅ Secure Access**: API key authentication and HTTPS
4. **✅ Performance Baseline**: <500ms response times under normal load
5. **✅ Monitoring**: Basic health checks and error tracking

**🚀 Ready for Phase 2: Production Hardening!**
