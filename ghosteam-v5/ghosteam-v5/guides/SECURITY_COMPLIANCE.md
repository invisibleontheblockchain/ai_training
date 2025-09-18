# 🔒 SECURITY & COMPLIANCE GUIDE

## 🎯 **OBJECTIVE**
Comprehensive security measures, access controls, and compliance considerations for production MLOps environment.

---

## 🛡️ **SECURITY ARCHITECTURE**

### **1. Authentication & Authorization**

#### **JWT-Based Authentication**
```python
# src/auth/jwt_auth.py
import jwt
import bcrypt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

class AuthManager:
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials"
                )
            return payload
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

auth_manager = AuthManager()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    return payload
```

#### **Role-Based Access Control (RBAC)**
```python
# src/auth/rbac.py
from enum import Enum
from typing import List, Dict, Any
from fastapi import HTTPException, status

class Role(Enum):
    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    VIEWER = "viewer"

class Permission(Enum):
    READ_MODELS = "read_models"
    WRITE_MODELS = "write_models"
    DELETE_MODELS = "delete_models"
    TRAIN_MODELS = "train_models"
    DEPLOY_MODELS = "deploy_models"
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    ADMIN_SYSTEM = "admin_system"

# Role permissions mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [
        Permission.READ_MODELS, Permission.WRITE_MODELS, Permission.DELETE_MODELS,
        Permission.TRAIN_MODELS, Permission.DEPLOY_MODELS, Permission.READ_DATA,
        Permission.WRITE_DATA, Permission.ADMIN_SYSTEM
    ],
    Role.DATA_SCIENTIST: [
        Permission.READ_MODELS, Permission.WRITE_MODELS, Permission.TRAIN_MODELS,
        Permission.READ_DATA, Permission.WRITE_DATA
    ],
    Role.ML_ENGINEER: [
        Permission.READ_MODELS, Permission.WRITE_MODELS, Permission.DEPLOY_MODELS,
        Permission.READ_DATA
    ],
    Role.VIEWER: [
        Permission.READ_MODELS, Permission.READ_DATA
    ]
}

def check_permission(user_role: str, required_permission: Permission) -> bool:
    """Check if user role has required permission"""
    try:
        role = Role(user_role)
        return required_permission in ROLE_PERMISSIONS.get(role, [])
    except ValueError:
        return False

def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get user from request context
            user = kwargs.get('current_user')
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user_role = user.get('role')
            if not check_permission(user_role, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

### **2. Data Encryption**

#### **Encryption at Rest**
```python
# src/security/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    def __init__(self, password: str = None):
        if password is None:
            password = os.getenv("ENCRYPTION_PASSWORD", "default-password")
        
        # Generate key from password
        password_bytes = password.encode()
        salt = os.getenv("ENCRYPTION_SALT", "default-salt").encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        self.cipher_suite = Fernet(key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted_data.decode()
    
    def encrypt_file(self, file_path: str, output_path: str):
        """Encrypt file"""
        with open(file_path, 'rb') as file:
            file_data = file.read()
        
        encrypted_data = self.cipher_suite.encrypt(file_data)
        
        with open(output_path, 'wb') as file:
            file.write(encrypted_data)
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str):
        """Decrypt file"""
        with open(encrypted_file_path, 'rb') as file:
            encrypted_data = file.read()
        
        decrypted_data = self.cipher_suite.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as file:
            file.write(decrypted_data)

# Usage for sensitive model data
encryptor = DataEncryption()

# Encrypt model artifacts before storage
def secure_model_storage(model_path: str):
    encrypted_path = f"{model_path}.encrypted"
    encryptor.encrypt_file(model_path, encrypted_path)
    os.remove(model_path)  # Remove unencrypted version
    return encrypted_path
```

#### **Encryption in Transit**
```python
# src/security/tls.py
import ssl
import certifi
from fastapi import FastAPI
import uvicorn

def create_ssl_context():
    """Create SSL context for HTTPS"""
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(certifi.where())
    return context

def run_secure_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8443):
    """Run server with HTTPS"""
    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_keyfile="certs/private.key",
        ssl_certfile="certs/certificate.crt",
        ssl_ca_certs="certs/ca-bundle.crt"
    )
```

### **3. Input Validation & Sanitization**

#### **Request Validation**
```python
# src/security/validation.py
from pydantic import BaseModel, validator, Field
from typing import List, Optional, Any
import re
import bleach

class PredictionRequest(BaseModel):
    data: List[float] = Field(..., min_items=1, max_items=1000)
    model_name: Optional[str] = Field(None, max_length=100)
    metadata: Optional[dict] = Field(None)
    
    @validator('data')
    def validate_data(cls, v):
        """Validate input data"""
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError('All data points must be numeric')
        
        if any(abs(x) > 1e6 for x in v):
            raise ValueError('Data values too large')
        
        return v
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name"""
        if v is None:
            return v
        
        # Only allow alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Invalid model name format')
        
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate and sanitize metadata"""
        if v is None:
            return v
        
        # Limit metadata size
        if len(str(v)) > 1000:
            raise ValueError('Metadata too large')
        
        # Sanitize string values
        if isinstance(v, dict):
            sanitized = {}
            for key, value in v.items():
                if isinstance(value, str):
                    sanitized[key] = bleach.clean(value)
                else:
                    sanitized[key] = value
            return sanitized
        
        return v

class UserRegistration(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    password: str = Field(..., min_length=8)
    role: str = Field(..., regex=r'^(admin|data_scientist|ml_engineer|viewer)$')
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username"""
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        
        return v
```

### **4. Security Headers & Middleware**

#### **Security Middleware**
```python
# src/security/middleware.py
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time
import hashlib
import hmac

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        self.clients = {
            ip: timestamps for ip, timestamps in self.clients.items()
            if any(t > current_time - self.period for t in timestamps)
        }
        
        # Check rate limit
        if client_ip in self.clients:
            timestamps = [t for t in self.clients[client_ip] if t > current_time - self.period]
            if len(timestamps) >= self.calls:
                return Response(
                    content="Rate limit exceeded",
                    status_code=429,
                    headers={"Retry-After": str(self.period)}
                )
            timestamps.append(current_time)
            self.clients[client_ip] = timestamps
        else:
            self.clients[client_ip] = [current_time]
        
        response = await call_next(request)
        return response

def setup_security_middleware(app: FastAPI):
    """Setup all security middleware"""
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"],  # Restrict to your domain
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Trusted hosts
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
    )
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware, calls=100, period=60)
```

---

## 📋 **COMPLIANCE FRAMEWORKS**

### **1. GDPR Compliance**

#### **Data Privacy Implementation**
```python
# src/compliance/gdpr.py
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class GDPRCompliance:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def log_data_processing(self, user_id: str, data_type: str, 
                           purpose: str, legal_basis: str):
        """Log data processing activities"""
        processing_record = {
            "user_id": user_id,
            "data_type": data_type,
            "purpose": purpose,
            "legal_basis": legal_basis,
            "timestamp": datetime.utcnow(),
            "retention_period": self._get_retention_period(data_type)
        }
        
        # Store in audit log
        self.db.execute(
            "INSERT INTO data_processing_log (user_id, data_type, purpose, legal_basis, timestamp, retention_period) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, data_type, purpose, legal_basis, processing_record["timestamp"], processing_record["retention_period"])
        )
        
        logger.info(f"Data processing logged for user {user_id}", extra=processing_record)
    
    def handle_data_subject_request(self, user_id: str, request_type: str) -> Dict[str, Any]:
        """Handle GDPR data subject requests"""
        
        if request_type == "access":
            return self._handle_access_request(user_id)
        elif request_type == "portability":
            return self._handle_portability_request(user_id)
        elif request_type == "erasure":
            return self._handle_erasure_request(user_id)
        elif request_type == "rectification":
            return self._handle_rectification_request(user_id)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    def _handle_access_request(self, user_id: str) -> Dict[str, Any]:
        """Handle data access request"""
        user_data = self.db.execute(
            "SELECT * FROM user_data WHERE user_id = ?", (user_id,)
        ).fetchall()
        
        processing_log = self.db.execute(
            "SELECT * FROM data_processing_log WHERE user_id = ?", (user_id,)
        ).fetchall()
        
        return {
            "user_data": user_data,
            "processing_activities": processing_log,
            "generated_at": datetime.utcnow()
        }
    
    def _handle_erasure_request(self, user_id: str) -> Dict[str, Any]:
        """Handle right to be forgotten request"""
        
        # Check if erasure is legally required
        if not self._can_erase_data(user_id):
            return {
                "status": "denied",
                "reason": "Legal obligation to retain data"
            }
        
        # Anonymize or delete user data
        self.db.execute("DELETE FROM user_data WHERE user_id = ?", (user_id,))
        self.db.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
        
        # Log erasure
        self.log_data_processing(user_id, "all", "erasure", "data_subject_request")
        
        return {
            "status": "completed",
            "erased_at": datetime.utcnow()
        }
    
    def _get_retention_period(self, data_type: str) -> int:
        """Get data retention period in days"""
        retention_periods = {
            "user_profile": 2555,  # 7 years
            "predictions": 1095,   # 3 years
            "training_data": 1825, # 5 years
            "logs": 365           # 1 year
        }
        return retention_periods.get(data_type, 365)
    
    def cleanup_expired_data(self):
        """Clean up data past retention period"""
        current_time = datetime.utcnow()
        
        # Get all data types and their retention periods
        data_types = self.db.execute(
            "SELECT DISTINCT data_type, retention_period FROM data_processing_log"
        ).fetchall()
        
        for data_type, retention_days in data_types:
            cutoff_date = current_time - timedelta(days=retention_days)
            
            # Delete expired data
            deleted_count = self.db.execute(
                f"DELETE FROM {data_type} WHERE created_at < ?", (cutoff_date,)
            ).rowcount
            
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} expired {data_type} records")
```

### **2. SOC 2 Compliance**

#### **Audit Logging**
```python
# src/compliance/audit.py
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

class AuditLogger:
    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = logging.getLogger("audit")
    
    def log_event(self, event_type: str, user_id: str, resource: str, 
                  action: str, outcome: str, details: Dict[str, Any] = None):
        """Log audit event"""
        
        audit_record = {
            "event_id": self._generate_event_id(),
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "outcome": outcome,
            "details": details or {},
            "ip_address": self._get_client_ip(),
            "user_agent": self._get_user_agent()
        }
        
        # Store in database
        self.db.execute(
            """INSERT INTO audit_log 
               (event_id, timestamp, event_type, user_id, resource, action, outcome, details, ip_address, user_agent)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                audit_record["event_id"],
                audit_record["timestamp"],
                audit_record["event_type"],
                audit_record["user_id"],
                audit_record["resource"],
                audit_record["action"],
                audit_record["outcome"],
                json.dumps(audit_record["details"]),
                audit_record["ip_address"],
                audit_record["user_agent"]
            )
        )
        
        # Log to file
        self.logger.info(json.dumps(audit_record))
    
    def log_authentication(self, user_id: str, outcome: str, details: Dict[str, Any] = None):
        """Log authentication events"""
        self.log_event("authentication", user_id, "auth_system", "login", outcome, details)
    
    def log_authorization(self, user_id: str, resource: str, action: str, outcome: str):
        """Log authorization events"""
        self.log_event("authorization", user_id, resource, action, outcome)
    
    def log_data_access(self, user_id: str, resource: str, action: str, outcome: str, details: Dict[str, Any] = None):
        """Log data access events"""
        self.log_event("data_access", user_id, resource, action, outcome, details)
    
    def log_system_change(self, user_id: str, resource: str, action: str, outcome: str, details: Dict[str, Any] = None):
        """Log system changes"""
        self.log_event("system_change", user_id, resource, action, outcome, details)
    
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate audit report for compliance"""
        
        events = self.db.execute(
            """SELECT * FROM audit_log 
               WHERE timestamp BETWEEN ? AND ?
               ORDER BY timestamp DESC""",
            (start_date.isoformat(), end_date.isoformat())
        ).fetchall()
        
        # Analyze events
        event_summary = {}
        for event in events:
            event_type = event["event_type"]
            if event_type not in event_summary:
                event_summary[event_type] = {"total": 0, "success": 0, "failure": 0}
            
            event_summary[event_type]["total"] += 1
            if event["outcome"] == "success":
                event_summary[event_type]["success"] += 1
            else:
                event_summary[event_type]["failure"] += 1
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(events),
            "event_summary": event_summary,
            "events": [dict(event) for event in events]
        }
```

---

## 🔍 **SECURITY MONITORING**

### **Security Incident Detection**
```python
# src/security/monitoring.py
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger("security")

class SecurityMonitor:
    def __init__(self, alert_manager):
        self.alert_manager = alert_manager
        self.suspicious_patterns = [
            r'(?i)(union|select|insert|delete|drop|create|alter)',  # SQL injection
            r'<script[^>]*>.*?</script>',  # XSS
            r'\.\./',  # Path traversal
            r'eval\(',  # Code injection
        ]
    
    def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze request for security threats"""
        
        threats_detected = []
        risk_score = 0
        
        # Check for suspicious patterns
        request_content = str(request_data)
        for pattern in self.suspicious_patterns:
            if re.search(pattern, request_content):
                threats_detected.append(f"Suspicious pattern detected: {pattern}")
                risk_score += 10
        
        # Check request frequency (potential DDoS)
        if self._check_request_frequency(request_data.get("client_ip")):
            threats_detected.append("High request frequency detected")
            risk_score += 20
        
        # Check for unusual data patterns
        if self._check_unusual_data_patterns(request_data):
            threats_detected.append("Unusual data patterns detected")
            risk_score += 15
        
        analysis_result = {
            "timestamp": datetime.utcnow(),
            "threats_detected": threats_detected,
            "risk_score": risk_score,
            "action_required": risk_score > 30
        }
        
        # Send alert if high risk
        if risk_score > 30:
            self.alert_manager.send_alert(
                "security_threat",
                f"High-risk security threat detected (score: {risk_score})",
                "critical",
                {"threats": threats_detected, "request_data": request_data}
            )
        
        return analysis_result
    
    def _check_request_frequency(self, client_ip: str) -> bool:
        """Check if request frequency is suspicious"""
        # Implementation would check request history
        # Return True if frequency is above threshold
        return False
    
    def _check_unusual_data_patterns(self, request_data: Dict[str, Any]) -> bool:
        """Check for unusual data patterns"""
        # Implementation would analyze data patterns
        # Return True if patterns are unusual
        return False
```

---

## 📋 **SECURITY CHECKLIST**

### **Pre-Production Security Audit**
- [ ] **Authentication**: JWT tokens, password hashing, session management
- [ ] **Authorization**: RBAC implementation, permission checks
- [ ] **Input Validation**: Request validation, sanitization, size limits
- [ ] **Encryption**: Data at rest, data in transit, key management
- [ ] **Security Headers**: HTTPS, CSP, HSTS, XSS protection
- [ ] **Rate Limiting**: API rate limits, DDoS protection
- [ ] **Audit Logging**: Comprehensive event logging, log integrity
- [ ] **Vulnerability Scanning**: Dependency scanning, code analysis
- [ ] **Secrets Management**: Environment variables, key rotation
- [ ] **Network Security**: Firewall rules, VPN access, network segmentation

### **Ongoing Security Maintenance**
- [ ] **Security Updates**: Regular dependency updates, OS patches
- [ ] **Access Reviews**: Quarterly user access reviews, role audits
- [ ] **Penetration Testing**: Annual security assessments
- [ ] **Incident Response**: Response procedures, contact lists
- [ ] **Backup Security**: Encrypted backups, secure storage
- [ ] **Monitoring**: Security event monitoring, anomaly detection
- [ ] **Training**: Security awareness training for team
- [ ] **Compliance**: Regular compliance assessments, documentation updates

---

## 🎯 **COMPLIANCE METRICS**

### **Security KPIs**
- **Security Incidents**: 0 critical incidents per month
- **Vulnerability Response**: <24 hours for critical vulnerabilities
- **Access Review**: 100% quarterly access reviews completed
- **Audit Compliance**: 100% audit trail coverage
- **Data Encryption**: 100% sensitive data encrypted

### **Privacy KPIs**
- **Data Subject Requests**: <30 days response time
- **Data Retention**: 100% compliance with retention policies
- **Consent Management**: 100% documented consent
- **Data Breach Response**: <72 hours notification time
- **Privacy Training**: 100% team completion rate

**🔒 Your security and compliance framework is now enterprise-ready!**
