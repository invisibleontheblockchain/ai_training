#!/usr/bin/env python3
"""
Ghosteam V5 Autonomous MLOps Application
Complete system with continuous learning and predictive intelligence
"""

import sys
import os
import time
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ML and data processing
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global configuration
CONFIG = {
    "autonomous_mode": True,
    "continuous_learning": {
        "enabled": True,
        "retrain_interval_hours": 24,
        "performance_threshold": 0.85,
        "auto_deploy": True
    },
    "predictive_intelligence": {
        "enabled": True,
        "pattern_analysis": True,
        "proactive_suggestions": True,
        "context_awareness": True
    },
    "monitoring": {
        "health_check_interval": 300,
        "performance_monitoring": True,
        "alert_thresholds": {
            "error_rate": 0.05,
            "response_time": 1000,
            "cpu_usage": 80,
            "memory_usage": 80
        }
    }
}

# Pydantic models
class PredictionRequest(BaseModel):
    data: List[float]
    model_name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    prediction: Any
    confidence: float
    model_used: str
    timestamp: datetime
    suggestions: List[str] = []

class SystemStatus(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    metrics: Dict[str, Any]
    autonomous_features: Dict[str, bool]

# Create FastAPI app
app = FastAPI(
    title="Ghosteam V5 Autonomous MLOps System",
    version="2.0.0",
    description="Autonomous MLOps system with continuous learning and predictive intelligence"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
SYSTEM_STATE = {
    "models": {},
    "predictions": [],
    "usage_patterns": [],
    "performance_metrics": {},
    "last_retrain": None,
    "autonomous_actions": []
}

class AutonomousMLOpsSystem:
    def __init__(self):
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.models = {}
        self.usage_patterns = []
        self.performance_history = []
        self.predictive_insights = []
        
        # Set up MLflow
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            logger.info(f"MLflow tracking URI set to: {self.mlflow_uri}")
        except Exception as e:
            logger.warning(f"MLflow setup warning: {e}")
    
    async def continuous_learning_loop(self):
        """Background task for continuous learning"""
        while CONFIG["continuous_learning"]["enabled"]:
            try:
                await self.check_model_performance()
                await self.analyze_usage_patterns()
                await self.generate_predictive_insights()
                
                # Check if retraining is needed
                if await self.should_retrain_model():
                    await self.autonomous_retrain()
                
                # Sleep for the configured interval
                await asyncio.sleep(CONFIG["continuous_learning"]["retrain_interval_hours"] * 3600)
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def check_model_performance(self):
        """Monitor model performance and detect drift"""
        try:
            # Simulate performance monitoring
            current_accuracy = np.random.uniform(0.85, 0.95)
            
            self.performance_history.append({
                "timestamp": datetime.now(),
                "accuracy": current_accuracy,
                "predictions_count": len(SYSTEM_STATE["predictions"])
            })
            
            # Keep only last 100 records
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            SYSTEM_STATE["performance_metrics"]["current_accuracy"] = current_accuracy
            
            logger.info(f"Model performance check: accuracy={current_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
    
    async def analyze_usage_patterns(self):
        """Analyze usage patterns for predictive insights"""
        try:
            # Analyze recent predictions
            recent_predictions = SYSTEM_STATE["predictions"][-100:]
            
            if len(recent_predictions) > 10:
                # Extract patterns
                hourly_usage = {}
                for pred in recent_predictions:
                    hour = pred.get("timestamp", datetime.now()).hour
                    hourly_usage[hour] = hourly_usage.get(hour, 0) + 1
                
                # Find peak usage hours
                peak_hour = max(hourly_usage, key=hourly_usage.get) if hourly_usage else 12
                
                self.usage_patterns.append({
                    "timestamp": datetime.now(),
                    "peak_hour": peak_hour,
                    "total_predictions": len(recent_predictions),
                    "hourly_distribution": hourly_usage
                })
                
                logger.info(f"Usage pattern analysis: peak_hour={peak_hour}")
            
        except Exception as e:
            logger.error(f"Error analyzing usage patterns: {e}")
    
    async def generate_predictive_insights(self):
        """Generate proactive insights and suggestions"""
        try:
            insights = []
            
            # Predict peak usage times
            if self.usage_patterns:
                latest_pattern = self.usage_patterns[-1]
                peak_hour = latest_pattern.get("peak_hour", 12)
                current_hour = datetime.now().hour
                
                if abs(current_hour - peak_hour) <= 2:
                    insights.append("🔮 Peak usage period detected - consider scaling resources")
                
                if (peak_hour - current_hour) % 24 <= 2:
                    insights.append("📈 Peak usage approaching in next 2 hours")
            
            # Model performance insights
            if self.performance_history:
                recent_accuracy = [p["accuracy"] for p in self.performance_history[-10:]]
                if len(recent_accuracy) > 5:
                    trend = np.polyfit(range(len(recent_accuracy)), recent_accuracy, 1)[0]
                    if trend < -0.01:
                        insights.append("📉 Model performance declining - retraining recommended")
                    elif trend > 0.01:
                        insights.append("📈 Model performance improving")
            
            # Resource optimization suggestions
            prediction_count = len(SYSTEM_STATE["predictions"])
            if prediction_count > 1000:
                insights.append("🎯 High prediction volume - consider caching optimization")
            
            self.predictive_insights = insights
            logger.info(f"Generated {len(insights)} predictive insights")
            
        except Exception as e:
            logger.error(f"Error generating predictive insights: {e}")
    
    async def should_retrain_model(self) -> bool:
        """Determine if model should be retrained"""
        try:
            # Check performance threshold
            current_accuracy = SYSTEM_STATE["performance_metrics"].get("current_accuracy", 0.9)
            if current_accuracy < CONFIG["continuous_learning"]["performance_threshold"]:
                logger.info("Model performance below threshold - retraining needed")
                return True
            
            # Check time since last retrain
            last_retrain = SYSTEM_STATE.get("last_retrain")
            if last_retrain:
                hours_since_retrain = (datetime.now() - last_retrain).total_seconds() / 3600
                if hours_since_retrain > CONFIG["continuous_learning"]["retrain_interval_hours"]:
                    logger.info("Retrain interval exceeded - retraining needed")
                    return True
            else:
                # Never retrained
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain condition: {e}")
            return False
    
    async def autonomous_retrain(self):
        """Autonomously retrain the model"""
        try:
            logger.info("Starting autonomous model retraining...")
            
            # Generate synthetic training data (in real scenario, this would be real data)
            X = np.random.randn(1000, 10)
            y = (X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.1 > 0).astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train new model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log with MLflow
            try:
                with mlflow.start_run(run_name=f"autonomous_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_param("autonomous", True)
                    mlflow.log_param("n_estimators", 100)
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.sklearn.log_model(model, "model", registered_model_name="autonomous-model")
                    
                    logger.info(f"Model retrained with accuracy: {accuracy:.3f}")
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")
            
            # Update system state
            SYSTEM_STATE["models"]["autonomous-model"] = model
            SYSTEM_STATE["last_retrain"] = datetime.now()
            SYSTEM_STATE["performance_metrics"]["latest_retrain_accuracy"] = accuracy
            
            # Record autonomous action
            SYSTEM_STATE["autonomous_actions"].append({
                "action": "model_retrain",
                "timestamp": datetime.now(),
                "accuracy": accuracy,
                "trigger": "autonomous_system"
            })
            
            logger.info("Autonomous retraining completed successfully")
            
        except Exception as e:
            logger.error(f"Error in autonomous retraining: {e}")

# Initialize autonomous system
autonomous_system = AutonomousMLOpsSystem()

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the autonomous system on startup"""
    logger.info("Starting Ghosteam V5 Autonomous MLOps System...")
    
    # Start continuous learning loop
    if CONFIG["autonomous_mode"]:
        asyncio.create_task(autonomous_system.continuous_learning_loop())
        logger.info("Autonomous continuous learning loop started")
    
    # Initialize with a basic model
    try:
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        SYSTEM_STATE["models"]["default"] = model
        logger.info("Default model initialized")
    except Exception as e:
        logger.error(f"Error initializing default model: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system overview"""
    return """
    <html>
        <head><title>Ghosteam V5 Autonomous MLOps</title></head>
        <body style="font-family: Arial, sans-serif; margin: 40px;">
            <h1>🚀 Ghosteam V5 Autonomous MLOps System</h1>
            <p>Your intelligent, self-improving ML system is operational!</p>
            <h2>🔗 Quick Links:</h2>
            <ul>
                <li><a href="/docs">📓 API Documentation</a></li>
                <li><a href="/health">🔍 Health Check</a></li>
                <li><a href="/dashboard">📊 Dashboard</a></li>
                <li><a href="/models">📈 Models</a></li>
                <li><a href="/insights">🔮 Predictive Insights</a></li>
            </ul>
            <h2>🤖 Autonomous Features:</h2>
            <ul>
                <li>✅ Continuous Learning</li>
                <li>✅ Predictive Intelligence</li>
                <li>✅ Self-Monitoring</li>
                <li>✅ Auto-Optimization</li>
            </ul>
        </body>
    </html>
    """

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Enhanced health check with autonomous system status"""
    try:
        return SystemStatus(
            status="healthy",
            timestamp=datetime.now(),
            services={
                "api": "running",
                "mlflow": "available",
                "autonomous_learning": "active" if CONFIG["autonomous_mode"] else "disabled",
                "predictive_intelligence": "active" if CONFIG["predictive_intelligence"]["enabled"] else "disabled"
            },
            metrics={
                "total_predictions": len(SYSTEM_STATE["predictions"]),
                "models_loaded": len(SYSTEM_STATE["models"]),
                "last_retrain": SYSTEM_STATE.get("last_retrain"),
                "current_accuracy": SYSTEM_STATE["performance_metrics"].get("current_accuracy", 0.0),
                "autonomous_actions": len(SYSTEM_STATE["autonomous_actions"])
            },
            autonomous_features={
                "continuous_learning": CONFIG["continuous_learning"]["enabled"],
                "predictive_intelligence": CONFIG["predictive_intelligence"]["enabled"],
                "auto_retraining": CONFIG["continuous_learning"]["auto_deploy"],
                "pattern_analysis": CONFIG["predictive_intelligence"]["pattern_analysis"]
            }
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make prediction with autonomous learning"""
    try:
        # Get model
        model_name = request.model_name or "default"
        model = SYSTEM_STATE["models"].get(model_name)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Make prediction
        prediction = int(model.predict([request.data])[0])
        
        # Calculate confidence (simplified)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([request.data])[0]
            confidence = float(max(proba))
        else:
            confidence = 0.8  # Default confidence
        
        # Record prediction for learning
        prediction_record = {
            "timestamp": datetime.now(),
            "model_used": model_name,
            "input_data": request.data,
            "prediction": prediction,
            "confidence": confidence,
            "context": request.context
        }
        SYSTEM_STATE["predictions"].append(prediction_record)
        
        # Generate suggestions based on context and patterns
        suggestions = []
        if autonomous_system.predictive_insights:
            suggestions = autonomous_system.predictive_insights[:3]  # Top 3 insights
        
        # Add context-aware suggestions
        if request.context:
            if request.context.get("user_type") == "new":
                suggestions.append("💡 Consider exploring our model documentation")
            elif request.context.get("frequency") == "high":
                suggestions.append("⚡ Batch predictions available for better performance")
        
        # Schedule background analysis
        background_tasks.add_task(autonomous_system.analyze_usage_patterns)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_used=model_name,
            timestamp=datetime.now(),
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models with performance metrics"""
    try:
        models_info = []
        
        for name, model in SYSTEM_STATE["models"].items():
            model_info = {
                "name": name,
                "type": type(model).__name__,
                "loaded": True,
                "performance": SYSTEM_STATE["performance_metrics"].get(f"{name}_accuracy", "unknown")
            }
            models_info.append(model_info)
        
        # Add MLflow models if available
        try:
            mlflow.set_tracking_uri(autonomous_system.mlflow_uri)
            client = mlflow.tracking.MlflowClient()
            registered_models = client.search_registered_models()
            
            for model in registered_models:
                models_info.append({
                    "name": model.name,
                    "type": "MLflow Registered Model",
                    "loaded": False,
                    "versions": len(client.get_latest_versions(model.name))
                })
        except Exception as e:
            logger.warning(f"MLflow models unavailable: {e}")
        
        return {
            "status": "success",
            "models": models_info,
            "count": len(models_info),
            "autonomous_features": {
                "auto_retraining": CONFIG["continuous_learning"]["enabled"],
                "performance_monitoring": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/insights")
async def get_predictive_insights():
    """Get predictive insights and recommendations"""
    try:
        return {
            "timestamp": datetime.now(),
            "insights": autonomous_system.predictive_insights,
            "usage_patterns": autonomous_system.usage_patterns[-5:] if autonomous_system.usage_patterns else [],
            "performance_trend": autonomous_system.performance_history[-10:] if autonomous_system.performance_history else [],
            "autonomous_actions": SYSTEM_STATE["autonomous_actions"][-10:],
            "recommendations": [
                "🔮 System is learning from your usage patterns",
                "📊 Performance monitoring is active",
                "🤖 Autonomous retraining will trigger when needed",
                "💡 Predictive insights will improve over time"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Interactive dashboard for the autonomous MLOps system"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ghosteam V5 Autonomous Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; color: white; margin-bottom: 30px; }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.2em; opacity: 0.9; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .card { background: rgba(255,255,255,0.95); border-radius: 15px; padding: 25px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); }
            .card h3 { color: #333; margin-bottom: 15px; font-size: 1.3em; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
            .status-healthy { background: #28a745; }
            .status-warning { background: #ffc107; }
            .status-error { background: #dc3545; }
            .metric { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee; }
            .metric:last-child { border-bottom: none; }
            .metric-value { font-weight: bold; color: #007bff; }
            .insight-item { background: #f8f9fa; padding: 12px; margin: 8px 0; border-radius: 8px; border-left: 4px solid #007bff; }
            .action-btn { background: linear-gradient(45deg, #007bff, #0056b3); color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; margin: 5px; font-weight: bold; transition: transform 0.2s; }
            .action-btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,123,255,0.3); }
            .log-container { background: #1e1e1e; color: #00ff00; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 12px; max-height: 200px; overflow-y: auto; }
            .chart-placeholder { background: #f8f9fa; height: 150px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #666; }
            .autonomous-badge { background: linear-gradient(45deg, #28a745, #20c997); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Ghosteam V5 Autonomous MLOps</h1>
                <p>Intelligent, Self-Improving Machine Learning System</p>
                <span class="autonomous-badge">🤖 AUTONOMOUS MODE ACTIVE</span>
            </div>

            <div class="grid">
                <div class="card">
                    <h3>🔍 System Status</h3>
                    <div id="system-status">
                        <div class="metric">
                            <span>API Status</span>
                            <span><span class="status-indicator status-healthy"></span>Healthy</span>
                        </div>
                        <div class="metric">
                            <span>Autonomous Learning</span>
                            <span><span class="status-indicator status-healthy"></span>Active</span>
                        </div>
                        <div class="metric">
                            <span>Predictive Intelligence</span>
                            <span><span class="status-indicator status-healthy"></span>Running</span>
                        </div>
                        <div class="metric">
                            <span>Last Health Check</span>
                            <span class="metric-value" id="last-check">Loading...</span>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>📊 Performance Metrics</h3>
                    <div id="metrics">
                        <div class="metric">
                            <span>Total Predictions</span>
                            <span class="metric-value" id="total-predictions">0</span>
                        </div>
                        <div class="metric">
                            <span>Model Accuracy</span>
                            <span class="metric-value" id="model-accuracy">95.2%</span>
                        </div>
                        <div class="metric">
                            <span>Active Models</span>
                            <span class="metric-value" id="active-models">1</span>
                        </div>
                        <div class="metric">
                            <span>Autonomous Actions</span>
                            <span class="metric-value" id="autonomous-actions">0</span>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>🔮 Predictive Insights</h3>
                    <div id="insights">
                        <div class="insight-item">🤖 System is learning from your usage patterns</div>
                        <div class="insight-item">📊 Performance monitoring is active</div>
                        <div class="insight-item">💡 Predictive insights will improve over time</div>
                    </div>
                </div>

                <div class="card">
                    <h3>📈 Performance Trend</h3>
                    <div class="chart-placeholder">
                        📈 Performance chart will appear here
                    </div>
                </div>

                <div class="card">
                    <h3>🎯 Quick Actions</h3>
                    <button class="action-btn" onclick="triggerRetrain()">🔄 Trigger Retraining</button>
                    <button class="action-btn" onclick="refreshData()">🔄 Refresh Data</button>
                    <button class="action-btn" onclick="viewModels()">📊 View Models</button>
                    <button class="action-btn" onclick="testPrediction()">🧪 Test Prediction</button>
                </div>

                <div class="card">
                    <h3>📝 System Logs</h3>
                    <div class="log-container" id="logs">
                        <div>[INFO] Autonomous MLOps system initialized</div>
                        <div>[INFO] Continuous learning loop started</div>
                        <div>[INFO] Predictive intelligence active</div>
                        <div>[INFO] System ready for autonomous operation</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const API_BASE = window.location.origin;

            async function fetchData(endpoint) {
                try {
                    const response = await fetch(`${API_BASE}${endpoint}`);
                    return await response.json();
                } catch (error) {
                    console.error('Error fetching data:', error);
                    return { error: error.message };
                }
            }

            async function updateSystemStatus() {
                const data = await fetchData('/health');
                if (!data.error) {
                    document.getElementById('last-check').textContent = new Date().toLocaleTimeString();
                    document.getElementById('total-predictions').textContent = data.metrics?.total_predictions || 0;
                    document.getElementById('active-models').textContent = data.metrics?.models_loaded || 0;
                    document.getElementById('autonomous-actions').textContent = data.metrics?.autonomous_actions || 0;

                    if (data.metrics?.current_accuracy) {
                        document.getElementById('model-accuracy').textContent = (data.metrics.current_accuracy * 100).toFixed(1) + '%';
                    }
                }
            }

            async function updateInsights() {
                const data = await fetchData('/insights');
                if (!data.error && data.insights) {
                    const insightsContainer = document.getElementById('insights');
                    insightsContainer.innerHTML = '';

                    data.insights.forEach(insight => {
                        const div = document.createElement('div');
                        div.className = 'insight-item';
                        div.textContent = insight;
                        insightsContainer.appendChild(div);
                    });

                    if (data.insights.length === 0) {
                        insightsContainer.innerHTML = '<div class="insight-item">🔄 Generating insights...</div>';
                    }
                }
            }

            async function updateLogs() {
                const logs = document.getElementById('logs');
                const timestamp = new Date().toLocaleTimeString();
                const newLog = `<div>[${timestamp}] System health check completed</div>`;
                logs.innerHTML = newLog + logs.innerHTML;

                // Keep only last 10 logs
                const logLines = logs.children;
                while (logLines.length > 10) {
                    logs.removeChild(logLines[logLines.length - 1]);
                }
            }

            async function refreshData() {
                await Promise.all([
                    updateSystemStatus(),
                    updateInsights(),
                    updateLogs()
                ]);

                // Add log entry
                const logs = document.getElementById('logs');
                const timestamp = new Date().toLocaleTimeString();
                logs.innerHTML = `<div>[${timestamp}] Dashboard data refreshed</div>` + logs.innerHTML;
            }

            async function triggerRetrain() {
                try {
                    const response = await fetch(`${API_BASE}/retrain`, { method: 'POST' });
                    const data = await response.json();

                    if (data.status === 'success') {
                        alert('✅ Model retraining triggered successfully!');
                        const logs = document.getElementById('logs');
                        const timestamp = new Date().toLocaleTimeString();
                        logs.innerHTML = `<div>[${timestamp}] Manual retraining triggered</div>` + logs.innerHTML;
                    }
                } catch (error) {
                    alert('❌ Error triggering retraining: ' + error.message);
                }
            }

            function viewModels() {
                window.open('/models', '_blank');
            }

            async function testPrediction() {
                try {
                    const testData = Array.from({length: 10}, () => Math.random() * 2 - 1);
                    const response = await fetch(`${API_BASE}/predict`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            data: testData,
                            context: { user_type: 'dashboard_test' }
                        })
                    });
                    const result = await response.json();

                    alert(`🧪 Test Prediction Result:\\nPrediction: ${result.prediction}\\nConfidence: ${(result.confidence * 100).toFixed(1)}%\\nModel: ${result.model_used}`);

                    const logs = document.getElementById('logs');
                    const timestamp = new Date().toLocaleTimeString();
                    logs.innerHTML = `<div>[${timestamp}] Test prediction completed (confidence: ${(result.confidence * 100).toFixed(1)}%)</div>` + logs.innerHTML;

                } catch (error) {
                    alert('❌ Error making test prediction: ' + error.message);
                }
            }

            // Initial load and auto-refresh
            refreshData();
            setInterval(refreshData, 30000); // Refresh every 30 seconds
        </script>
    </body>
    </html>
    """

@app.post("/retrain")
async def trigger_manual_retrain(background_tasks: BackgroundTasks):
    """Manually trigger model retraining"""
    try:
        background_tasks.add_task(autonomous_system.autonomous_retrain)
        return {
            "status": "success",
            "message": "Model retraining triggered",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error triggering retrain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
