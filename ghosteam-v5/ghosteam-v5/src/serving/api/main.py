from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import mlflow
import numpy as np
import pandas as pd
import redis
import json
import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from src.config.settings import settings
from src.data.feature_store.feast_config import FeatureStoreManager
from src.monitoring.drift_detection.drift_detector import DriftDetector

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Global variables
redis_client = None
feature_store = None
model_cache = {}
drift_detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global redis_client, feature_store, drift_detector
    
    try:
        # Initialize Redis
        redis_client = redis.Redis.from_url(settings.redis_url)
        logger.info("Redis client initialized")
        
        # Initialize Feature Store
        feature_store = FeatureStoreManager()
        logger.info("Feature store initialized")
        
        # Initialize Drift Detector
        drift_detector = DriftDetector()
        logger.info("Drift detector initialized")
        
        logger.info("Application startup completed")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    if redis_client:
        redis_client.close()
    logger.info("Application shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Ghosteam V5 Model Serving API",
    version="1.0.0",
    description="Production model serving for Ghosteam V5 learning system",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_hosts(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    entity_ids: List[str] = Field(..., description="List of entity IDs")
    features: List[str] = Field(..., description="List of feature names")
    model_name: str = Field(default="ghosteam_v5_model", description="Model name")
    model_version: str = Field(default="latest", description="Model version")
    
    class Config:
        schema_extra = {
            "example": {
                "entity_ids": ["user_1", "user_2"],
                "features": ["user_features:age", "user_features:activity_score"],
                "model_name": "ghosteam_v5_model",
                "model_version": "latest"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float] = Field(..., description="Model predictions")
    probabilities: Optional[List[List[float]]] = Field(None, description="Class probabilities")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [0.75, 0.82],
                "probabilities": [[0.25, 0.75], [0.18, 0.82]],
                "model_info": {
                    "name": "ghosteam_v5_model",
                    "version": "1.0",
                    "created_at": "2024-01-01T00:00:00"
                },
                "latency_ms": 15.2,
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    data: List[Dict[str, Any]] = Field(..., description="Batch data for prediction")
    model_name: str = Field(default="ghosteam_v5_model", description="Model name")
    model_version: str = Field(default="latest", description="Model version")
    
    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {"user_id": "user_1", "age": 25, "activity_score": 0.8},
                    {"user_id": "user_2", "age": 30, "activity_score": 0.6}
                ],
                "model_name": "ghosteam_v5_model",
                "model_version": "latest"
            }
        }


class ModelInfo(BaseModel):
    """Model information response."""
    name: str
    version: str
    created_at: str
    model_type: str
    metrics: Dict[str, float]
    feature_count: int
    status: str


# Model cache management
def get_model(model_name: str, version: str = "latest") -> Dict[str, Any]:
    """Get model from cache or load from MLflow.
    
    Args:
        model_name: Name of the model
        version: Model version
        
    Returns:
        Dictionary containing model and metadata
    """
    cache_key = f"{model_name}:{version}"
    
    if cache_key not in model_cache:
        try:
            # Load model from MLflow
            client = mlflow.tracking.MlflowClient()
            
            if version == "latest":
                model_versions = client.get_latest_versions(
                    model_name, stages=["Production", "Staging"]
                )
                if not model_versions:
                    raise ValueError(f"No model versions found for {model_name}")
                model_version = model_versions[0]
            else:
                model_version = client.get_model_version(model_name, version)
            
            model_uri = f"models:/{model_name}/{model_version.version}"
            
            # Load model based on flavor
            try:
                model = mlflow.sklearn.load_model(model_uri)
                model_type = "sklearn"
            except:
                try:
                    model = mlflow.xgboost.load_model(model_uri)
                    model_type = "xgboost"
                except:
                    model = mlflow.pyfunc.load_model(model_uri)
                    model_type = "pyfunc"
            
            model_cache[cache_key] = {
                "model": model,
                "version": model_version.version,
                "created_at": model_version.creation_timestamp,
                "model_type": model_type,
                "loaded_at": time.time()
            }
            
            logger.info(f"Loaded model {model_name}:{model_version.version}")
            
            # Clean cache if too many models
            if len(model_cache) > settings.model_cache_size:
                oldest_key = min(model_cache.keys(), 
                               key=lambda k: model_cache[k]["loaded_at"])
                del model_cache[oldest_key]
                logger.info(f"Removed oldest model from cache: {oldest_key}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}:{version}: {e}")
            raise HTTPException(status_code=404, detail=f"Model not found: {e}")
    
    return model_cache[cache_key]


async def log_prediction_metrics(
    model_name: str,
    latency_ms: float,
    prediction_count: int,
    error: Optional[str] = None
):
    """Log prediction metrics to monitoring system."""
    try:
        metrics = {
            "model_name": model_name,
            "latency_ms": latency_ms,
            "prediction_count": prediction_count,
            "timestamp": time.time(),
            "error": error
        }
        
        # Store in Redis for monitoring
        if redis_client:
            redis_client.lpush("prediction_metrics", json.dumps(metrics))
            redis_client.ltrim("prediction_metrics", 0, 9999)  # Keep last 10k metrics
            
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Redis connection
        redis_status = "healthy" if redis_client and redis_client.ping() else "unhealthy"
        
        # Check feature store
        fs_status = "healthy"
        try:
            feature_store.list_feature_views()
        except:
            fs_status = "unhealthy"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "redis": redis_status,
                "feature_store": fs_status,
                "model_cache_size": len(model_cache)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models."""
    try:
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()
        
        model_list = []
        for model in models:
            latest_version = None
            if model.latest_versions:
                latest_version = model.latest_versions[0]
                
                # Get model metrics from MLflow
                run = client.get_run(latest_version.run_id)
                metrics = run.data.metrics
                
                model_list.append(ModelInfo(
                    name=model.name,
                    version=latest_version.version,
                    created_at=str(latest_version.creation_timestamp),
                    model_type=latest_version.source.split("/")[-1] if latest_version.source else "unknown",
                    metrics=metrics,
                    feature_count=len(run.data.params.get("feature_names", "").split(",")) if run.data.params.get("feature_names") else 0,
                    status=latest_version.current_stage
                ))
        
        return model_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Make predictions using the specified model."""
    start_time = time.time()
    
    try:
        # Get model
        model_info = get_model(request.model_name, request.model_version)
        model = model_info["model"]
        
        # Get features from feature store
        entity_rows = [{"user_id": eid} for eid in request.entity_ids]
        
        # Validate features exist
        if not feature_store.validate_features(request.features):
            raise HTTPException(status_code=400, detail="Invalid features specified")
        
        feature_vector = feature_store.get_online_features(
            features=request.features,
            entity_rows=entity_rows
        )
        
        # Convert to prediction format
        feature_df = feature_vector.to_df()
        X = feature_df.drop(['user_id'], axis=1, errors='ignore')
        
        if X.empty:
            raise HTTPException(status_code=400, detail="No features retrieved")
        
        # Make predictions
        predictions = model.predict(X)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X).tolist()
            except:
                pass
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log metrics in background
        background_tasks.add_task(
            log_prediction_metrics,
            request.model_name,
            latency_ms,
            len(predictions)
        )
        
        response = PredictionResponse(
            predictions=predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            probabilities=probabilities,
            model_info={
                "name": request.model_name,
                "version": model_info["version"],
                "created_at": str(model_info["created_at"]),
                "model_type": model_info["model_type"]
            },
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        
        # Log error metrics
        background_tasks.add_task(
            log_prediction_metrics,
            request.model_name,
            latency_ms,
            0,
            str(e)
        )
        
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch")
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """Make batch predictions."""
    start_time = time.time()
    
    try:
        # Get model
        model_info = get_model(request.model_name, request.model_version)
        model = model_info["model"]
        
        # Convert batch data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(df).tolist()
            except:
                pass
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log metrics
        background_tasks.add_task(
            log_prediction_metrics,
            request.model_name,
            latency_ms,
            len(predictions)
        )
        
        return {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "probabilities": probabilities,
            "model_info": {
                "name": request.model_name,
                "version": model_info["version"],
                "created_at": str(model_info["created_at"])
            },
            "latency_ms": latency_ms,
            "batch_size": len(request.data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


@app.get("/metrics")
async def get_metrics():
    """Get prediction metrics."""
    try:
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis not available")
        
        # Get recent metrics
        metrics_data = redis_client.lrange("prediction_metrics", 0, 99)
        metrics = [json.loads(m) for m in metrics_data]
        
        if not metrics:
            return {"message": "No metrics available"}
        
        # Calculate aggregated metrics
        total_predictions = sum(m["prediction_count"] for m in metrics)
        avg_latency = sum(m["latency_ms"] for m in metrics) / len(metrics)
        error_count = sum(1 for m in metrics if m.get("error"))
        
        return {
            "total_predictions": total_predictions,
            "average_latency_ms": avg_latency,
            "error_count": error_count,
            "error_rate": error_count / len(metrics) if metrics else 0,
            "recent_metrics": metrics[:10]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}")


@app.delete("/cache")
async def clear_cache():
    """Clear model cache."""
    global model_cache
    model_cache.clear()
    return {"message": "Model cache cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.model_serving_host,
        port=settings.model_serving_port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1
    )
