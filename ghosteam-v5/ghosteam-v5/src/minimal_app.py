
import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Ghosteam V5 MLOps API",
    version="1.0.0",
    description="Production MLOps system with MLflow integration"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Ghosteam V5 MLOps System",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "mlflow_version": mlflow.__version__,
        "services": {
            "api": "running",
            "mlflow": "available"
        }
    }

@app.get("/models")
async def list_models():
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()

        model_list = []
        for model in models:
            versions = client.get_latest_versions(model.name)
            model_info = {
                "name": model.name,
                "versions": [{"version": v.version, "stage": v.current_stage} for v in versions]
            }
            model_list.append(model_info)

        return {
            "status": "success",
            "models": model_list,
            "count": len(models)
        }
    except Exception as e:
        logger.warning(f"MLflow client error: {e}")
        return {
            "status": "mlflow_unavailable",
            "error": str(e),
            "models": [],
            "count": 0
        }

@app.post("/predict")
async def predict(data: dict):
    # Placeholder for model prediction
    return {
        "status": "success",
        "prediction": "placeholder",
        "model": "ghosteam-v5-model",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
