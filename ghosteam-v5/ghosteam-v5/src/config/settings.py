import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = Field(default="Ghosteam V5", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")
    
    # Database
    database_url: Optional[str] = Field(
        default=None, 
        description="Database connection URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379", 
        description="Redis connection URL"
    )
    
    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    mlflow_registry_uri: Optional[str] = Field(
        default=None,
        description="MLflow model registry URI"
    )
    mlflow_artifact_root: str = Field(
        default="./mlruns",
        description="MLflow artifact storage location"
    )
    
    # Feast Feature Store
    feast_repo_path: str = Field(
        default="./feast_repo",
        description="Feast repository path"
    )
    feast_online_store_path: str = Field(
        default="online_store.db",
        description="Feast online store database path"
    )
    feast_offline_store_type: str = Field(
        default="file",
        description="Feast offline store type"
    )
    feast_online_store_type: str = Field(
        default="redis",
        description="Feast online store type"
    )
    
    # Model Serving
    model_serving_host: str = Field(
        default="0.0.0.0",
        description="Model serving host"
    )
    model_serving_port: int = Field(
        default=8080,
        description="Model serving port"
    )
    model_cache_size: int = Field(
        default=10,
        description="Maximum number of models to cache"
    )
    model_cache_ttl: int = Field(
        default=3600,
        description="Model cache TTL in seconds"
    )
    
    # Monitoring
    prometheus_gateway: str = Field(
        default="http://localhost:9091",
        description="Prometheus pushgateway URL"
    )
    grafana_url: str = Field(
        default="http://localhost:3000",
        description="Grafana dashboard URL"
    )
    grafana_api_key: Optional[str] = Field(
        default=None,
        description="Grafana API key"
    )
    
    # Security
    secret_key: str = Field(
        default_factory=lambda: os.urandom(32).hex(),
        description="Secret key for JWT tokens"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    
    # Kubernetes
    k8s_namespace: str = Field(
        default="ghosteam-v5",
        description="Kubernetes namespace"
    )
    kserve_domain: str = Field(
        default="example.com",
        description="KServe domain"
    )
    kubeflow_namespace: str = Field(
        default="kubeflow",
        description="Kubeflow namespace"
    )
    
    # Storage
    s3_bucket: Optional[str] = Field(
        default=None,
        description="S3 bucket name"
    )
    s3_region: str = Field(
        default="us-west-2",
        description="S3 region"
    )
    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS secret access key"
    )
    
    # Azure Storage (optional)
    azure_storage_account: Optional[str] = Field(
        default=None,
        description="Azure storage account name"
    )
    azure_storage_key: Optional[str] = Field(
        default=None,
        description="Azure storage account key"
    )
    azure_container_name: Optional[str] = Field(
        default=None,
        description="Azure container name"
    )
    
    # Google Cloud Storage (optional)
    google_application_credentials: Optional[str] = Field(
        default=None,
        description="Path to Google Cloud service account JSON"
    )
    gcs_bucket: Optional[str] = Field(
        default=None,
        description="Google Cloud Storage bucket name"
    )
    
    # Drift Detection
    drift_detection_threshold: float = Field(
        default=0.05,
        description="Drift detection threshold"
    )
    drift_check_interval_hours: int = Field(
        default=24,
        description="Drift check interval in hours"
    )
    drift_reference_window_days: int = Field(
        default=30,
        description="Reference window for drift detection in days"
    )
    
    # Model Training
    default_model_type: str = Field(
        default="random_forest",
        description="Default model type for training"
    )
    training_data_window_days: int = Field(
        default=90,
        description="Training data window in days"
    )
    validation_split: float = Field(
        default=0.2,
        description="Validation split ratio"
    )
    test_split: float = Field(
        default=0.1,
        description="Test split ratio"
    )
    
    # Performance Monitoring
    max_prediction_latency_ms: int = Field(
        default=100,
        description="Maximum acceptable prediction latency in milliseconds"
    )
    min_model_accuracy: float = Field(
        default=0.85,
        description="Minimum acceptable model accuracy"
    )
    alert_email: Optional[str] = Field(
        default=None,
        description="Email for alerts"
    )
    slack_webhook_url: Optional[str] = Field(
        default=None,
        description="Slack webhook URL for alerts"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path"
    )
    
    # Development
    reload: bool = Field(
        default=False,
        description="Auto-reload for development"
    )
    workers: int = Field(
        default=1,
        description="Number of worker processes"
    )
    
    # Production
    max_workers: int = Field(
        default=16,
        description="Maximum number of workers"
    )
    worker_timeout: int = Field(
        default=30,
        description="Worker timeout in seconds"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
    
    def get_database_url(self) -> str:
        """Get database URL with fallback."""
        if self.database_url:
            return self.database_url
        return "sqlite:///./ghosteam_v5.db"
    
    def get_mlflow_backend_store_uri(self) -> str:
        """Get MLflow backend store URI."""
        if self.mlflow_registry_uri:
            return self.mlflow_registry_uri
        return "sqlite:///./mlflow.db"
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def get_allowed_hosts(self) -> List[str]:
        """Get allowed hosts for CORS."""
        if self.is_production():
            return [
                f"https://{self.kserve_domain}",
                f"https://api.{self.kserve_domain}",
                f"https://mlflow.{self.kserve_domain}",
                f"https://grafana.{self.kserve_domain}"
            ]
        return ["*"]


# Global settings instance
settings = Settings()
