import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from src.serving.api.main import app
from src.data.feature_store.feast_config import FeatureStoreManager
from src.monitoring.drift_detection.drift_detector import DriftDetector
from src.models.training.sklearn_models import SklearnClassifier


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'user_id': [f'user_{i}' for i in range(n_samples)],
        'age': np.random.randint(18, 80, n_samples),
        'activity_score': np.random.uniform(0, 1, n_samples),
        'engagement_rate': np.random.uniform(0, 1, n_samples),
        'total_sessions': np.random.randint(0, 100, n_samples),
        'avg_session_duration': np.random.uniform(0, 3600, n_samples),
        'is_premium': np.random.choice([True, False], n_samples),
        'days_since_signup': np.random.randint(0, 365, n_samples),
        'target': np.random.binomial(1, 0.3, n_samples),
        'event_timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H')
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_training_data():
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, pd.Series(y, name='target')


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with patch('mlflow.start_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metrics'), \
         patch('mlflow.sklearn.log_model'), \
         patch('mlflow.set_experiment'):
        yield


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = Mock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.lpush.return_value = 1
    mock_redis.ltrim.return_value = True
    mock_redis.lrange.return_value = []
    
    with patch('redis.Redis.from_url', return_value=mock_redis):
        yield mock_redis


@pytest.fixture
def mock_feature_store():
    """Mock feature store for testing."""
    mock_fs = Mock(spec=FeatureStoreManager)
    
    # Mock online features response
    mock_fs.get_online_features.return_value.to_df.return_value = pd.DataFrame({
        'user_id': ['user_1', 'user_2'],
        'age': [25, 30],
        'activity_score': [0.8, 0.6]
    })
    
    # Mock historical features response
    mock_fs.get_historical_features.return_value.to_df.return_value = pd.DataFrame({
        'user_id': ['user_1', 'user_2', 'user_3'],
        'age': [25, 30, 35],
        'activity_score': [0.8, 0.6, 0.7],
        'target': [1, 0, 1]
    })
    
    mock_fs.validate_features.return_value = True
    mock_fs.list_feature_views.return_value = []
    
    return mock_fs


@pytest.fixture
def drift_detector():
    """Create drift detector instance."""
    return DriftDetector(methods=["ks", "mmd"], threshold=0.05)


@pytest.fixture
def sklearn_classifier():
    """Create sklearn classifier instance."""
    return SklearnClassifier(
        model_name="test_classifier",
        algorithm="random_forest",
        n_estimators=10,
        random_state=42
    )


@pytest.fixture
def mock_model():
    """Mock trained model for testing."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0.7, 0.3])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.7, 0.3]])
    return mock_model


@pytest.fixture
def sample_drift_data():
    """Generate sample data for drift testing."""
    np.random.seed(42)
    
    # Reference data
    reference = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(0, 1, 1000),
        'feature_3': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.binomial(1, 0.3, 1000)
    })
    
    # Current data with drift
    current = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1, 500),  # Mean shift
        'feature_2': np.random.normal(0, 1.5, 500),  # Variance shift
        'feature_3': np.random.choice(['A', 'B', 'C', 'D'], 500),  # New category
        'target': np.random.binomial(1, 0.4, 500)  # Target shift
    })
    
    return reference, current


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test_db.db")


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data."""
    return {
        "entity_ids": ["user_1", "user_2"],
        "features": ["user_features:age", "user_features:activity_score"],
        "model_name": "test_model",
        "model_version": "1.0"
    }


@pytest.fixture
def sample_batch_prediction_request():
    """Sample batch prediction request data."""
    return {
        "data": [
            {"user_id": "user_1", "age": 25, "activity_score": 0.8},
            {"user_id": "user_2", "age": 30, "activity_score": 0.6}
        ],
        "model_name": "test_model",
        "model_version": "1.0"
    }


# Performance testing fixtures
@pytest.fixture
def performance_data():
    """Generate larger dataset for performance testing."""
    np.random.seed(42)
    n_samples = 10000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, pd.Series(y, name='target')


# Database fixtures
@pytest.fixture
def test_database_url():
    """Test database URL."""
    return "sqlite:///test_ghosteam.db"


# Async fixtures
@pytest.fixture
async def async_client():
    """Async test client."""
    from httpx import AsyncClient
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
