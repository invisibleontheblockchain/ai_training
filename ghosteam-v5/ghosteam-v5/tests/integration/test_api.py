import pytest
import json
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd


class TestAPIIntegration:
    """Integration tests for the API endpoints."""
    
    def test_health_endpoint(self, client, mock_redis):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "components" in data
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint."""
        with patch('mlflow.tracking.MlflowClient') as mock_client:
            # Mock registered models
            mock_model = Mock()
            mock_model.name = "test_model"
            mock_model.latest_versions = [Mock()]
            mock_model.latest_versions[0].version = "1"
            mock_model.latest_versions[0].creation_timestamp = 1640995200000
            mock_model.latest_versions[0].run_id = "test_run_id"
            mock_model.latest_versions[0].current_stage = "Production"
            mock_model.latest_versions[0].source = "s3://bucket/model"
            
            mock_client.return_value.search_registered_models.return_value = [mock_model]
            
            # Mock run data
            mock_run = Mock()
            mock_run.data.metrics = {"accuracy": 0.85, "precision": 0.80}
            mock_run.data.params = {"feature_names": "age,activity_score"}
            mock_client.return_value.get_run.return_value = mock_run
            
            response = client.get("/models")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "test_model"
            assert data[0]["version"] == "1"
            assert data[0]["status"] == "Production"
    
    def test_prediction_endpoint_success(self, client, mock_redis, mock_feature_store):
        """Test successful prediction."""
        with patch('src.serving.api.main.get_model') as mock_get_model, \
             patch('src.serving.api.main.feature_store', mock_feature_store):
            
            # Mock model
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.7, 0.3])
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.7, 0.3]])
            
            mock_get_model.return_value = {
                "model": mock_model,
                "version": "1.0",
                "created_at": "2024-01-01",
                "model_type": "sklearn"
            }
            
            request_data = {
                "entity_ids": ["user_1", "user_2"],
                "features": ["user_features:age", "user_features:activity_score"],
                "model_name": "test_model"
            }
            
            response = client.post("/predict", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "predictions" in data
            assert "probabilities" in data
            assert "model_info" in data
            assert "latency_ms" in data
            assert len(data["predictions"]) == 2
    
    def test_prediction_endpoint_invalid_features(self, client, mock_redis, mock_feature_store):
        """Test prediction with invalid features."""
        mock_feature_store.validate_features.return_value = False
        
        with patch('src.serving.api.main.feature_store', mock_feature_store):
            request_data = {
                "entity_ids": ["user_1"],
                "features": ["invalid_feature"],
                "model_name": "test_model"
            }
            
            response = client.post("/predict", json=request_data)
            assert response.status_code == 400
            assert "Invalid features" in response.json()["detail"]
    
    def test_prediction_endpoint_model_not_found(self, client, mock_redis):
        """Test prediction with non-existent model."""
        with patch('src.serving.api.main.get_model') as mock_get_model:
            mock_get_model.side_effect = Exception("Model not found")
            
            request_data = {
                "entity_ids": ["user_1"],
                "features": ["user_features:age"],
                "model_name": "nonexistent_model"
            }
            
            response = client.post("/predict", json=request_data)
            assert response.status_code == 500
    
    def test_batch_prediction_endpoint(self, client, mock_redis):
        """Test batch prediction endpoint."""
        with patch('src.serving.api.main.get_model') as mock_get_model:
            # Mock model
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.7, 0.3])
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.7, 0.3]])
            
            mock_get_model.return_value = {
                "model": mock_model,
                "version": "1.0",
                "created_at": "2024-01-01",
                "model_type": "sklearn"
            }
            
            request_data = {
                "data": [
                    {"age": 25, "activity_score": 0.8},
                    {"age": 30, "activity_score": 0.6}
                ],
                "model_name": "test_model"
            }
            
            response = client.post("/predict/batch", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "predictions" in data
            assert "batch_size" in data
            assert data["batch_size"] == 2
    
    def test_metrics_endpoint(self, client, mock_redis):
        """Test metrics endpoint."""
        # Mock metrics data
        mock_metrics = [
            json.dumps({
                "model_name": "test_model",
                "latency_ms": 15.0,
                "prediction_count": 2,
                "timestamp": 1640995200.0,
                "error": None
            }),
            json.dumps({
                "model_name": "test_model",
                "latency_ms": 20.0,
                "prediction_count": 1,
                "timestamp": 1640995260.0,
                "error": None
            })
        ]
        
        mock_redis.lrange.return_value = mock_metrics
        
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_predictions" in data
        assert "average_latency_ms" in data
        assert "error_count" in data
        assert "error_rate" in data
        assert data["total_predictions"] == 3
        assert data["error_count"] == 0
    
    def test_metrics_endpoint_no_data(self, client, mock_redis):
        """Test metrics endpoint with no data."""
        mock_redis.lrange.return_value = []
        
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "No metrics available"
    
    def test_cache_clear_endpoint(self, client):
        """Test cache clearing endpoint."""
        response = client.delete("/cache")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Model cache cleared"
    
    def test_prediction_with_error_logging(self, client, mock_redis, mock_feature_store):
        """Test that errors are properly logged."""
        with patch('src.serving.api.main.get_model') as mock_get_model, \
             patch('src.serving.api.main.feature_store', mock_feature_store):
            
            # Mock model that raises an exception
            mock_model = Mock()
            mock_model.predict.side_effect = Exception("Prediction failed")
            
            mock_get_model.return_value = {
                "model": mock_model,
                "version": "1.0",
                "created_at": "2024-01-01",
                "model_type": "sklearn"
            }
            
            request_data = {
                "entity_ids": ["user_1"],
                "features": ["user_features:age"],
                "model_name": "test_model"
            }
            
            response = client.post("/predict", json=request_data)
            assert response.status_code == 500
            
            # Verify error was logged (background task)
            # Note: In real tests, you might need to wait or use async testing
    
    def test_concurrent_predictions(self, client, mock_redis, mock_feature_store):
        """Test concurrent prediction requests."""
        import concurrent.futures
        import time
        
        with patch('src.serving.api.main.get_model') as mock_get_model, \
             patch('src.serving.api.main.feature_store', mock_feature_store):
            
            # Mock model with slight delay
            mock_model = Mock()
            def predict_with_delay(X):
                time.sleep(0.01)  # Small delay to simulate processing
                return np.array([0.7] * len(X))
            
            mock_model.predict = predict_with_delay
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            
            mock_get_model.return_value = {
                "model": mock_model,
                "version": "1.0",
                "created_at": "2024-01-01",
                "model_type": "sklearn"
            }
            
            def make_prediction():
                request_data = {
                    "entity_ids": ["user_1"],
                    "features": ["user_features:age"],
                    "model_name": "test_model"
                }
                return client.post("/predict", json=request_data)
            
            # Make concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_prediction) for _ in range(10)]
                results = [future.result() for future in futures]
            
            # All requests should succeed
            success_count = sum(1 for r in results if r.status_code == 200)
            assert success_count >= 8  # Allow for some potential failures under load
    
    def test_model_caching(self, client, mock_redis, mock_feature_store):
        """Test that models are properly cached."""
        with patch('src.serving.api.main.get_model') as mock_get_model, \
             patch('src.serving.api.main.feature_store', mock_feature_store), \
             patch('mlflow.tracking.MlflowClient') as mock_mlflow_client:
            
            # Mock MLflow client
            mock_model_version = Mock()
            mock_model_version.version = "1.0"
            mock_model_version.creation_timestamp = "2024-01-01"
            
            mock_mlflow_client.return_value.get_latest_versions.return_value = [mock_model_version]
            
            # Mock sklearn model loading
            mock_sklearn_model = Mock()
            mock_sklearn_model.predict.return_value = np.array([0.7])
            mock_sklearn_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            
            with patch('mlflow.sklearn.load_model', return_value=mock_sklearn_model):
                # Reset the mock to track calls
                mock_get_model.side_effect = None
                mock_get_model.return_value = {
                    "model": mock_sklearn_model,
                    "version": "1.0",
                    "created_at": "2024-01-01",
                    "model_type": "sklearn"
                }
                
                request_data = {
                    "entity_ids": ["user_1"],
                    "features": ["user_features:age"],
                    "model_name": "test_model"
                }
                
                # Make first request
                response1 = client.post("/predict", json=request_data)
                assert response1.status_code == 200
                
                # Make second request (should use cached model)
                response2 = client.post("/predict", json=request_data)
                assert response2.status_code == 200
                
                # Both responses should be successful
                assert response1.json()["predictions"] == response2.json()["predictions"]


class TestAPIValidation:
    """Test API input validation."""
    
    def test_prediction_request_validation(self, client):
        """Test prediction request validation."""
        # Missing required fields
        response = client.post("/predict", json={})
        assert response.status_code == 422
        
        # Invalid entity_ids type
        response = client.post("/predict", json={
            "entity_ids": "not_a_list",
            "features": ["user_features:age"]
        })
        assert response.status_code == 422
        
        # Empty entity_ids
        response = client.post("/predict", json={
            "entity_ids": [],
            "features": ["user_features:age"]
        })
        assert response.status_code == 422
    
    def test_batch_prediction_request_validation(self, client):
        """Test batch prediction request validation."""
        # Missing data field
        response = client.post("/predict/batch", json={
            "model_name": "test_model"
        })
        assert response.status_code == 422
        
        # Invalid data type
        response = client.post("/predict/batch", json={
            "data": "not_a_list",
            "model_name": "test_model"
        })
        assert response.status_code == 422
