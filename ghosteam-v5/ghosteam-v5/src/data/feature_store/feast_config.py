try:
    from feast import FeatureStore, Entity, FeatureView, Field, FileSource
    from feast.types import Float32, Int32, String, Bool
    FEAST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Feast not available: {e}")
    FEAST_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class FeatureStore:
        def __init__(self, *args, **kwargs): pass
    class Entity:
        def __init__(self, *args, **kwargs): pass
    class FeatureView:
        def __init__(self, *args, **kwargs): pass
    class Field:
        def __init__(self, *args, **kwargs): pass
    class FileSource:
        def __init__(self, *args, **kwargs): pass
    class Float32: pass
    class Int32: pass
    class String: pass
    class Bool: pass

from datetime import timedelta
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Define entities only if Feast is available
if FEAST_AVAILABLE:
    user_entity = Entity(
        name="user_id",
        value_type=String,
        description="User identifier"
    )

    model_entity = Entity(
        name="model_id",
        value_type=String,
        description="Model identifier"
    )

    session_entity = Entity(
        name="session_id",
        value_type=String,
        description="Session identifier"
    )
else:
    # Create dummy entities when Feast is not available
    user_entity = Entity()
    model_entity = Entity()
    session_entity = Entity()

# Define data sources only if Feast is available
if FEAST_AVAILABLE:
    user_features_source = FileSource(
        name="user_features_source",
        path="data/features/user_features.parquet",
        timestamp_field="event_timestamp"
    )

    model_performance_source = FileSource(
        name="model_performance_source",
        path="data/features/model_performance.parquet",
        timestamp_field="event_timestamp"
    )

    session_features_source = FileSource(
        name="session_features_source",
        path="data/features/session_features.parquet",
        timestamp_field="event_timestamp"
    )
else:
    # Create dummy sources when Feast is not available
    user_features_source = FileSource()
    model_performance_source = FileSource()
    session_features_source = FileSource()

# Define feature views
user_features = FeatureView(
    name="user_features",
    entities=[user_entity],
    schema=[
        Field(name="age", dtype=Int32),
        Field(name="activity_score", dtype=Float32),
        Field(name="engagement_rate", dtype=Float32),
        Field(name="total_sessions", dtype=Int32),
        Field(name="avg_session_duration", dtype=Float32),
        Field(name="is_premium", dtype=Bool),
        Field(name="days_since_signup", dtype=Int32),
        Field(name="last_activity_score", dtype=Float32),
    ],
    online=True,
    offline=True,
    ttl=timedelta(days=1),
    source=user_features_source,
    description="User demographic and behavioral features"
)

model_performance_features = FeatureView(
    name="model_performance",
    entities=[model_entity],
    schema=[
        Field(name="accuracy", dtype=Float32),
        Field(name="precision", dtype=Float32),
        Field(name="recall", dtype=Float32),
        Field(name="f1_score", dtype=Float32),
        Field(name="latency_p50", dtype=Float32),
        Field(name="latency_p95", dtype=Float32),
        Field(name="latency_p99", dtype=Float32),
        Field(name="prediction_count", dtype=Int32),
        Field(name="error_rate", dtype=Float32),
        Field(name="drift_score", dtype=Float32),
    ],
    online=True,
    offline=True,
    ttl=timedelta(hours=1),
    source=model_performance_source,
    description="Model performance and monitoring metrics"
)

session_features = FeatureView(
    name="session_features",
    entities=[session_entity],
    schema=[
        Field(name="session_duration", dtype=Float32),
        Field(name="page_views", dtype=Int32),
        Field(name="clicks", dtype=Int32),
        Field(name="bounce_rate", dtype=Float32),
        Field(name="conversion_rate", dtype=Float32),
        Field(name="device_type", dtype=String),
        Field(name="browser", dtype=String),
        Field(name="referrer_type", dtype=String),
    ],
    online=True,
    offline=True,
    ttl=timedelta(hours=6),
    source=session_features_source,
    description="Session-level behavioral features"
)

class FeatureStoreManager:
    """Manager for Feast feature store operations."""

    def __init__(self, repo_path: str = None):
        """Initialize feature store manager.

        Args:
            repo_path: Path to Feast repository
        """
        self.repo_path = repo_path or settings.feast_repo_path
        self.feast_available = FEAST_AVAILABLE

        if not FEAST_AVAILABLE:
            logger.warning("Feast not available - feature store will operate in mock mode")
            self.store = None
            return

        try:
            self.store = FeatureStore(repo_path=self.repo_path)
            logger.info(f"Initialized Feast feature store at {self.repo_path}")
        except Exception as e:
            logger.error(f"Failed to initialize feature store: {e}")
            self.store = None
            self.feast_available = False
    
    def get_online_features(self,
                           entity_rows: List[Dict[str, Any]],
                           features: List[str]) -> pd.DataFrame:
        """Get online features for real-time inference.

        Args:
            entity_rows: List of entity dictionaries
            features: List of feature names in format "feature_view:feature_name"

        Returns:
            DataFrame with features
        """
        if not self.feast_available or self.store is None:
            logger.warning("Feast not available - returning mock features")
            # Return mock features for testing/development
            mock_data = {}
            for entity_row in entity_rows:
                for key, value in entity_row.items():
                    if key not in mock_data:
                        mock_data[key] = []
                    mock_data[key].append(value)

            # Add mock feature values
            for feature in features:
                feature_name = feature.split(':')[-1] if ':' in feature else feature
                mock_data[feature_name] = [0.5] * len(entity_rows)

            return pd.DataFrame(mock_data)

        try:
            feature_vector = self.store.get_online_features(
                features=features,
                entity_rows=entity_rows
            )
            return feature_vector.to_df()
        except Exception as e:
            logger.error(f"Failed to get online features: {e}")
            raise
    
    def get_historical_features(self, 
                               entity_df: pd.DataFrame, 
                               features: List[str]) -> pd.DataFrame:
        """Get historical features for training.
        
        Args:
            entity_df: DataFrame with entities and timestamps
            features: List of feature names
            
        Returns:
            DataFrame with historical features
        """
        try:
            training_df = self.store.get_historical_features(
                entity_df=entity_df,
                features=features
            ).to_df()
            return training_df
        except Exception as e:
            logger.error(f"Failed to get historical features: {e}")
            raise
    
    def materialize_incremental(self, 
                               end_date: pd.Timestamp,
                               feature_views: Optional[List[str]] = None):
        """Materialize features incrementally to online store.
        
        Args:
            end_date: End date for materialization
            feature_views: List of feature view names to materialize
        """
        try:
            self.store.materialize_incremental(end_date=end_date)
            logger.info(f"Materialized features incrementally to {end_date}")
        except Exception as e:
            logger.error(f"Failed to materialize features: {e}")
            raise
    
    def materialize(self, 
                   start_date: pd.Timestamp,
                   end_date: pd.Timestamp,
                   feature_views: Optional[List[str]] = None):
        """Materialize features for a date range.
        
        Args:
            start_date: Start date for materialization
            end_date: End date for materialization
            feature_views: List of feature view names to materialize
        """
        try:
            self.store.materialize(
                start_date=start_date,
                end_date=end_date,
                feature_views=feature_views
            )
            logger.info(f"Materialized features from {start_date} to {end_date}")
        except Exception as e:
            logger.error(f"Failed to materialize features: {e}")
            raise
    
    def get_feature_view(self, name: str) -> FeatureView:
        """Get feature view by name.
        
        Args:
            name: Feature view name
            
        Returns:
            FeatureView object
        """
        try:
            return self.store.get_feature_view(name)
        except Exception as e:
            logger.error(f"Failed to get feature view {name}: {e}")
            raise
    
    def list_feature_views(self) -> List[FeatureView]:
        """List all feature views.
        
        Returns:
            List of FeatureView objects
        """
        try:
            return self.store.list_feature_views()
        except Exception as e:
            logger.error(f"Failed to list feature views: {e}")
            raise
    
    def validate_features(self, features: List[str]) -> bool:
        """Validate that features exist in the feature store.

        Args:
            features: List of feature names

        Returns:
            True if all features exist
        """
        if not self.feast_available or self.store is None:
            logger.warning("Feast not available - skipping feature validation")
            return True  # Allow all features in mock mode

        try:
            available_features = []
            for fv in self.list_feature_views():
                for field in fv.schema:
                    available_features.append(f"{fv.name}:{field.name}")

            missing_features = set(features) - set(available_features)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                return False

            return True
        except Exception as e:
            logger.error(f"Failed to validate features: {e}")
            return False
    
    def get_feature_statistics(self, feature_view_name: str) -> Dict[str, Any]:
        """Get statistics for features in a feature view.
        
        Args:
            feature_view_name: Name of the feature view
            
        Returns:
            Dictionary with feature statistics
        """
        try:
            # This would typically query your offline store for statistics
            # For now, return a placeholder
            return {
                "feature_view": feature_view_name,
                "num_features": len(self.get_feature_view(feature_view_name).schema),
                "last_updated": pd.Timestamp.now(),
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Failed to get feature statistics: {e}")
            return {}


# Global feature store manager instance
feature_store_manager = FeatureStoreManager()
