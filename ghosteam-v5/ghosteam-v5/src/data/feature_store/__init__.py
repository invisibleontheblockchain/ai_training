"""Feature store components."""

try:
    from .feast_config import FeatureStoreManager
    __all__ = ["FeatureStoreManager"]
except ImportError as e:
    print(f"Warning: FeatureStoreManager not available: {e}")

    # Create a mock FeatureStoreManager for graceful degradation
    class FeatureStoreManager:
        def __init__(self, *args, **kwargs):
            print("Warning: Using mock FeatureStoreManager - feast not available")
            self.feast_available = False

        def get_online_features(self, entity_rows, features):
            import pandas as pd
            # Return mock features
            mock_data = {}
            for entity_row in entity_rows:
                for key, value in entity_row.items():
                    if key not in mock_data:
                        mock_data[key] = []
                    mock_data[key].append(value)

            for feature in features:
                feature_name = feature.split(':')[-1] if ':' in feature else feature
                mock_data[feature_name] = [0.5] * len(entity_rows)

            return pd.DataFrame(mock_data)

        def validate_features(self, features):
            return True  # Allow all features in mock mode

        def list_feature_views(self):
            return []

    __all__ = ["FeatureStoreManager"]
