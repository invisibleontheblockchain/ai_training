import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Comprehensive feature engineering pipeline."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.transformations = {}
        self.feature_stats = {}
    
    def create_temporal_features(self, 
                                df: pd.DataFrame, 
                                timestamp_col: str = "event_timestamp") -> pd.DataFrame:
        """Create temporal features from timestamp column.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column {timestamp_col} not found")
            return df
        
        # Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract temporal components
        df[f"{timestamp_col}_year"] = df[timestamp_col].dt.year
        df[f"{timestamp_col}_month"] = df[timestamp_col].dt.month
        df[f"{timestamp_col}_day"] = df[timestamp_col].dt.day
        df[f"{timestamp_col}_hour"] = df[timestamp_col].dt.hour
        df[f"{timestamp_col}_dayofweek"] = df[timestamp_col].dt.dayofweek
        df[f"{timestamp_col}_quarter"] = df[timestamp_col].dt.quarter
        df[f"{timestamp_col}_is_weekend"] = df[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Create cyclical features for periodic patterns
        df[f"{timestamp_col}_hour_sin"] = np.sin(2 * np.pi * df[f"{timestamp_col}_hour"] / 24)
        df[f"{timestamp_col}_hour_cos"] = np.cos(2 * np.pi * df[f"{timestamp_col}_hour"] / 24)
        df[f"{timestamp_col}_dayofweek_sin"] = np.sin(2 * np.pi * df[f"{timestamp_col}_dayofweek"] / 7)
        df[f"{timestamp_col}_dayofweek_cos"] = np.cos(2 * np.pi * df[f"{timestamp_col}_dayofweek"] / 7)
        df[f"{timestamp_col}_month_sin"] = np.sin(2 * np.pi * df[f"{timestamp_col}_month"] / 12)
        df[f"{timestamp_col}_month_cos"] = np.cos(2 * np.pi * df[f"{timestamp_col}_month"] / 12)
        
        # Time since features
        reference_date = df[timestamp_col].max()
        df[f"{timestamp_col}_days_since"] = (reference_date - df[timestamp_col]).dt.days
        df[f"{timestamp_col}_hours_since"] = (reference_date - df[timestamp_col]).dt.total_seconds() / 3600
        
        logger.info(f"Created temporal features from {timestamp_col}")
        return df
    
    def create_aggregation_features(self, 
                                   df: pd.DataFrame,
                                   group_cols: List[str],
                                   agg_cols: List[str],
                                   agg_functions: List[str] = ["mean", "sum", "count", "std", "min", "max"]) -> pd.DataFrame:
        """Create aggregation features.
        
        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            agg_cols: Columns to aggregate
            agg_functions: Aggregation functions to apply
            
        Returns:
            DataFrame with aggregation features
        """
        df = df.copy()
        
        for group_col in group_cols:
            if group_col not in df.columns:
                logger.warning(f"Group column {group_col} not found")
                continue
                
            for agg_col in agg_cols:
                if agg_col not in df.columns:
                    logger.warning(f"Aggregation column {agg_col} not found")
                    continue
                
                for agg_func in agg_functions:
                    try:
                        feature_name = f"{group_col}_{agg_col}_{agg_func}"
                        
                        if agg_func == "count":
                            agg_values = df.groupby(group_col)[agg_col].count()
                        elif agg_func == "nunique":
                            agg_values = df.groupby(group_col)[agg_col].nunique()
                        else:
                            agg_values = df.groupby(group_col)[agg_col].agg(agg_func)
                        
                        df[feature_name] = df[group_col].map(agg_values)
                        
                    except Exception as e:
                        logger.warning(f"Failed to create aggregation feature {feature_name}: {e}")
        
        logger.info(f"Created aggregation features for {len(group_cols)} group columns")
        return df
    
    def create_interaction_features(self, 
                                   df: pd.DataFrame,
                                   feature_pairs: List[Tuple[str, str]],
                                   operations: List[str] = ["multiply", "add", "subtract", "divide"]) -> pd.DataFrame:
        """Create interaction features between feature pairs.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of feature pairs to interact
            operations: Operations to apply
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 not in df.columns or feat2 not in df.columns:
                logger.warning(f"Features {feat1} or {feat2} not found")
                continue
            
            # Only create interactions for numeric features
            if not (pd.api.types.is_numeric_dtype(df[feat1]) and pd.api.types.is_numeric_dtype(df[feat2])):
                continue
            
            for operation in operations:
                try:
                    feature_name = f"{feat1}_{operation}_{feat2}"
                    
                    if operation == "multiply":
                        df[feature_name] = df[feat1] * df[feat2]
                    elif operation == "add":
                        df[feature_name] = df[feat1] + df[feat2]
                    elif operation == "subtract":
                        df[feature_name] = df[feat1] - df[feat2]
                    elif operation == "divide":
                        # Avoid division by zero
                        df[feature_name] = df[feat1] / (df[feat2] + 1e-8)
                    elif operation == "ratio":
                        df[feature_name] = df[feat1] / (df[feat1] + df[feat2] + 1e-8)
                    
                except Exception as e:
                    logger.warning(f"Failed to create interaction feature {feature_name}: {e}")
        
        logger.info(f"Created interaction features for {len(feature_pairs)} feature pairs")
        return df
    
    def create_statistical_features(self, 
                                   df: pd.DataFrame,
                                   numeric_cols: List[str],
                                   window_sizes: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """Create rolling statistical features.
        
        Args:
            df: Input DataFrame (should be sorted by timestamp)
            numeric_cols: Numeric columns to create features for
            window_sizes: Rolling window sizes
            
        Returns:
            DataFrame with statistical features
        """
        df = df.copy()
        
        for col in numeric_cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found")
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            for window in window_sizes:
                try:
                    # Rolling statistics
                    df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window, min_periods=1).std()
                    df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window, min_periods=1).min()
                    df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window, min_periods=1).max()
                    
                    # Lag features
                    df[f"{col}_lag_{window}"] = df[col].shift(window)
                    
                    # Change features
                    df[f"{col}_change_{window}"] = df[col] - df[col].shift(window)
                    df[f"{col}_pct_change_{window}"] = df[col].pct_change(periods=window)
                    
                except Exception as e:
                    logger.warning(f"Failed to create statistical features for {col}: {e}")
        
        logger.info(f"Created statistical features for {len(numeric_cols)} columns")
        return df
    
    def handle_missing_values(self, 
                             df: pd.DataFrame,
                             strategy: str = "auto",
                             numeric_strategy: str = "median",
                             categorical_strategy: str = "most_frequent") -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Overall strategy ("auto", "drop", "impute")
            numeric_strategy: Strategy for numeric columns
            categorical_strategy: Strategy for categorical columns
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        # Get missing value statistics
        missing_stats = df.isnull().sum()
        missing_pct = (missing_stats / len(df)) * 100
        
        if strategy == "auto":
            # Drop columns with >50% missing values
            high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
            if high_missing_cols:
                df = df.drop(columns=high_missing_cols)
                logger.info(f"Dropped columns with >50% missing: {high_missing_cols}")
            
            # Drop rows with >80% missing values
            row_missing_pct = (df.isnull().sum(axis=1) / len(df.columns)) * 100
            high_missing_rows = row_missing_pct > 80
            if high_missing_rows.any():
                df = df[~high_missing_rows]
                logger.info(f"Dropped {high_missing_rows.sum()} rows with >80% missing values")
        
        # Impute remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numeric imputation
        if numeric_cols:
            if "numeric_imputer" not in self.imputers:
                if numeric_strategy == "knn":
                    self.imputers["numeric_imputer"] = KNNImputer(n_neighbors=5)
                else:
                    self.imputers["numeric_imputer"] = SimpleImputer(strategy=numeric_strategy)
            
            df[numeric_cols] = self.imputers["numeric_imputer"].fit_transform(df[numeric_cols])
        
        # Categorical imputation
        if categorical_cols:
            if "categorical_imputer" not in self.imputers:
                self.imputers["categorical_imputer"] = SimpleImputer(strategy=categorical_strategy)
            
            df[categorical_cols] = self.imputers["categorical_imputer"].fit_transform(df[categorical_cols])
        
        logger.info(f"Handled missing values using {strategy} strategy")
        return df
    
    def encode_categorical_features(self, 
                                   df: pd.DataFrame,
                                   categorical_cols: Optional[List[str]] = None,
                                   encoding_type: str = "auto",
                                   max_categories: int = 10) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical columns
            encoding_type: Type of encoding ("auto", "onehot", "label", "target")
            max_categories: Maximum categories for one-hot encoding
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            unique_count = df[col].nunique()
            
            # Determine encoding strategy
            if encoding_type == "auto":
                if unique_count <= max_categories:
                    strategy = "onehot"
                else:
                    strategy = "label"
            else:
                strategy = encoding_type
            
            try:
                if strategy == "onehot":
                    # One-hot encoding
                    if f"{col}_onehot" not in self.encoders:
                        self.encoders[f"{col}_onehot"] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded = self.encoders[f"{col}_onehot"].fit_transform(df[[col]])
                    else:
                        encoded = self.encoders[f"{col}_onehot"].transform(df[[col]])
                    
                    # Create column names
                    feature_names = [f"{col}_{cat}" for cat in self.encoders[f"{col}_onehot"].categories_[0]]
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                    
                    # Add encoded columns and drop original
                    df = pd.concat([df, encoded_df], axis=1)
                    df = df.drop(columns=[col])
                
                elif strategy == "label":
                    # Label encoding
                    if f"{col}_label" not in self.encoders:
                        self.encoders[f"{col}_label"] = LabelEncoder()
                        df[col] = self.encoders[f"{col}_label"].fit_transform(df[col].astype(str))
                    else:
                        df[col] = self.encoders[f"{col}_label"].transform(df[col].astype(str))
                
            except Exception as e:
                logger.warning(f"Failed to encode column {col}: {e}")
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return df
    
    def scale_features(self, 
                      df: pd.DataFrame,
                      numeric_cols: Optional[List[str]] = None,
                      scaling_type: str = "standard") -> pd.DataFrame:
        """Scale numeric features.
        
        Args:
            df: Input DataFrame
            numeric_cols: List of numeric columns to scale
            scaling_type: Type of scaling ("standard", "minmax", "robust")
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return df
        
        # Initialize scaler
        if scaling_type not in self.scalers:
            if scaling_type == "standard":
                self.scalers[scaling_type] = StandardScaler()
            elif scaling_type == "minmax":
                self.scalers[scaling_type] = MinMaxScaler()
            elif scaling_type == "robust":
                from sklearn.preprocessing import RobustScaler
                self.scalers[scaling_type] = RobustScaler()
        
        # Fit and transform
        df[numeric_cols] = self.scalers[scaling_type].fit_transform(df[numeric_cols])
        
        logger.info(f"Scaled {len(numeric_cols)} numeric columns using {scaling_type} scaling")
        return df
    
    def select_features(self, 
                       df: pd.DataFrame,
                       target_col: str,
                       method: str = "univariate",
                       k: int = 20) -> pd.DataFrame:
        """Select top k features based on statistical tests.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            method: Feature selection method
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return df
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Initialize feature selector
        if method not in self.feature_selectors:
            if method == "univariate":
                self.feature_selectors[method] = SelectKBest(score_func=f_classif, k=k)
            elif method == "mutual_info":
                self.feature_selectors[method] = SelectKBest(score_func=mutual_info_classif, k=k)
        
        # Fit and transform
        X_selected = self.feature_selectors[method].fit_transform(X, y)
        selected_features = X.columns[self.feature_selectors[method].get_support()].tolist()
        
        # Create result DataFrame
        result_df = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
        result_df[target_col] = y
        
        logger.info(f"Selected {len(selected_features)} features using {method} method")
        return result_df
    
    def get_feature_importance_scores(self, method: str = "univariate") -> Dict[str, float]:
        """Get feature importance scores from the last feature selection.
        
        Args:
            method: Feature selection method used
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if method not in self.feature_selectors:
            return {}
        
        selector = self.feature_selectors[method]
        if hasattr(selector, 'scores_'):
            feature_names = selector.feature_names_in_[selector.get_support()]
            scores = selector.scores_[selector.get_support()]
            return dict(zip(feature_names, scores))
        
        return {}
