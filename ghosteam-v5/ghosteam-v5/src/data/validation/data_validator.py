import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import great_expectations as ge
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Comprehensive data quality validation using Great Expectations and Pandera."""
    
    def __init__(self):
        self.context = ge.get_context()
        self.validation_results = []
    
    def create_user_features_schema(self) -> DataFrameSchema:
        """Create Pandera schema for user features."""
        return DataFrameSchema({
            "user_id": Column(pa.String, checks=[
                Check.str_length(min_value=1, max_value=50),
                Check(lambda x: x.str.match(r'^[a-zA-Z0-9_-]+$').all(), 
                      error="Invalid user_id format")
            ]),
            "age": Column(pa.Int, checks=[
                Check.in_range(min_value=13, max_value=120),
                Check(lambda x: x.notna().all(), error="Age cannot be null")
            ]),
            "activity_score": Column(pa.Float, checks=[
                Check.in_range(min_value=0.0, max_value=1.0),
                Check(lambda x: x.notna().all(), error="Activity score cannot be null")
            ]),
            "engagement_rate": Column(pa.Float, checks=[
                Check.in_range(min_value=0.0, max_value=1.0)
            ]),
            "total_sessions": Column(pa.Int, checks=[
                Check.greater_than_or_equal_to(0)
            ]),
            "avg_session_duration": Column(pa.Float, checks=[
                Check.greater_than_or_equal_to(0.0)
            ]),
            "is_premium": Column(pa.Bool),
            "days_since_signup": Column(pa.Int, checks=[
                Check.greater_than_or_equal_to(0)
            ]),
            "event_timestamp": Column(pa.DateTime)
        })
    
    def create_model_performance_schema(self) -> DataFrameSchema:
        """Create Pandera schema for model performance data."""
        return DataFrameSchema({
            "model_id": Column(pa.String, checks=[
                Check.str_length(min_value=1, max_value=100)
            ]),
            "accuracy": Column(pa.Float, checks=[
                Check.in_range(min_value=0.0, max_value=1.0)
            ]),
            "precision": Column(pa.Float, checks=[
                Check.in_range(min_value=0.0, max_value=1.0)
            ]),
            "recall": Column(pa.Float, checks=[
                Check.in_range(min_value=0.0, max_value=1.0)
            ]),
            "f1_score": Column(pa.Float, checks=[
                Check.in_range(min_value=0.0, max_value=1.0)
            ]),
            "latency_p95": Column(pa.Float, checks=[
                Check.greater_than_or_equal_to(0.0)
            ]),
            "prediction_count": Column(pa.Int, checks=[
                Check.greater_than_or_equal_to(0)
            ]),
            "error_rate": Column(pa.Float, checks=[
                Check.in_range(min_value=0.0, max_value=1.0)
            ]),
            "drift_score": Column(pa.Float, checks=[
                Check.greater_than_or_equal_to(0.0)
            ]),
            "event_timestamp": Column(pa.DateTime)
        })
    
    def validate_dataframe(self, 
                          df: pd.DataFrame, 
                          schema: DataFrameSchema,
                          data_type: str) -> Dict[str, Any]:
        """Validate DataFrame against schema.
        
        Args:
            df: DataFrame to validate
            schema: Pandera schema
            data_type: Type of data being validated
            
        Returns:
            Validation results dictionary
        """
        validation_result = {
            "data_type": data_type,
            "timestamp": datetime.now(),
            "total_rows": len(df),
            "passed": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Validate with Pandera
            validated_df = schema.validate(df, lazy=True)
            validation_result["passed"] = True
            logger.info(f"Data validation passed for {data_type}")
            
        except pa.errors.SchemaErrors as e:
            validation_result["errors"] = [str(error) for error in e.failure_cases]
            logger.error(f"Data validation failed for {data_type}: {e}")
            
        except Exception as e:
            validation_result["errors"] = [str(e)]
            logger.error(f"Unexpected error during validation: {e}")
        
        # Additional custom validations
        custom_checks = self._run_custom_checks(df, data_type)
        validation_result["warnings"].extend(custom_checks.get("warnings", []))
        
        self.validation_results.append(validation_result)
        return validation_result
    
    def _run_custom_checks(self, df: pd.DataFrame, data_type: str) -> Dict[str, List[str]]:
        """Run custom data quality checks.
        
        Args:
            df: DataFrame to check
            data_type: Type of data
            
        Returns:
            Dictionary with warnings and errors
        """
        warnings = []
        errors = []
        
        # Check for duplicates
        if df.duplicated().any():
            duplicate_count = df.duplicated().sum()
            warnings.append(f"Found {duplicate_count} duplicate rows")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                missing_pct = (count / len(df)) * 100
                if missing_pct > 50:
                    errors.append(f"Column {col} has {missing_pct:.1f}% missing values")
                elif missing_pct > 10:
                    warnings.append(f"Column {col} has {missing_pct:.1f}% missing values")
        
        # Check data freshness
        if "event_timestamp" in df.columns:
            latest_timestamp = df["event_timestamp"].max()
            hours_old = (datetime.now() - latest_timestamp).total_seconds() / 3600
            
            if hours_old > 24:
                warnings.append(f"Data is {hours_old:.1f} hours old")
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    outlier_pct = (len(outliers) / len(df)) * 100
                    if outlier_pct > 5:
                        warnings.append(f"Column {col} has {outlier_pct:.1f}% outliers")
        
        return {"warnings": warnings, "errors": errors}
    
    def create_great_expectations_suite(self, 
                                      df: pd.DataFrame, 
                                      suite_name: str) -> ExpectationSuite:
        """Create Great Expectations suite from DataFrame.
        
        Args:
            df: DataFrame to analyze
            suite_name: Name for the expectation suite
            
        Returns:
            ExpectationSuite object
        """
        # Convert to Great Expectations dataset
        ge_df = PandasDataset(df)
        
        # Create expectation suite
        suite = ge_df.get_expectation_suite()
        suite.expectation_suite_name = suite_name
        
        # Add basic expectations
        for col in df.columns:
            # Expect column to exist
            ge_df.expect_column_to_exist(col)
            
            # For numeric columns
            if df[col].dtype in ['int64', 'float64']:
                # Expect values to be not null (if mostly not null)
                null_pct = df[col].isnull().mean()
                if null_pct < 0.1:
                    ge_df.expect_column_values_to_not_be_null(col)
                
                # Expect values to be between min and max
                min_val = df[col].min()
                max_val = df[col].max()
                ge_df.expect_column_values_to_be_between(
                    col, min_value=min_val, max_value=max_val
                )
            
            # For string columns
            elif df[col].dtype == 'object':
                # Expect values to match regex pattern (if applicable)
                if col.endswith('_id'):
                    ge_df.expect_column_values_to_match_regex(
                        col, regex=r'^[a-zA-Z0-9_-]+$'
                    )
        
        return suite
    
    def validate_with_great_expectations(self, 
                                       df: pd.DataFrame,
                                       suite: ExpectationSuite) -> Dict[str, Any]:
        """Validate DataFrame with Great Expectations suite.
        
        Args:
            df: DataFrame to validate
            suite: ExpectationSuite to use
            
        Returns:
            Validation results
        """
        ge_df = PandasDataset(df)
        results = ge_df.validate(expectation_suite=suite)
        
        return {
            "success": results.success,
            "statistics": results.statistics,
            "results": [
                {
                    "expectation_type": result.expectation_config.expectation_type,
                    "success": result.success,
                    "result": result.result
                }
                for result in results.results
            ]
        }
    
    def generate_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile.
        
        Args:
            df: DataFrame to profile
            
        Returns:
            Data profile dictionary
        """
        profile = {
            "timestamp": datetime.now(),
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "columns": {}
        }
        
        for col in df.columns:
            col_profile = {
                "dtype": str(df[col].dtype),
                "null_count": df[col].isnull().sum(),
                "null_percentage": df[col].isnull().mean() * 100,
                "unique_count": df[col].nunique(),
                "unique_percentage": (df[col].nunique() / len(df)) * 100
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_profile.update({
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "q25": df[col].quantile(0.25),
                    "q50": df[col].quantile(0.50),
                    "q75": df[col].quantile(0.75)
                })
            
            elif df[col].dtype == 'object':
                col_profile.update({
                    "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    "avg_length": df[col].astype(str).str.len().mean()
                })
            
            profile["columns"][col] = col_profile
        
        return profile
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results.
        
        Returns:
            Summary of validation results
        """
        if not self.validation_results:
            return {"message": "No validation results available"}
        
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for r in self.validation_results if r["passed"])
        
        return {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "success_rate": (passed_validations / total_validations) * 100,
            "latest_validation": self.validation_results[-1],
            "all_results": self.validation_results
        }
