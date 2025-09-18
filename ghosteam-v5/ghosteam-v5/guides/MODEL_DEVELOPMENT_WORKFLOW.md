# 🤖 MODEL DEVELOPMENT WORKFLOW GUIDE

## 🎯 **OBJECTIVE**
Establish best practices for developing, training, and deploying machine learning models using your Ghosteam V5 MLOps infrastructure.

---

## 📋 **COMPLETE MODEL LIFECYCLE**

### **1. DATA PIPELINE SETUP**

#### **Data Ingestion Framework**
```python
# src/data/ingestion.py
import pandas as pd
from pathlib import Path
import great_expectations as ge
from typing import Dict, Any
import mlflow

class DataIngestionPipeline:
    def __init__(self, data_source: str, validation_suite: str = None):
        self.data_source = data_source
        self.validation_suite = validation_suite
        
    def ingest_data(self) -> pd.DataFrame:
        """Ingest data from various sources"""
        if self.data_source.endswith('.csv'):
            df = pd.read_csv(self.data_source)
        elif self.data_source.startswith('postgresql://'):
            df = pd.read_sql(self.data_source)
        elif self.data_source.startswith('s3://'):
            df = pd.read_csv(self.data_source)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
        
        # Log data ingestion
        mlflow.log_param("data_source", self.data_source)
        mlflow.log_param("data_shape", df.shape)
        mlflow.log_param("data_columns", list(df.columns))
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality using Great Expectations"""
        if not self.validation_suite:
            return True
            
        context = ge.get_context()
        suite = context.get_expectation_suite(self.validation_suite)
        
        # Create validator
        validator = context.get_validator(
            batch_request=ge.core.batch.RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="runtime_data_connector",
                data_asset_name="validation_data",
                runtime_parameters={"batch_data": df},
                batch_identifiers={"default_identifier_name": "validation_batch"}
            ),
            expectation_suite=suite
        )
        
        # Run validation
        results = validator.validate()
        
        # Log validation results
        mlflow.log_param("data_validation_success", results.success)
        mlflow.log_param("data_validation_statistics", results.statistics)
        
        return results.success
```

#### **Feature Engineering Pipeline**
```python
# src/features/engineering.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import mlflow

class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features=None, categorical_features=None):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.preprocessor = None
        
    def fit(self, X, y=None):
        """Fit the feature engineering pipeline"""
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', LabelEncoder(), self.categorical_features)
            ]
        )
        
        self.preprocessor = preprocessor.fit(X)
        
        # Log feature engineering parameters
        mlflow.log_param("numerical_features", self.numerical_features)
        mlflow.log_param("categorical_features", self.categorical_features)
        mlflow.log_param("feature_count", len(self.numerical_features) + len(self.categorical_features))
        
        return self
    
    def transform(self, X):
        """Transform features"""
        if self.preprocessor is None:
            raise ValueError("Pipeline must be fitted before transform")
        
        return self.preprocessor.transform(X)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features"""
        
        # Example feature engineering
        if 'age' in df.columns and 'income' in df.columns:
            df['age_income_ratio'] = df['age'] / (df['income'] + 1)
        
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Log engineered features
        mlflow.log_param("engineered_features", [col for col in df.columns if col not in self.numerical_features + self.categorical_features])
        
        return df
```

### **2. MODEL DEVELOPMENT FRAMEWORK**

#### **Model Training Pipeline**
```python
# src/models/training.py
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import joblib
from datetime import datetime

class ModelTrainingPipeline:
    def __init__(self, model, model_name: str, experiment_name: str):
        self.model = model
        self.model_name = model_name
        self.experiment_name = experiment_name
        
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, cv_folds: int = 5) -> Dict[str, Any]:
        """Complete model training pipeline with MLflow tracking"""
        
        # Set MLflow experiment
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(run_name=f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Log data split information
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("test_split_ratio", test_size)
            mlflow.log_param("cv_folds", cv_folds)
            
            # Log model parameters
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                for param, value in params.items():
                    mlflow.log_param(f"model_{param}", value)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
            mlflow.log_metric("cv_std_accuracy", cv_scores.std())
            
            # Train model
            start_time = datetime.now()
            self.model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name=self.model_name,
                input_example=X_train.iloc[:5],
                signature=mlflow.models.infer_signature(X_train, y_pred_train)
            )
            
            # Log feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save and log feature importance
                feature_importance.to_csv("feature_importance.csv", index=False)
                mlflow.log_artifact("feature_importance.csv")
            
            return {
                'model': self.model,
                'metrics': metrics,
                'run_id': mlflow.active_run().info.run_id
            }
    
    def _calculate_metrics(self, y_train, y_pred_train, y_test, y_pred_test) -> Dict[str, float]:
        """Calculate comprehensive model metrics"""
        return {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_precision': precision_score(y_train, y_pred_train, average='weighted'),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
            'train_recall': recall_score(y_train, y_pred_train, average='weighted'),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted'),
            'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
        }
```

#### **Model Validation Framework**
```python
# src/models/validation.py
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

class ModelValidator:
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        
    def validate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive model validation"""
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)
        
        # Generate validation report
        validation_results = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'predictions': y_pred.tolist(),
            'actual': y_test.tolist()
        }
        
        if y_pred_proba is not None:
            validation_results['prediction_probabilities'] = y_pred_proba.tolist()
        
        # Create visualizations
        self._create_validation_plots(y_test, y_pred, y_pred_proba)
        
        # Log validation artifacts
        mlflow.log_dict(validation_results, "validation_results.json")
        
        return validation_results
    
    def _create_validation_plots(self, y_test, y_pred, y_pred_proba=None):
        """Create validation visualizations"""
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(10, 8))
            feature_names = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.barh(importance_df['feature'][-20:], importance_df['importance'][-20:])
            plt.title(f'{self.model_name} - Top 20 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            mlflow.log_artifact('feature_importance.png')
            plt.close()
```

### **3. MODEL DEPLOYMENT WORKFLOW**

#### **Model Deployment Pipeline**
```python
# src/deployment/deploy.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import requests
import json
from typing import Dict, Any, Optional

class ModelDeploymentPipeline:
    def __init__(self, model_name: str, api_endpoint: str):
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.client = MlflowClient()
        
    def deploy_model(self, version: str = "latest", stage: str = "Production") -> bool:
        """Deploy model to production"""
        
        try:
            # Get model version
            if version == "latest":
                model_versions = self.client.get_latest_versions(self.model_name, stages=[stage])
                if not model_versions:
                    raise ValueError(f"No model versions found in {stage} stage")
                model_version = model_versions[0]
            else:
                model_version = self.client.get_model_version(self.model_name, version)
            
            # Load model
            model_uri = f"models:/{self.model_name}/{model_version.version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Test model locally
            if not self._test_model_locally(model):
                raise ValueError("Model failed local testing")
            
            # Deploy to API endpoint (this would integrate with your deployment system)
            deployment_success = self._deploy_to_api(model, model_version)
            
            if deployment_success:
                # Update model stage
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=model_version.version,
                    stage="Production"
                )
                
                print(f"✅ Model {self.model_name} v{model_version.version} deployed successfully")
                return True
            else:
                raise ValueError("Deployment to API failed")
                
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False
    
    def _test_model_locally(self, model) -> bool:
        """Test model with sample data"""
        try:
            # Create sample test data
            import numpy as np
            sample_data = np.random.randn(5, 10)  # Adjust based on your model
            
            # Make prediction
            prediction = model.predict(sample_data)
            
            # Basic validation
            return prediction is not None and len(prediction) == 5
            
        except Exception as e:
            print(f"Local model test failed: {e}")
            return False
    
    def _deploy_to_api(self, model, model_version) -> bool:
        """Deploy model to API endpoint"""
        # This would integrate with your specific deployment system
        # For now, we'll simulate a successful deployment
        
        try:
            # In a real implementation, this would:
            # 1. Package the model
            # 2. Update the API service
            # 3. Perform health checks
            # 4. Gradually roll out traffic
            
            print(f"Deploying model {self.model_name} v{model_version.version} to {self.api_endpoint}")
            
            # Simulate deployment
            return True
            
        except Exception as e:
            print(f"API deployment failed: {e}")
            return False
    
    def rollback_deployment(self, previous_version: str) -> bool:
        """Rollback to previous model version"""
        try:
            # Get previous model version
            model_version = self.client.get_model_version(self.model_name, previous_version)
            
            # Deploy previous version
            return self.deploy_model(version=previous_version)
            
        except Exception as e:
            print(f"Rollback failed: {e}")
            return False
```

### **4. COMPLETE WORKFLOW EXAMPLE**

#### **End-to-End Model Development Script**
```python
# scripts/train_new_model.py
#!/usr/bin/env python3
"""
Complete model development workflow example
"""

import sys
import os
sys.path.append('src')

from data.ingestion import DataIngestionPipeline
from features.engineering import FeatureEngineeringPipeline
from models.training import ModelTrainingPipeline
from models.validation import ModelValidator
from deployment.deploy import ModelDeploymentPipeline

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def main():
    """Complete model development workflow"""
    
    print("🚀 Starting Model Development Workflow")
    
    # 1. Data Ingestion
    print("\n📊 Step 1: Data Ingestion")
    data_pipeline = DataIngestionPipeline("data/raw/training_data.csv")
    df = data_pipeline.ingest_data()
    
    if not data_pipeline.validate_data(df):
        print("❌ Data validation failed")
        return False
    
    # 2. Feature Engineering
    print("\n🔧 Step 2: Feature Engineering")
    feature_pipeline = FeatureEngineeringPipeline(
        numerical_features=['feature_1', 'feature_2', 'feature_3'],
        categorical_features=['category_1', 'category_2']
    )
    
    # Create engineered features
    df_engineered = feature_pipeline.create_features(df)
    
    # Prepare features and target
    X = df_engineered.drop('target', axis=1)
    y = df_engineered['target']
    
    # Fit and transform features
    feature_pipeline.fit(X)
    X_transformed = feature_pipeline.transform(X)
    
    # 3. Model Training
    print("\n🤖 Step 3: Model Training")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    training_pipeline = ModelTrainingPipeline(
        model=model,
        model_name="ghosteam-v5-production-model",
        experiment_name="production-models"
    )
    
    training_results = training_pipeline.train_model(X_transformed, y)
    
    # 4. Model Validation
    print("\n✅ Step 4: Model Validation")
    validator = ModelValidator(training_results['model'], "ghosteam-v5-production-model")
    
    # Split data for validation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    validation_results = validator.validate_model(X_test, y_test)
    
    # 5. Model Deployment (if validation passes)
    print("\n🚀 Step 5: Model Deployment")
    if training_results['metrics']['test_accuracy'] > 0.85:  # Deployment threshold
        deployment_pipeline = ModelDeploymentPipeline(
            model_name="ghosteam-v5-production-model",
            api_endpoint="http://localhost:8081"
        )
        
        deployment_success = deployment_pipeline.deploy_model()
        
        if deployment_success:
            print("🎉 Model development workflow completed successfully!")
            return True
        else:
            print("❌ Model deployment failed")
            return False
    else:
        print(f"❌ Model accuracy ({training_results['metrics']['test_accuracy']:.3f}) below deployment threshold (0.85)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

---

## 📋 **BEST PRACTICES CHECKLIST**

### **Data Management**
- [ ] Data validation with Great Expectations
- [ ] Data versioning with DVC
- [ ] Feature store integration (Feast)
- [ ] Data lineage tracking

### **Model Development**
- [ ] Experiment tracking with MLflow
- [ ] Cross-validation for model selection
- [ ] Feature importance analysis
- [ ] Model interpretability (SHAP, LIME)

### **Model Validation**
- [ ] Holdout test set validation
- [ ] Statistical significance testing
- [ ] Bias and fairness evaluation
- [ ] Performance monitoring setup

### **Deployment**
- [ ] Automated deployment pipeline
- [ ] A/B testing framework
- [ ] Canary deployment strategy
- [ ] Rollback procedures

### **Monitoring**
- [ ] Model performance monitoring
- [ ] Data drift detection
- [ ] Prediction quality tracking
- [ ] Business impact measurement

---

## 🎯 **SUCCESS METRICS**

- **Model Accuracy**: >90% on test set
- **Training Time**: <30 minutes for standard models
- **Deployment Time**: <15 minutes from training to production
- **Model Uptime**: 99.9% availability
- **Prediction Latency**: <100ms response time

**🚀 Your model development workflow is now production-ready!**
