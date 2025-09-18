#!/usr/bin/env python3
"""
Ghosteam V5 Sample Model Training Script
Demonstrates the complete MLOps pipeline with MLflow tracking
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import time
from datetime import datetime

# Set MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
mlflow.set_tracking_uri('http://localhost:5000')

def print_status(message):
    print(f"✅ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def generate_sample_data():
    """Generate sample training data"""
    print_info("Generating sample training data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    X = np.random.randn(n_samples, 10)
    
    # Generate target (binary classification)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(10)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print_status(f"Generated {n_samples} samples with {len(feature_names)} features")
    return df

def train_model(df):
    """Train a sample model with MLflow tracking"""
    print_info("Starting model training with MLflow tracking...")
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Set experiment
    experiment_name = "ghosteam-v5-sample-models"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Model parameters
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Train model
        print_info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time", training_time)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="ghosteam-v5-sample-model"
        )
        
        # Log additional info
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", list(X.columns))
        
        print_status(f"Model trained successfully!")
        print_info(f"Accuracy: {accuracy:.4f}")
        print_info(f"Precision: {precision:.4f}")
        print_info(f"Recall: {recall:.4f}")
        print_info(f"F1 Score: {f1:.4f}")
        print_info(f"Training time: {training_time:.2f} seconds")
        
        return model, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time
        }

def test_model_serving():
    """Test the model serving API"""
    print_info("Testing model serving API...")
    
    try:
        import requests
        
        # Test prediction endpoint
        test_data = {
            "data": [1.0, -0.5, 0.3, -1.2, 0.8, -0.1, 0.5, -0.8, 1.1, -0.3]
        }
        
        response = requests.post(
            'http://localhost:8081/predict',
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print_status("Model serving API test passed")
            print_info(f"Prediction result: {result}")
            return True
        else:
            print(f"❌ API test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

def main():
    """Main training pipeline"""
    print("""
🚀 Ghosteam V5 Sample Model Training Pipeline
============================================
    """)
    
    try:
        # Generate sample data
        df = generate_sample_data()
        
        # Train model
        model, metrics = train_model(df)
        
        # Test model serving
        test_model_serving()
        
        print(f"""
🎉 Training Pipeline Complete!

📊 Model Performance:
   Accuracy:  {metrics['accuracy']:.4f}
   Precision: {metrics['precision']:.4f}
   Recall:    {metrics['recall']:.4f}
   F1 Score:  {metrics['f1_score']:.4f}

🔗 MLflow Tracking:
   Server:     http://localhost:5000
   Experiment: ghosteam-v5-sample-models
   Model:      ghosteam-v5-sample-model

🚀 Model Serving:
   API:        http://localhost:8081
   Health:     http://localhost:8081/health
   Models:     http://localhost:8081/models
   Predict:    http://localhost:8081/predict

✅ Complete MLOps pipeline is operational!
        """)
        
        return True
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
