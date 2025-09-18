#!/bin/bash

# Setup Sample Data for Testing
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_header "🗃️ Setting up Sample Data for Testing"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create sample data generation script
cat > scripts/generate_sample_data.py << 'EOF'
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_user_features(n_samples=5000):
    """Generate sample user features."""
    print("Generating user features...")
    
    # Generate user data
    user_ids = [f'user_{i:06d}' for i in range(n_samples)]
    
    data = {
        'user_id': user_ids,
        'age': np.random.normal(35, 12, n_samples).astype(int).clip(18, 80),
        'activity_score': np.random.beta(2, 5, n_samples),
        'engagement_rate': np.random.beta(3, 7, n_samples),
        'total_sessions': np.random.poisson(20, n_samples),
        'avg_session_duration': np.random.exponential(300, n_samples),
        'is_premium': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        'days_since_signup': np.random.exponential(100, n_samples).astype(int),
        'last_activity_score': np.random.beta(2, 5, n_samples),
        'event_timestamp': [
            datetime.now() - timedelta(hours=np.random.randint(0, 24*30))
            for _ in range(n_samples)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlations to make it realistic
    # Premium users tend to have higher engagement
    premium_mask = df['is_premium']
    df.loc[premium_mask, 'engagement_rate'] *= 1.5
    df.loc[premium_mask, 'activity_score'] *= 1.3
    
    # Older users tend to have longer sessions
    df['avg_session_duration'] += (df['age'] - 30) * 10
    
    # Clip values to reasonable ranges
    df['engagement_rate'] = df['engagement_rate'].clip(0, 1)
    df['activity_score'] = df['activity_score'].clip(0, 1)
    df['avg_session_duration'] = df['avg_session_duration'].clip(10, 7200)
    
    return df

def generate_session_features(n_samples=10000):
    """Generate sample session features."""
    print("Generating session features...")
    
    session_ids = [f'session_{i:08d}' for i in range(n_samples)]
    
    data = {
        'session_id': session_ids,
        'session_duration': np.random.exponential(400, n_samples),
        'page_views': np.random.poisson(8, n_samples),
        'clicks': np.random.poisson(12, n_samples),
        'bounce_rate': np.random.beta(2, 8, n_samples),
        'conversion_rate': np.random.beta(1, 20, n_samples),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples, p=[0.6, 0.3, 0.1]),
        'browser': np.random.choice(['chrome', 'firefox', 'safari', 'edge'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'referrer_type': np.random.choice(['direct', 'search', 'social', 'email'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'event_timestamp': [
            datetime.now() - timedelta(hours=np.random.randint(0, 24*7))
            for _ in range(n_samples)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Add correlations
    # Mobile users tend to have shorter sessions
    mobile_mask = df['device_type'] == 'mobile'
    df.loc[mobile_mask, 'session_duration'] *= 0.7
    df.loc[mobile_mask, 'page_views'] *= 0.8
    
    # Social referrers tend to have higher bounce rates
    social_mask = df['referrer_type'] == 'social'
    df.loc[social_mask, 'bounce_rate'] *= 1.3
    
    # Clip values
    df['bounce_rate'] = df['bounce_rate'].clip(0, 1)
    df['conversion_rate'] = df['conversion_rate'].clip(0, 1)
    df['session_duration'] = df['session_duration'].clip(10, 7200)
    
    return df

def generate_training_data(user_df, n_samples=3000):
    """Generate training data with target variable."""
    print("Generating training data...")
    
    # Sample users for training
    sample_users = user_df.sample(n_samples)
    
    # Create target variable based on user features
    # Higher engagement and activity should predict positive outcome
    target_prob = (
        0.3 +  # Base probability
        0.3 * sample_users['engagement_rate'] +
        0.2 * sample_users['activity_score'] +
        0.1 * sample_users['is_premium'].astype(float) +
        0.1 * np.random.random(len(sample_users))  # Random noise
    )
    
    target = np.random.binomial(1, target_prob.clip(0, 1))
    
    training_data = sample_users.copy()
    training_data['target'] = target
    
    return training_data

def generate_drift_data(original_df, drift_type='mean_shift', drift_strength=0.5):
    """Generate data with artificial drift for testing."""
    print(f"Generating drift data with {drift_type}...")
    
    drift_df = original_df.copy()
    
    if drift_type == 'mean_shift':
        # Shift the mean of numeric features
        numeric_cols = ['age', 'activity_score', 'engagement_rate', 'avg_session_duration']
        for col in numeric_cols:
            if col in drift_df.columns:
                shift = drift_df[col].std() * drift_strength
                drift_df[col] += shift
    
    elif drift_type == 'variance_shift':
        # Change the variance of numeric features
        numeric_cols = ['age', 'activity_score', 'engagement_rate']
        for col in numeric_cols:
            if col in drift_df.columns:
                mean_val = drift_df[col].mean()
                drift_df[col] = mean_val + (drift_df[col] - mean_val) * (1 + drift_strength)
    
    elif drift_type == 'categorical_shift':
        # Change categorical distributions
        if 'device_type' in drift_df.columns:
            # Shift towards more mobile usage
            mobile_indices = drift_df['device_type'] == 'desktop'
            shift_indices = np.random.choice(
                drift_df[mobile_indices].index, 
                size=int(len(drift_df[mobile_indices]) * drift_strength),
                replace=False
            )
            drift_df.loc[shift_indices, 'device_type'] = 'mobile'
    
    # Update timestamps to be recent
    drift_df['event_timestamp'] = [
        datetime.now() - timedelta(hours=np.random.randint(0, 24))
        for _ in range(len(drift_df))
    ]
    
    return drift_df

def main():
    """Generate all sample data."""
    print("Starting sample data generation...")
    
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    
    # Generate datasets
    user_features = generate_user_features(5000)
    session_features = generate_session_features(10000)
    training_data = generate_training_data(user_features, 3000)
    
    # Generate drift datasets
    drift_data_mean = generate_drift_data(user_features.sample(1000), 'mean_shift', 0.8)
    drift_data_variance = generate_drift_data(user_features.sample(1000), 'variance_shift', 0.6)
    drift_data_categorical = generate_drift_data(session_features.sample(1000), 'categorical_shift', 0.4)
    
    # Save datasets
    print("Saving datasets...")
    
    # Raw data
    user_features.to_parquet('data/raw/user_features.parquet', index=False)
    session_features.to_parquet('data/raw/session_features.parquet', index=False)
    
    # Features for Feast
    user_features.to_parquet('data/features/user_features.parquet', index=False)
    session_features.to_parquet('data/features/session_features.parquet', index=False)
    
    # Training data
    training_data.to_parquet('data/processed/training_data.parquet', index=False)
    
    # Drift test data
    drift_data_mean.to_parquet('data/processed/drift_test_mean.parquet', index=False)
    drift_data_variance.to_parquet('data/processed/drift_test_variance.parquet', index=False)
    drift_data_categorical.to_parquet('data/processed/drift_test_categorical.parquet', index=False)
    
    # Create reference data for drift detection
    reference_data = user_features.sample(2000)
    reference_data.to_parquet('data/processed/reference_data.parquet', index=False)
    
    print("✅ Sample data generation completed!")
    print(f"Generated {len(user_features)} user records")
    print(f"Generated {len(session_features)} session records")
    print(f"Generated {len(training_data)} training samples")
    print(f"Generated drift test datasets")
    
    # Print data summary
    print("\nData Summary:")
    print("User Features:")
    print(user_features.describe())
    print("\nTraining Data Target Distribution:")
    print(training_data['target'].value_counts())

if __name__ == "__main__":
    main()
EOF

# Run the data generation script
print_status "Generating sample data..."
python scripts/generate_sample_data.py

# Initialize Feast feature store
print_status "Initializing Feast feature store..."
cd feast_repo

# Create feature_store.yaml if it doesn't exist
if [ ! -f "feature_store.yaml" ]; then
    cat > feature_store.yaml << 'EOF'
project: ghosteam_v5
registry: feast_registry.db
provider: local
online_store:
    type: redis
    connection_string: localhost:6379
offline_store:
    type: file
EOF
fi

# Copy feature definitions
cp ../src/data/feature_store/feast_config.py .

# Apply feature definitions
print_status "Applying Feast feature definitions..."
feast apply

# Materialize features to online store
print_status "Materializing features to online store..."
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

cd ..

print_header "✅ Sample Data Setup Complete!"

echo ""
echo "Generated Data:"
echo "📊 User features: 5,000 records"
echo "📱 Session features: 10,000 records"
echo "🎯 Training data: 3,000 samples"
echo "🔄 Drift test data: Multiple datasets"
echo ""

echo "Feast Feature Store:"
echo "✅ Feature definitions applied"
echo "✅ Features materialized to online store"
echo ""

echo "Next: Run continuous learning tests with ./scripts/test_continuous_learning.sh"
