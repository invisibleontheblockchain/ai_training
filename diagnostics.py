import os
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join('c:\\AI_Training', 'model_diagnostics.log')
)
logger = logging.getLogger('model_diagnostics')

def check_model_files():
    """Verify all necessary model files exist and have appropriate sizes"""
    model_dir = os.path.join('c:\\AI_Training', 'models')
    required_files = ['config.json', 'weights.bin', 'tokenizer.json']
    
    logger.info(f"Checking model files in {model_dir}")
    issues = []
    
    if not os.path.exists(model_dir):
        return ["Models directory not found"]
    
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            issues.append(f"Missing required file: {file}")
        elif os.path.getsize(file_path) < 1000:  # Arbitrary small size check
            issues.append(f"File may be corrupted (too small): {file}")
    
    return issues

def check_training_logs():
    """Analyze training logs for potential issues"""
    log_dir = os.path.join('c:\\AI_Training', 'logs')
    
    if not os.path.exists(log_dir):
        return ["Training logs directory not found"]
    
    # Find the most recent log file
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    if not log_files:
        return ["No training log files found"]
    
    latest_log = max(log_files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
    logger.info(f"Analyzing most recent training log: {latest_log}")
    
    issues = []
    with open(os.path.join(log_dir, latest_log), 'r') as f:
        content = f.read()
        
        # Check for common training issues
        if "error" in content.lower() or "exception" in content.lower():
            issues.append("Found errors in training logs")
        
        # Check if training completed successfully
        if "training complete" not in content.lower():
            issues.append("Training may not have completed successfully")
        
        # Check for convergence issues
        if "loss: nan" in content.lower() or "loss: inf" in content.lower():
            issues.append("Training encountered NaN or Inf loss values")
    
    return issues

def check_autonomy_settings():
    """Check autonomy-related configuration settings"""
    config_path = os.path.join('c:\\AI_Training', 'models', 'config.json')
    
    if not os.path.exists(config_path):
        return ["Model configuration file not found"]
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        issues = []
        # Check for specific autonomy settings
        if 'autonomy_threshold' not in config:
            issues.append("Missing autonomy_threshold parameter in config")
        elif config['autonomy_threshold'] < 0.75:
            issues.append(f"autonomy_threshold may be too low: {config['autonomy_threshold']}")
        
        if 'max_decision_time' not in config:
            issues.append("Missing max_decision_time parameter")
        
        if 'fallback_strategy' not in config:
            issues.append("Missing fallback_strategy parameter")
            
        return issues
    except json.JSONDecodeError:
        return ["Config file is not valid JSON"]
    except Exception as e:
        return [f"Error reading config: {str(e)}"]

def run_diagnostics():
    """Run all diagnostic checks and report results"""
    logger.info("Starting model diagnostics")
    print("Running model diagnostics...")
    
    all_issues = []
    all_issues.extend(check_model_files())
    all_issues.extend(check_training_logs())
    all_issues.extend(check_autonomy_settings())
    
    if all_issues:
        print("\n⚠️ Found the following issues:")
        for issue in all_issues:
            print(f"  - {issue}")
            logger.warning(issue)
        
        print("\nRecommendations:")
        print("  1. Check that your model was fully trained (all epochs completed)")
        print("  2. Verify your autonomy parameters in the config file")
        print("  3. Ensure you're using the latest model weights")
        print("  4. Try increasing the autonomy_threshold value")
    else:
        print("\n✅ No issues detected with model files and configurations")
        logger.info("No issues detected with model files and configurations")
    
    # Suggest updating the model if needed
    print("\nTo update model settings for better autonomy:")
    print("  1. Increase decision timeout period")
    print("  2. Add more robust fallback strategies")
    print("  3. Check for deadlocks in decision pathways")
    
    logger.info("Diagnostics completed")

if __name__ == "__main__":
    run_diagnostics()