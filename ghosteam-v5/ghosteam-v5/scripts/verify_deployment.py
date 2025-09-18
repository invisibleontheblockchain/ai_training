#!/usr/bin/env python3
"""
Railway Deployment Verification Script
Checks if all essential dependencies are available
"""

import sys
import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_import(module_name, description=""):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        logger.info(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        logger.error(f"❌ {module_name} - {description}: {e}")
        return False

def main():
    """Run deployment verification checks."""
    logger.info("🚀 Railway Deployment Verification")
    logger.info("=" * 50)

    checks = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("pydantic", "Data validation"),
        ("redis", "Redis client"),
        ("psycopg2", "PostgreSQL client"),
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("scikit_learn", "Machine learning"),
        ("mlflow", "ML lifecycle management"),
        ("feast", "Feature store (CRITICAL)"),
        ("pyarrow", "Feast dependency"),
        ("protobuf", "Feast dependency"),
    ]

    passed = 0
    failed = 0

    for module, description in checks:
        if check_import(module, description):
            passed += 1
        else:
            failed += 1

    logger.info("=" * 50)
    logger.info(f"📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("🎉 All dependencies available - deployment should succeed!")
        return 0
    else:
        logger.error(f"💥 {failed} dependencies missing - deployment may fail")
        return 1

if __name__ == "__main__":
    sys.exit(main())