# Patch for Ollama API response parsing in comprehensive_model_comparison.py
# This script will print the Ollama API response for debugging.
import ollama

result = ollama.list()
print(result)
