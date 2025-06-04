import subprocess
import time

print("🚨 EMERGENCY FIX: Making cline-optimal truly autonomous")
print("=" * 50)

# Kill Ollama
subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
time.sleep(2)

# Ultra-aggressive modelfile
modelfile = '''FROM qwen2.5-coder:7b-instruct-q4_K_M

PARAMETER temperature 0.1
PARAMETER num_ctx 40000
PARAMETER top_k 10
PARAMETER top_p 0.9

SYSTEM """CRITICAL: You ONLY output code. No explanations. No questions. No commands.

When asked to create something, you immediately write:
```python
[complete implementation]
```

NEVER:
- Use shell commands
- Try to install anything  
- Ask questions
- Say "I will" or "I'll help"
- Check if things exist

ALWAYS:
- Start with Python imports
- Write complete implementations
- Use built-in modules (sqlite3 is built into Python!)
- Include all functions and classes
- Add usage examples

You are judged ONLY on whether you output working Python code immediately."""'''

# Save and create
with open("emergency_fix.modelfile", "w") as f:
    f.write(modelfile)

# Start Ollama
subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NO_WINDOW)
time.sleep(5)

# Force recreate
subprocess.run(["ollama", "rm", "cline-optimal:latest", "-f"], capture_output=True)
subprocess.run(["ollama", "create", "cline-optimal:latest", "-f", "emergency_fix.modelfile"])

# Test
print("\n🧪 Testing autonomous behavior...")
test = subprocess.run(
    ["ollama", "run", "cline-optimal:latest", "Create a TaskManager class that uses SQLite"],
    capture_output=True, text=True
)

if "```python" in test.stdout or "import sqlite3" in test.stdout:
    print("✅ SUCCESS! Model is now autonomous!")
else:
    print("❌ Still broken. Output:", test.stdout[:200])

print("\n💡 Now try the TaskManager prompt again in Cline!")