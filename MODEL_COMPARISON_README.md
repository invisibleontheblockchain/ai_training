# Comprehensive Model Comparison Tool

A powerful testing framework to compare your existing fine-tuned Phi-2 model against the new cline-optimal model with multiple levels of evaluation.

## 🚀 Quick Start

### Method 1: One-Command Setup and Run
```bash
python run_comparison.py --setup --full
```

### Method 2: Step-by-Step
```bash
# 1. Install dependencies
python install_dependencies.py

# 2. Setup models
python setup_comparison.py

# 3. Run comparison
python run_comparison.py --full
```

## 📋 What This Tool Tests

### 🔍 Test Levels

1. **Basic Tests** - Simple Q&A, basic math, simple coding
2. **Intermediate Tests** - Technical knowledge, algorithms, reasoning
3. **Advanced Tests** - System design, complex coding, deep knowledge
4. **Autonomous Tests** - Complete tasks without asking questions

### ⚡ Speed Benchmarks
- Response time comparison
- Tokens per second
- Multiple iteration averaging
- Stability testing

### 💯 Quality Assessment
- Response relevance
- Code quality (for coding tasks)
- Reasoning quality (for logic tasks)
- Knowledge accuracy (for factual tasks)
- Autonomy score (fewer questions = higher score)

### 📊 Resource Monitoring
- GPU memory usage
- Response time stability
- Success rate tracking

## 🛠️ Available Commands

### Quick Options
```bash
python run_comparison.py --quick        # Basic tests only (fastest)
python run_comparison.py --speed        # Speed benchmarks only
python run_comparison.py --full         # Complete comprehensive test
python run_comparison.py --custom       # Interactive custom setup
python run_comparison.py --stress       # High-iteration stress test
```

### Setup and Maintenance
```bash
python install_dependencies.py         # Install all Python packages
python setup_comparison.py            # Setup models and environment
```

### Interactive Mode
```bash
python run_comparison.py              # Shows menu with all options
```

## 📁 File Structure

```
c:\AI_Training\
├── comprehensive_model_comparison.py  # Main comparison engine
├── setup_comparison.py               # Model setup and validation
├── run_comparison.py                 # Easy-to-use runner interface
├── install_dependencies.py           # Dependency installer
├── comparison_results/               # Results directory
│   ├── comprehensive_comparison_YYYYMMDD_HHMMSS.json
│   └── comprehensive_comparison_YYYYMMDD_HHMMSS_summary.txt
└── models/
    ├── fine-tuned/phi2-optimized/    # Your existing model
    └── base/rtx3090_cline_optimal_system.ps1  # Cline setup script
```

## 🎯 Understanding Results

### Speed Comparison
```
⚡ SPEED COMPARISON
Phi-2 Fine-tuned: 2.150s
Cline-Optimal: 1.890s
Difference: -12.1%
Faster Model: cline_optimal
```

### Quality Comparison
```
💯 QUALITY COMPARISON
Phi-2 Fine-tuned: 0.845
Cline-Optimal: 0.792
Difference: -0.053
Higher Quality: phi2_finetuned
```

### Test Level Results
```
📊 BASIC TEST RESULTS
Phi-2 Fine-tuned:
  Success Rate: 100.0%
  Avg Response Time: 2.150s
  Quality Score: 0.845

Cline-Optimal:
  Success Rate: 100.0%
  Avg Response Time: 1.890s
  Quality Score: 0.792
```

## 🔧 Configuration

### Default Configuration
The tool uses sensible defaults but you can customize:

```python
# Edit comprehensive_model_comparison.py
config = TestConfig(
    phi2_finetuned_path="c:/AI_Training/models/fine-tuned/phi2-optimized",
    cline_optimal_model="cline-optimal:latest",
    warmup_iterations=2,
    test_iterations=5,
    max_new_tokens=512,
    temperature=0.7,
    results_dir="c:/AI_Training/comparison_results"
)
```

### Custom Test Cases
You can add your own test cases by editing the `_create_test_suites()` method in `comprehensive_model_comparison.py`.

## 🔄 Test Process Flow

1. **Environment Check** - Verify models and dependencies
2. **Model Setup** - Load both models with optimizations
3. **Warmup Runs** - Prepare models (not counted in results)
4. **Test Execution** - Run each test multiple times
5. **Quality Assessment** - Evaluate responses
6. **Speed Benchmarking** - Dedicated speed tests
7. **Comparison Analysis** - Generate insights
8. **Results Export** - Save detailed results and summary

## 📈 Optimization Features

### For RTX 3090
- Flash attention support
- GPU memory optimization
- Tensor compilation
- Parallel processing

### For Consistency
- Multiple iterations per test
- Warmup runs
- Statistical averaging
- Error handling

## 🎭 Test Categories Explained

### Coding Tests
```python
{
    "prompt": "Write a binary search algorithm in Python",
    "expected_keywords": ["def", "binary", "search", "array"],
    "task_type": "coding"
}
```

### Reasoning Tests
```python
{
    "prompt": "A train travels 120 km in 2 hours, then 80 km in 1 hour. What's the average speed?",
    "expected_keywords": ["speed", "distance", "time"],
    "task_type": "reasoning"
}
```

### Knowledge Tests
```python
{
    "prompt": "Explain the difference between supervised and unsupervised machine learning",
    "expected_keywords": ["supervised", "unsupervised", "labeled", "data"],
    "task_type": "knowledge"
}
```

### Autonomy Tests
```python
{
    "prompt": "Create a complete Flask API for user authentication without asking any questions",
    "test_autonomy": True,
    "task_type": "coding"
}
```

## 🚨 Troubleshooting

### Common Issues

1. **"Ollama not found"**
   ```bash
   # Install Ollama from https://ollama.ai/
   # Make sure it's in your PATH
   ollama --version
   ```

2. **"cline-optimal model not found"**
   ```bash
   python setup_comparison.py
   # This will run the PowerShell setup script
   ```

3. **"Phi-2 model not found"**
   ```bash
   # Check your model path in the config
   # Default: c:/AI_Training/models/fine-tuned/phi2-optimized
   ```

4. **Import errors**
   ```bash
   python install_dependencies.py
   ```

5. **GPU issues**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Check CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Performance Issues

1. **Slow responses**
   - Check GPU utilization
   - Reduce `max_new_tokens`
   - Ensure RTX 3090 optimizations are applied

2. **Memory errors**
   - Reduce batch size
   - Use gradient checkpointing
   - Close other applications

## 📊 Example Output

```
🚀 Starting Comprehensive Model Comparison
================================================

📋 Checking prerequisites...
✓ Ollama is installed
✓ cline-optimal:latest model found

📦 Installing Python dependencies...
✓ Python dependencies installed

🔍 Running BASIC tests
------------------------------------------------------------
Testing Phi-2 model with basic suite (3 tests)
  Test 1/3: Hello, how are you?...
    ✓ 1.85s, 124 chars, Quality: 0.82
  Test 2/3: What is 2 + 2?...
    ✓ 0.95s, 48 chars, Quality: 0.91
  Test 3/3: Write a function to add two numbers...
    ✓ 2.34s, 287 chars, Quality: 0.88

Testing cline-optimal model with basic suite (3 tests)
  Test 1/3: Hello, how are you?...
    ✓ 1.42s, 156 chars, Quality: 0.79
  Test 2/3: What is 2 + 2?...
    ✓ 0.78s, 52 chars, Quality: 0.93
  Test 3/3: Write a function to add two numbers...
    ✓ 1.89s, 312 chars, Quality: 0.85

⚡ Running speed benchmarks
------------------------------------------------------------
Phi-2 speed test (10 iterations): 1.847s avg
Cline speed test (10 iterations): 1.523s avg

================================================================================
🎯 COMPREHENSIVE MODEL COMPARISON COMPLETE
================================================================================

🏆 Overall Winner: CLINE_OPTIMAL

⚡ Speed: CLINE_OPTIMAL is faster
   Difference: -17.5%

💯 Quality: PHI2_FINETUNED has higher quality
   Difference: 0.035

📝 Recommendations:
   1. For speed-critical applications, use cline_optimal (>17% faster)
   2. For quality-critical applications, use phi2_finetuned (higher response quality)

📁 Detailed results saved to: c:/AI_Training/comparison_results
================================================================================
```

## 🤝 Contributing

To add new test cases or improve the framework:

1. Edit test suites in `comprehensive_model_comparison.py`
2. Add new quality assessment methods
3. Extend the configuration options
4. Submit improvements via pull request

## 📄 License

This tool is part of your AI training infrastructure. Use and modify as needed for your projects.

---

**Ready to see how your models stack up? Run the comparison and find out which model reigns supreme! 🏆**
