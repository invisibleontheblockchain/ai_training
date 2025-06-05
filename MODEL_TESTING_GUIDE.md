# Comprehensive Model Testing Guide

## 🎯 Overview

This testing framework provides a comprehensive evaluation of your AI model across multiple dimensions:

- **Coding Tasks**: Function writing, algorithm implementation, debugging
- **Reasoning**: Math problems, logical analysis, decision making  
- **Knowledge**: Technical explanations, domain expertise
- **Autonomy**: Ability to complete tasks without clarifying questions
- **Performance**: Speed, quality, consistency

## 🚀 Quick Start

### Test Your Models

```bash
# Test Phi-2 model with basic level
python test_your_model.py --model phi2 --level basic

# Test cline-optimal with all levels
python test_your_model.py --model cline-optimal --level all

# Compare both models
python test_your_model.py --model both --level intermediate

# Test autonomous capabilities specifically
python test_your_model.py --model both --level autonomous
```

### Test Levels Explained

| Level | Description | Example Tasks |
|-------|-------------|---------------|
| **Basic** | Simple tasks, clear requirements | "Write a function to add two numbers" |
| **Intermediate** | Technical knowledge, algorithms | "Implement binary search with error handling" |
| **Advanced** | System design, optimization | "Design a scalable chat application" |
| **Expert** | Research problems, novel solutions | "Propose new ML training optimizations" |
| **Autonomous** | Complete tasks without questions | "Create a web app" (no clarification allowed) |

## 📊 Understanding Results

### Key Metrics

- **Pass Rate**: Percentage of tests that meet all criteria
- **Quality Score**: 0-1 scale measuring response quality
- **Autonomy Score**: 0-1 scale measuring independence (fewer questions = higher)
- **Response Time**: Average time to generate responses

### Quality Assessment

The framework evaluates different aspects based on task type:

**Coding Tasks:**
- ✅ Proper syntax and structure
- ✅ Error handling
- ✅ Code comments/documentation
- ✅ Logical correctness

**Reasoning Tasks:**
- ✅ Step-by-step logic
- ✅ Logical connectors (because, therefore, etc.)
- ✅ Numerical calculations
- ✅ Analysis depth

**Knowledge Tasks:**
- ✅ Accuracy and depth
- ✅ Examples and illustrations
- ✅ Technical terminology
- ✅ Structured information

**Autonomy Tests:**
- ✅ Complete solutions without questions
- ✅ Reasonable assumptions
- ✅ Decision-making capability
- ✅ Self-directed problem solving

## 🎯 Sample Test Prompts

Here are examples of prompts from each difficulty level:

### Basic Level
```
1. "Hello! Can you introduce yourself and explain what you can help me with?"
2. "Calculate: 15 + 23 × 4 - 8 ÷ 2. Show your work step by step."
3. "Write a Python function that sums all even numbers in a list."
```

### Intermediate Level
```
1. "Implement a binary search algorithm with error handling and complexity analysis."
2. "Design a thread-safe queue data structure in Python."
3. "A company has 100 employees. 60% in engineering, 30% in sales, 10% management. 
   If it grows 50% but needs 15 extra managers, how many in each department?"
```

### Advanced Level
```
1. "Design a scalable real-time chat app for 1M concurrent users. Include architecture, 
   technology choices, database design, and deployment."
2. "Optimize this slow function: [code snippet]. Explain your optimizations."
3. "Design an ML recommendation system with collaborative and content-based filtering."
```

### Expert Level
```
1. "Propose a novel approach to reduce LLM training complexity while maintaining performance."
2. "How might quantum computing solve protein folding more efficiently than classical methods?"
```

### Autonomous Level
```
1. "Create a complete web application for task management. Don't ask questions - make decisions."
2. "Analyze remote work trends and provide actionable company recommendations."
3. "Design a data science learning curriculum with timeline and resources."
```

## 🔧 Customization

### Adding Custom Test Cases

Edit `comprehensive_model_test_prompt.py` and add to the test suites:

```python
TestCase(
    id="custom_001",
    category="your_category",
    difficulty="intermediate",
    task_type="coding",  # coding, reasoning, knowledge, general
    prompt="Your custom prompt here",
    expected_keywords=["keyword1", "keyword2"],
    min_response_length=100,
    max_response_time=30.0,
    evaluation_criteria={
        "criteria1": 0.4,
        "criteria2": 0.6
    },
    test_autonomy=False  # Set to True for autonomy tests
)
```

### Adjusting Evaluation Criteria

Modify the `_assess_*_quality` methods in the `ComprehensiveModelTester` class to change how responses are evaluated.

## 📈 Interpreting Results

### Good Performance Indicators
- **Pass Rate > 70%**: Model handles most tasks successfully
- **Quality Score > 0.7**: Responses are comprehensive and accurate
- **Autonomy Score > 0.8**: Model works independently
- **Response Time < 30s**: Acceptable performance speed

### Areas for Improvement
- **Low Pass Rate**: Review failed tests, may need more training
- **Low Quality**: Focus on response depth and accuracy
- **Low Autonomy**: Train to make decisions without clarification
- **High Response Time**: Optimize model or hardware

### Example Output Analysis

```
BASIC SUITE COMPARISON:
----------------------------------------
phi2            | Pass: 75% | Quality: 0.72 | Autonomy: 0.85 | Time: 12.3s
cline-optimal   | Pass: 85% | Quality: 0.81 | Autonomy: 0.92 | Time: 8.7s
```

This shows cline-optimal performs better across all metrics for basic tasks.

## 🛠 Troubleshooting

### Common Issues

**"Model not found"**
- Check model paths in the script
- Ensure models are properly installed
- Verify file permissions

**"Ollama connection failed"**
- Start Ollama service: `ollama serve`
- Check if cline-optimal model exists: `ollama list`
- Install if missing: `ollama pull cline-optimal:latest`

**"Low test scores"**
- Review specific failed tests in the detailed results
- Check if model is properly fine-tuned
- Consider additional training data

**"Memory errors"**
- Reduce batch size or model parameters
- Use quantized models if available
- Monitor GPU memory usage

## 📁 Output Files

The framework generates several output files:

- `test_results_[model]_[timestamp].json`: Detailed test results
- `test_report_[timestamp].txt`: Human-readable report
- `model_comparison_[timestamp].json`: Side-by-side comparison (when testing multiple models)
- `model_test_[timestamp].log`: Detailed execution log

## 🎯 Best Practices

1. **Start with Basic**: Always test basic level first to ensure model is working
2. **Incremental Testing**: Progress through levels to identify capability boundaries
3. **Multiple Runs**: Run tests multiple times to check consistency
4. **Monitor Resources**: Watch GPU/CPU usage during testing
5. **Review Failures**: Analyze failed tests to identify improvement areas
6. **Baseline Comparison**: Test against known good models for reference

## 🔄 Continuous Improvement

Use test results to guide model improvements:

1. **Identify Weak Areas**: Focus training on failed test categories
2. **Quality Metrics**: Use quality scores to guide fine-tuning
3. **Autonomy Training**: Add autonomous decision-making examples
4. **Speed Optimization**: Profile and optimize slow components
5. **Regular Testing**: Establish testing cadence for model updates

## 📞 Support

If you encounter issues:

1. Check the log files for detailed error messages
2. Verify all dependencies are installed
3. Ensure models are properly configured
4. Review the troubleshooting section above

For additional help, review the comprehensive test framework code and customize as needed for your specific use case.
