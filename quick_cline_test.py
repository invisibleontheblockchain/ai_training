#!/usr/bin/env python3
"""
Quick Test for cline-optimal:latest
===================================
Simple baseline testing to evaluate cline-optimal performance
"""

import ollama
import time
import json
from datetime import datetime

def test_cline_optimal():
    """Test cline-optimal with various prompts"""
    
    test_cases = [
        {
            "name": "Basic Greeting",
            "prompt": "Hello, introduce yourself briefly",
            "category": "conversation"
        },
        {
            "name": "Simple Math",
            "prompt": "What is 15 + 23 × 4 - 8 ÷ 2? Show your work.",
            "category": "reasoning"
        },
        {
            "name": "Basic Coding",
            "prompt": "Write a Python function to find the maximum number in a list",
            "category": "coding"
        },
        {
            "name": "Technical Knowledge",
            "prompt": "Explain the difference between a list and a tuple in Python",
            "category": "knowledge"
        },
        {
            "name": "Algorithm Implementation",
            "prompt": "Implement a binary search algorithm in Python with error handling",
            "category": "coding"
        },
        {
            "name": "Autonomous Task",
            "prompt": "Create a simple Flask API for user registration without asking questions",
            "category": "autonomous"
        }
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "cline-optimal:latest",
        "tests": []
    }
    
    print("🧪 Testing cline-optimal:latest")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}/{len(test_cases)}: {test_case['name']}")
        print(f"Category: {test_case['category']}")
        print(f"Prompt: {test_case['prompt'][:60]}...")
        
        try:
            # Measure response time
            start_time = time.time()
            
            response = ollama.chat(
                model='cline-optimal:latest',
                messages=[{"role": "user", "content": test_case['prompt']}],
                options={
                    "temperature": 0.7,
                    "num_predict": 512
                }
            )
            
            response_time = time.time() - start_time
            response_text = response['message']['content']
            
            # Basic quality assessment
            quality_score = assess_quality(test_case, response_text)
            autonomy_score = assess_autonomy(response_text)
            
            test_result = {
                "name": test_case['name'],
                "category": test_case['category'],
                "prompt": test_case['prompt'],
                "response": response_text,
                "response_time": response_time,
                "response_length": len(response_text),
                "quality_score": quality_score,
                "autonomy_score": autonomy_score,
                "success": True
            }
            
            results["tests"].append(test_result)
            
            print(f"✓ Response time: {response_time:.2f}s")
            print(f"✓ Response length: {len(response_text)} chars")
            print(f"✓ Quality score: {quality_score:.2f}")
            print(f"✓ Autonomy score: {autonomy_score:.2f}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            test_result = {
                "name": test_case['name'],
                "category": test_case['category'],
                "prompt": test_case['prompt'],
                "error": str(e),
                "success": False
            }
            results["tests"].append(test_result)
    
    # Calculate overall metrics
    successful_tests = [t for t in results["tests"] if t.get("success", False)]
    
    if successful_tests:
        avg_response_time = sum(t["response_time"] for t in successful_tests) / len(successful_tests)
        avg_quality = sum(t["quality_score"] for t in successful_tests) / len(successful_tests)
        avg_autonomy = sum(t["autonomy_score"] for t in successful_tests) / len(successful_tests)
        avg_length = sum(t["response_length"] for t in successful_tests) / len(successful_tests)
        
        results["summary"] = {
            "total_tests": len(test_cases),
            "successful_tests": len(successful_tests),
            "success_rate": len(successful_tests) / len(test_cases),
            "avg_response_time": avg_response_time,
            "avg_quality_score": avg_quality,
            "avg_autonomy_score": avg_autonomy,
            "avg_response_length": avg_length
        }
        
        print("\n" + "=" * 50)
        print("📊 SUMMARY RESULTS")
        print("=" * 50)
        print(f"Success Rate: {results['summary']['success_rate']:.1%}")
        print(f"Average Response Time: {avg_response_time:.2f}s")
        print(f"Average Quality Score: {avg_quality:.2f}")
        print(f"Average Autonomy Score: {avg_autonomy:.2f}")
        print(f"Average Response Length: {avg_length:.0f} chars")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cline_optimal_test_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    return results

def assess_quality(test_case, response):
    """Simple quality assessment"""
    score = 0.0
    
    # Length check
    if len(response) > 20:
        score += 0.3
    
    # Category-specific checks
    category = test_case['category']
    response_lower = response.lower()
    
    if category == 'coding':
        if 'def ' in response or 'function' in response_lower:
            score += 0.4
        if 'return' in response_lower:
            score += 0.3
    elif category == 'reasoning':
        if any(word in response_lower for word in ['because', 'therefore', 'step']):
            score += 0.4
        if any(char.isdigit() for char in response):
            score += 0.3
    elif category == 'knowledge':
        if len(response) > 100:
            score += 0.4
        if any(word in response_lower for word in ['example', 'difference', 'used']):
            score += 0.3
    else:
        score += 0.7  # Default for conversation/autonomous
    
    return min(score, 1.0)

def assess_autonomy(response):
    """Assess autonomy (fewer questions = higher score)"""
    question_count = response.count('?')
    clarification_phrases = [
        'do you want', 'would you like', 'could you clarify', 'please specify',
        'what kind of', 'which type', 'more details'
    ]
    
    clarification_count = sum(1 for phrase in clarification_phrases if phrase in response.lower())
    total_issues = question_count + clarification_count
    
    if total_issues == 0:
        return 1.0
    elif total_issues <= 2:
        return 0.7
    elif total_issues <= 4:
        return 0.4
    else:
        return 0.1

if __name__ == "__main__":
    test_cline_optimal()
