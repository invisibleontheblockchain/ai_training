"""
Enhanced Evaluation Metrics for AI Model Comparison
==================================================
Comprehensive evaluation beyond ROUGE-L for production AI systems.
"""

import re
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Comprehensive evaluation results"""
    rouge_l: float
    response_time: float
    autonomy_score: float
    code_quality: float
    reasoning_quality: float
    knowledge_accuracy: float
    overall_score: float
    task_type: str
    response_length: int
    question_count: int

class EnhancedEvaluator:
    """Enhanced evaluation metrics for AI model responses"""
    
    def __init__(self):
        self.code_patterns = {
            'functions': r'def\s+\w+\s*\([^)]*\):',
            'classes': r'class\s+\w+.*:',
            'imports': r'^(?:from\s+\w+\s+)?import\s+\w+',
            'comments': r'#.*$|""".*?"""|\'\'\'.*?\'\'\'',
            'error_handling': r'try:|except:|finally:|raise\s+\w+',
            'docstrings': r'""".*?"""|\'\'\'.*?\'\'\''
        }
        
        self.reasoning_indicators = [
            'because', 'therefore', 'since', 'as a result', 'consequently',
            'step 1', 'step 2', 'first', 'second', 'next', 'finally',
            'given that', 'assuming', 'if we', 'let\'s consider'
        ]
        
        self.knowledge_indicators = [
            'definition', 'concept', 'theory', 'principle', 'law',
            'research shows', 'studies indicate', 'evidence suggests',
            'according to', 'established', 'fundamental'
        ]
    
    def evaluate_code_quality(self, response: str) -> float:
        """Evaluate code quality in the response"""
        if not response:
            return 0.0
        
        score = 0.0
        
        # Check for code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', response)
        if not code_blocks:
            return 0.1  # Minimal score if no code blocks found
        
        code_content = '\n'.join(code_blocks)
        
        # Pattern-based quality checks
        for pattern_name, pattern in self.code_patterns.items():
            matches = len(re.findall(pattern, code_content, re.MULTILINE))
            
            if pattern_name == 'functions' and matches > 0:
                score += 0.2  # Functions present
            elif pattern_name == 'comments' and matches > 0:
                score += 0.15  # Comments/documentation
            elif pattern_name == 'error_handling' and matches > 0:
                score += 0.15  # Error handling
            elif pattern_name == 'imports' and matches > 0:
                score += 0.1   # Proper imports
        
        # Check for explanation accompanying code
        explanation_score = 0.2 if len(response.replace(''.join(code_blocks), '').strip()) > 100 else 0.1
        score += explanation_score
        
        # Check for working example/usage
        if 'example' in response.lower() or 'usage' in response.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def evaluate_reasoning_quality(self, response: str) -> float:
        """Evaluate logical reasoning quality"""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        score = 0.0
        
        # Count reasoning indicators
        reasoning_count = sum(1 for indicator in self.reasoning_indicators 
                            if indicator in response_lower)
        
        # Structure analysis
        structured_score = 0.0
        if '1.' in response or 'step 1' in response_lower:
            structured_score += 0.3  # Numbered steps
        if len(response.split('\n')) > 3:
            structured_score += 0.2  # Multi-line structure
        
        # Logical flow score
        logical_score = min(0.3, reasoning_count * 0.05)
        
        # Mathematical content (for reasoning tasks)
        math_score = 0.0
        if any(symbol in response for symbol in ['=', '+', '-', '*', '/', '%', '(', ')']):
            math_score = 0.2
        
        total_score = logical_score + structured_score + math_score
        return min(1.0, total_score)
    
    def evaluate_knowledge_accuracy(self, response: str) -> float:
        """Evaluate knowledge accuracy and depth"""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        score = 0.0
        
        # Knowledge indicators
        knowledge_count = sum(1 for indicator in self.knowledge_indicators 
                            if indicator in response_lower)
        knowledge_score = min(0.3, knowledge_count * 0.1)
        
        # Depth indicators
        depth_score = 0.0
        if len(response) > 500:  # Comprehensive response
            depth_score += 0.2
        if response.count('.') > 5:  # Multiple sentences/points
            depth_score += 0.2
        
        # Specific examples or applications
        example_score = 0.0
        if any(word in response_lower for word in ['example', 'instance', 'case', 'application']):
            example_score = 0.2
        
        # Technical accuracy (basic checks)
        accuracy_score = 0.1  # Base accuracy score
        if not any(word in response_lower for word in ['maybe', 'might be', 'possibly', 'unclear']):
            accuracy_score = 0.2  # Confident response
        
        total_score = knowledge_score + depth_score + example_score + accuracy_score
        return min(1.0, total_score)
    
    def calculate_rouge_l(self, reference: str, hypothesis: str) -> float:
        """Calculate ROUGE-L score"""
        if not reference or not hypothesis:
            return 0.0
        
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if not ref_tokens or not hyp_tokens:
            return 0.0
        
        lcs_len = lcs_length(ref_tokens, hyp_tokens)
        
        if lcs_len == 0:
            return 0.0
        
        precision = lcs_len / len(hyp_tokens)
        recall = lcs_len / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score
    
    def evaluate_comprehensive(self, 
                             response: str, 
                             reference: str, 
                             response_time: float,
                             task_type: str,
                             autonomy_metrics: Dict) -> EvaluationResults:
        """Perform comprehensive evaluation"""
        
        # Calculate individual metrics
        rouge_l = self.calculate_rouge_l(reference, response)
        code_quality = self.evaluate_code_quality(response) if task_type == 'coding' else 0.5
        reasoning_quality = self.evaluate_reasoning_quality(response) if task_type == 'reasoning' else 0.5
        knowledge_accuracy = self.evaluate_knowledge_accuracy(response) if task_type == 'knowledge' else 0.5
        
        # Task-specific weighting
        weights = {
            'coding': {'code': 0.4, 'reasoning': 0.2, 'knowledge': 0.2, 'rouge': 0.2},
            'reasoning': {'code': 0.1, 'reasoning': 0.5, 'knowledge': 0.2, 'rouge': 0.2},
            'knowledge': {'code': 0.1, 'reasoning': 0.2, 'knowledge': 0.5, 'rouge': 0.2},
            'general': {'code': 0.25, 'reasoning': 0.25, 'knowledge': 0.25, 'rouge': 0.25}
        }
        
        task_weights = weights.get(task_type, weights['general'])
        
        # Calculate weighted overall score
        overall_score = (
            code_quality * task_weights['code'] +
            reasoning_quality * task_weights['reasoning'] +
            knowledge_accuracy * task_weights['knowledge'] +
            rouge_l * task_weights['rouge']
        )
        
        # Apply autonomy bonus (up to 10% improvement)
        autonomy_bonus = autonomy_metrics.get('autonomy_score', 0.5) * 0.1
        overall_score = min(1.0, overall_score + autonomy_bonus)
        
        return EvaluationResults(
            rouge_l=rouge_l,
            response_time=response_time,
            autonomy_score=autonomy_metrics.get('autonomy_score', 0.0),
            code_quality=code_quality,
            reasoning_quality=reasoning_quality,
            knowledge_accuracy=knowledge_accuracy,
            overall_score=overall_score,
            task_type=task_type,
            response_length=len(response),
            question_count=autonomy_metrics.get('question_count', 0)
        )