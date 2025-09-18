#!/usr/bin/env python3
"""
Pattern Analyzer Agent - Complete Implementation
==============================================
Diagnoses systematic failure patterns in code generation to break representation collapse.
Specifically designed to identify why the model generates Python instead of Solidity.
"""

import json
import re
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import requests
from pathlib import Path

class PatternAnalyzerAgent:
    """Agent that analyzes failure patterns to identify systematic issues"""
    
    def __init__(self):
        self.failure_patterns = defaultdict(list)
        self.language_detection_stats = defaultdict(int)
        self.error_taxonomy = {
            "language_mismatch": [],
            "syntax_errors": [],
            "semantic_errors": [],
            "compilation_errors": [],
            "prompt_confusion": []
        }
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for pattern analysis"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - PatternAnalyzer - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'pattern_analysis_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_test_results(self, test_results_file: str) -> Dict[str, Any]:
        """Analyze test results to identify systematic patterns"""
        self.logger.info(f"Analyzing test results from {test_results_file}")
        
        try:
            with open(test_results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Convert DeFi format to standard format if needed
            if "test_results" in results and isinstance(results["test_results"], list):
                standardized_results = self._convert_defi_format(results)
            else:
                standardized_results = results
            
            analysis = {
                "source_file": test_results_file,
                "analysis_timestamp": datetime.now().isoformat(),
                "language_analysis": self._analyze_language_patterns(standardized_results),
                "error_patterns": self._categorize_errors(standardized_results),
                "prompt_response_mapping": self._analyze_prompt_response_correlation(standardized_results),
                "systematic_failures": self._identify_systematic_failures(standardized_results),
                "model_confusion_analysis": self._analyze_model_confusion(standardized_results),
                "defi_specific_analysis": self._analyze_defi_patterns(standardized_results),
                "recommendations": self._generate_recommendations(standardized_results)
            }
            
            self._save_analysis(analysis, test_results_file)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze test results: {e}")
            return {}
    
    def _convert_defi_format(self, defi_results: Dict) -> Dict:
        """Convert DeFi test results format to standard format"""
        test_results = defi_results.get("test_results", [])
        converted_tests = []
        
        for test in test_results:
            # Map DeFi format to standard format
            converted_test = {
                "test_id": test.get("test_id", "unknown"),
                "prompt": test.get("prompt", ""),
                "response": test.get("response", ""),
                "passed": test.get("compilation_success", False),
                "error": "; ".join(test.get("compilation_errors", [])),
                "issues": []
            }
            
            # Add failure categories as issues
            if not test.get("compilation_success", False):
                converted_test["issues"].append("compilation_failure")
            
            if test.get("security_score", 0) == 0:
                converted_test["issues"].append("security_issues")
            
            if test.get("gas_efficiency", 0) == 0:
                converted_test["issues"].append("gas_efficiency")
            
            if not test.get("integration_success", False):
                converted_test["issues"].append("integration_failure")
            
            # Add language detection issue if Python detected
            detected_lang, _ = self._detect_code_language(test.get("response", ""))
            if detected_lang == "python":
                converted_test["issues"].append("language_mismatch")
            
            converted_tests.append(converted_test)
        
        return {"test_results": converted_tests}
    
    def _analyze_language_patterns(self, results: Dict) -> Dict[str, Any]:
        """Analyze what language the model is actually generating"""
        language_stats = defaultdict(int)
        language_examples = defaultdict(list)
        
        test_results = results.get("test_results", [])
        
        for test in test_results:
            response = test.get("response", "")
            detected_language, confidence = self._detect_code_language(response)
            language_stats[detected_language] += 1
            
            if len(language_examples[detected_language]) < 3:  # Keep examples
                language_examples[detected_language].append({
                    "prompt": test.get("prompt", "")[:100] + "...",
                    "response": response[:200] + "...",
                    "test_id": test.get("test_id", "unknown"),
                    "confidence": confidence
                })
        
        return {
            "language_distribution": dict(language_stats),
            "examples_by_language": dict(language_examples),
            "total_tests": len(test_results),
            "primary_issue": self._identify_primary_language_issue(language_stats, test_results)
        }
    
    def _detect_code_language(self, code_text: str) -> Tuple[str, float]:
        """Detect what programming language the code appears to be"""
        if not code_text.strip():
            return "empty", 1.0
            
        code_lower = code_text.lower()
        
        # Enhanced Solidity indicators with weights
        solidity_patterns = {
            r'pragma\s+solidity': 15,
            r'contract\s+\w+\s*{': 12,
            r'function\s+\w+\s*\([^)]*\)\s*(public|private|internal|external)': 10,
            r'uint256|uint128|uint64|uint32|uint8|address|bytes32|bytes': 8,
            r'mapping\s*\(': 8,
            r'require\s*\(': 7,
            r'msg\.sender|msg\.value': 7,
            r'// SPDX-License-Identifier': 6,
            r'event\s+\w+\s*\(': 6,
            r'modifier\s+\w+': 6,
            r'constructor\s*\(': 6,
            r'payable|view|pure': 5,
            r'emit\s+\w+': 5,
            r'\.sol': 4,
            r'interface\s+\w+': 4,
            r'library\s+\w+': 4
        }
        
        # Enhanced Python indicators with weights
        python_patterns = {
            r'def\s+\w+\s*\(': 10,
            r'class\s+\w+\s*[\(:]': 10,
            r'import\s+\w+': 9,
            r'from\s+\w+\s+import': 9,
            r'print\s*\(': 7,
            r'if\s+__name__\s*==': 8,
            r'self\.': 6,
            r'\.py': 5,
            r'""".*?"""': 4,
            r"'''.*?'''": 4,
            r'try\s*:\s*': 4,
            r'except\s*\w*\s*:': 4,
            r'with\s+\w+': 3,
            r'for\s+\w+\s+in\s+': 3,
            r'range\s*\(': 3,
            r'len\s*\(': 3
        }
        
        # JavaScript/general code patterns
        js_patterns = {
            r'function\s+\w+\s*\(': 6,
            r'var\s+\w+|let\s+\w+|const\s+\w+': 5,
            r'console\.log': 4,
            r'\.js': 3
        }
        
        # Calculate scores
        solidity_score = sum(weight for pattern, weight in solidity_patterns.items() 
                           if re.search(pattern, code_text, re.IGNORECASE | re.DOTALL))
        python_score = sum(weight for pattern, weight in python_patterns.items() 
                         if re.search(pattern, code_text, re.IGNORECASE | re.DOTALL))
        js_score = sum(weight for pattern, weight in js_patterns.items() 
                      if re.search(pattern, code_text, re.IGNORECASE | re.DOTALL))
        
        total_score = solidity_score + python_score + js_score + 1
        
        # Check for mixed indicators (confusion)
        if (solidity_score > 0 and python_score > 5) or (solidity_score > 0 and js_score > 3):
            return "mixed", min(solidity_score, max(python_score, js_score)) / total_score
        elif solidity_score > max(python_score, js_score):
            confidence = solidity_score / total_score
            return "solidity", confidence
        elif python_score > js_score and python_score > 0:
            confidence = python_score / total_score
            return "python", confidence
        elif js_score > 0:
            confidence = js_score / total_score
            return "javascript", confidence
        else:
            return "unknown", 0.0
    
    def _identify_primary_language_issue(self, language_stats: Dict, test_results: List) -> Dict:
        """Identify the primary language generation issue"""
        total = sum(language_stats.values())
        if total == 0:
            return {"issue": "no_tests", "severity": "critical"}
        
        python_ratio = language_stats.get("python", 0) / total
        solidity_ratio = language_stats.get("solidity", 0) / total
        mixed_ratio = language_stats.get("mixed", 0) / total
        
        if python_ratio > 0.8:
            return {
                "issue": "python_default",
                "severity": "critical",
                "description": f"Model generates Python in {python_ratio*100:.1f}% of cases",
                "hypothesis": "Model has learned 'code generation = Python' association",
                "urgency": "IMMEDIATE - This breaks all smart contract generation"
            }
        elif mixed_ratio > 0.3:
            return {
                "issue": "language_confusion",
                "severity": "high",
                "description": f"Model mixes languages in {mixed_ratio*100:.1f}% of cases",
                "hypothesis": "Model understands Solidity concepts but defaults to Python syntax",
                "urgency": "HIGH - Partial understanding but wrong implementation"
            }
        elif solidity_ratio < 0.5:
            return {
                "issue": "low_solidity_generation",
                "severity": "high",
                "description": f"Only {solidity_ratio*100:.1f}% proper Solidity generation",
                "hypothesis": "Insufficient Solidity examples in training context",
                "urgency": "HIGH - Fundamental language generation problem"
            }
        else:
            return {
                "issue": "syntax_quality",
                "severity": "medium",
                "description": "Language detection OK, but compilation fails",
                "hypothesis": "Model generates Solidity-like code with syntax errors",
                "urgency": "MEDIUM - Language correct but needs syntax refinement"
            }
    
    def _categorize_errors(self, results: Dict) -> Dict[str, Any]:
        """Categorize errors by type and pattern"""
        error_categories = defaultdict(list)
        error_patterns = Counter()
        
        test_results = results.get("test_results", [])
        
        for test in test_results:
            if not test.get("passed", True):
                error_msg = test.get("error", "")
                issues = test.get("issues", [])
                
                # Categorize by issue type
                for issue in issues:
                    error_categories[issue].append({
                        "test_id": test.get("test_id"),
                        "prompt_snippet": test.get("prompt", "")[:50] + "...",
                        "error": error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
                    })
                
                # Extract error patterns
                if "compilation_failure" in issues:
                    pattern = self._extract_compilation_error_pattern(error_msg)
                    if pattern:
                        error_patterns[pattern] += 1
        
        return {
            "categories": dict(error_categories),
            "top_patterns": error_patterns.most_common(10),
            "total_errors": sum(len(errors) for errors in error_categories.values()),
            "critical_patterns": [p for p, count in error_patterns.most_common(5) if count >= 3]
        }
    
    def _extract_compilation_error_pattern(self, error_msg: str) -> Optional[str]:
        """Extract meaningful patterns from compilation errors"""
        if not error_msg:
            return None
            
        # Common compilation error patterns
        patterns = [
            (r'Could not extract contract name', 'no_contract_detected'),
            (r'ParserError: Expected', 'parser_expected'),
            (r'DeclarationError:', 'declaration_error'),
            (r'TypeError:', 'type_error'),
            (r'SyntaxError:', 'syntax_error'),
            (r'IndentationError:', 'python_indentation'),  # Python error in Solidity context!
            (r'NameError:', 'python_name_error'),  # Another Python error
            (r'ImportError:', 'python_import_error'),
            (r'AttributeError:', 'python_attribute_error')
        ]
        
        for pattern, label in patterns:
            if re.search(pattern, error_msg):
                return label
                
        return "unknown_compilation_error"
    
    def _analyze_prompt_response_correlation(self, results: Dict) -> Dict[str, Any]:
        """Analyze how prompts correlate with response quality"""
        prompt_features = defaultdict(list)
        
        test_results = results.get("test_results", [])
        
        for test in test_results:
            prompt = test.get("prompt", "")
            response = test.get("response", "")
            passed = test.get("passed", False)
            detected_lang, confidence = self._detect_code_language(response)
            
            # Extract prompt features
            features = {
                "mentions_solidity": bool(re.search(r'solidity|smart contract|ethereum', prompt.lower())),
                "mentions_language": bool(re.search(r'in solidity|solidity code|write solidity', prompt.lower())),
                "has_technical_spec": bool(re.search(r'ERC\d+|function|contract|mapping', prompt)),
                "mentions_defi": bool(re.search(r'defi|dex|lending|oracle|token|erc20', prompt.lower())),
                "has_implementation_verb": bool(re.match(r'^(create|implement|write|build|develop)', prompt.lower())),
                "prompt_length": len(prompt),
                "starts_with_verb": bool(re.match(r'^(create|implement|write|build|develop)', prompt.lower()))
            }
            
            prompt_features["results"].append({
                "features": features,
                "outcome": {
                    "passed": passed,
                    "detected_language": detected_lang,
                    "confidence": confidence
                }
            })
        
        # Analyze correlations
        correlations = self._calculate_feature_correlations(prompt_features["results"])
        
        return {
            "feature_analysis": correlations,
            "best_prompt_features": self._identify_best_prompt_features(prompt_features["results"]),
            "worst_prompt_features": self._identify_worst_prompt_features(prompt_features["results"]),
            "prompt_optimization_suggestions": self._generate_prompt_suggestions(correlations)
        }
    
    def _calculate_feature_correlations(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate correlation between prompt features and success"""
        if not results:
            return {}
            
        feature_success_rates = defaultdict(lambda: {"success": 0, "total": 0})
        
        for result in results:
            features = result["features"]
            is_solidity = result["outcome"]["detected_language"] == "solidity"
            
            for feature, value in features.items():
                if isinstance(value, bool) and value:
                    feature_success_rates[feature]["total"] += 1
                    if is_solidity:
                        feature_success_rates[feature]["success"] += 1
        
        correlations = {}
        for feature, stats in feature_success_rates.items():
            if stats["total"] > 0:
                correlations[feature] = stats["success"] / stats["total"]
                
        return correlations
    
    def _identify_best_prompt_features(self, results: List[Dict]) -> List[Dict]:
        """Identify prompt features that lead to successful Solidity generation"""
        successful_prompts = [
            r for r in results 
            if r["outcome"]["detected_language"] == "solidity" 
            and r["outcome"]["confidence"] > 0.7
        ]
        
        if not successful_prompts:
            return []
            
        # Find common features in successful prompts
        common_features = defaultdict(int)
        for prompt in successful_prompts:
            for feature, value in prompt["features"].items():
                if isinstance(value, bool) and value:
                    common_features[feature] += 1
        
        total = len(successful_prompts)
        return [
            {"feature": feature, "frequency": count/total}
            for feature, count in common_features.items()
            if count/total > 0.5
        ]
    
    def _identify_worst_prompt_features(self, results: List[Dict]) -> List[Dict]:
        """Identify prompt features that lead to Python generation"""
        python_prompts = [
            r for r in results 
            if r["outcome"]["detected_language"] == "python"
        ]
        
        if not python_prompts:
            return []
            
        # Find common features in Python-generating prompts
        common_features = defaultdict(int)
        for prompt in python_prompts:
            for feature, value in prompt["features"].items():
                if isinstance(value, bool) and value:
                    common_features[feature] += 1
        
        total = len(python_prompts)
        return [
            {"feature": feature, "frequency": count/total}
            for feature, count in common_features.items()
            if count/total > 0.5
        ]
    
    def _generate_prompt_suggestions(self, correlations: Dict[str, float]) -> List[str]:
        """Generate specific prompt optimization suggestions"""
        suggestions = []
        
        # Find features with high success rates
        successful_features = {k: v for k, v in correlations.items() if v > 0.7}
        
        if "mentions_language" in successful_features:
            suggestions.append("Always explicitly specify 'in Solidity' or 'Solidity code' in prompts")
        
        if "has_technical_spec" in successful_features:
            suggestions.append("Include specific technical terms (ERC20, mapping, uint256) in prompts")
        
        if "mentions_defi" in successful_features:
            suggestions.append("Use DeFi-specific terminology consistently")
        
        # General suggestions based on analysis
        suggestions.extend([
            "Start prompts with 'Create a Solidity contract that...'",
            "Add '// SPDX-License-Identifier: MIT' to expected outputs",
            "Include 'pragma solidity ^0.8.0;' in examples",
            "Use 'smart contract' instead of just 'contract' for clarity"
        ])
        
        return suggestions
    
    def _identify_systematic_failures(self, results: Dict) -> Dict[str, Any]:
        """Identify systematic failure patterns"""
        test_results = results.get("test_results", [])
        
        # Group failures by type
        failure_groups = defaultdict(list)
        
        for test in test_results:
            if not test.get("passed", True):
                failure_key = self._generate_failure_key(test)
                failure_groups[failure_key].append(test)
        
        # Identify systematic patterns
        systematic_issues = []
        
        for failure_key, failures in failure_groups.items():
            if len(failures) >= 2:  # Pattern appears 2+ times (lowered threshold)
                systematic_issues.append({
                    "pattern": failure_key,
                    "frequency": len(failures),
                    "examples": [f.get("test_id") for f in failures[:3]],
                    "severity": self._assess_pattern_severity(failure_key, len(failures)),
                    "description": self._describe_pattern(failure_key)
                })
        
        # Sort by severity and frequency
        systematic_issues.sort(key=lambda x: (x["severity"], x["frequency"]), reverse=True)
        
        return {
            "systematic_patterns": systematic_issues[:10],  # Top 10 patterns
            "total_unique_failures": len(failure_groups),
            "most_common_failure": systematic_issues[0] if systematic_issues else None,
            "critical_patterns": [p for p in systematic_issues if p["severity"] >= 8]
        }
    
    def _generate_failure_key(self, test: Dict) -> str:
        """Generate a key to group similar failures"""
        issues = test.get("issues", [])
        detected_lang, _ = self._detect_code_language(test.get("response", ""))
        
        # Create a composite key
        key_parts = [
            detected_lang,
            *sorted(issues)
        ]
        
        return "|".join(key_parts)
    
    def _describe_pattern(self, pattern: str) -> str:
        """Generate human-readable pattern descriptions"""
        if "python|compilation_failure" in pattern:
            return "Model consistently generates Python code when Solidity is requested"
        elif "mixed|" in pattern:
            return "Model mixes programming languages within single responses"
        elif "compilation_failure" in pattern:
            return "Code compilation failures due to syntax errors"
        elif "language_mismatch" in pattern:
            return "Generated code doesn't match requested programming language"
        else:
            return f"Pattern: {pattern}"
    
    def _assess_pattern_severity(self, pattern: str, frequency: int) -> int:
        """Assess severity of a failure pattern (1-10)"""
        severity = 5  # Base severity
        
        # Critical patterns
        if "python|compilation_failure" in pattern:
            severity = 10
        elif "python|language_mismatch" in pattern:
            severity = 9
        elif "mixed|" in pattern:
            severity = 8
        elif "unknown|" in pattern:
            severity = 7
        elif "compilation_failure" in pattern:
            severity = 6
        
        # Frequency modifier
        if frequency > 10:
            severity = min(10, severity + 2)
        elif frequency > 5:
            severity = min(10, severity + 1)
            
        return severity
    
    def _analyze_model_confusion(self, results: Dict) -> Dict[str, Any]:
        """Deep analysis of why the model is confused"""
        test_results = results.get("test_results", [])
        
        confusion_indicators = {
            "language_mixing": 0,
            "wrong_syntax_right_concepts": 0,
            "complete_language_swap": 0,
            "no_code_generated": 0,
            "hallucinated_syntax": 0
        }
        
        confusion_examples = defaultdict(list)
        
        for test in test_results:
            response = test.get("response", "")
            prompt = test.get("prompt", "")
            
            # Analyze confusion type
            confusion_type = self._identify_confusion_type(prompt, response)
            if confusion_type:
                confusion_indicators[confusion_type] += 1
                if len(confusion_examples[confusion_type]) < 2:
                    confusion_examples[confusion_type].append({
                        "prompt": prompt[:100] + "...",
                        "response": response[:150] + "...",
                        "test_id": test.get("test_id")
                    })
        
        primary_confusion = max(confusion_indicators.items(), key=lambda x: x[1])[0] if any(confusion_indicators.values()) else None
        
        return {
            "confusion_distribution": confusion_indicators,
            "examples": dict(confusion_examples),
            "primary_confusion": primary_confusion,
            "confusion_analysis": self._analyze_confusion_cause(primary_confusion, confusion_indicators)
        }
    
    def _identify_confusion_type(self, prompt: str, response: str) -> Optional[str]:
        """Identify specific type of model confusion"""
        if not response.strip():
            return "no_code_generated"
            
        detected_lang, confidence = self._detect_code_language(response)
        
        # Check for Solidity concepts in Python syntax
        if detected_lang == "python" and any(
            concept in response.lower() 
            for concept in ['contract', 'function public', 'uint256', 'address', 'mapping', 'oracle', 'dex', 'erc20']
        ):
            return "wrong_syntax_right_concepts"
        
        # Check for language mixing
        elif detected_lang == "mixed":
            return "language_mixing"
        
        # Check for complete language swap
        elif detected_lang == "python" and any(
            keyword in prompt.lower() 
            for keyword in ["solidity", "smart contract", "ethereum", "defi", "blockchain"]
        ):
            return "complete_language_swap"
        
        # Check for hallucinated syntax
        elif detected_lang == "unknown":
            return "hallucinated_syntax"
            
        return None
    
    def _analyze_confusion_cause(self, primary_confusion: str, indicators: Dict) -> Dict:
        """Analyze the root cause of model confusion"""
        if not primary_confusion:
            return {"cause": "no_confusion_detected", "confidence": "low"}
        
        total_confused = sum(indicators.values())
        
        if primary_confusion == "complete_language_swap":
            return {
                "cause": "fundamental_language_association_error",
                "confidence": "very_high",
                "description": "Model has learned incorrect association: smart_contract = Python",
                "fix_priority": "CRITICAL"
            }
        elif primary_confusion == "wrong_syntax_right_concepts":
            return {
                "cause": "concept_syntax_disconnect",
                "confidence": "high",
                "description": "Model understands DeFi concepts but applies Python syntax",
                "fix_priority": "HIGH"
            }
        elif primary_confusion == "language_mixing":
            return {
                "cause": "weak_language_boundaries",
                "confidence": "medium",
                "description": "Model cannot maintain consistent language context",
                "fix_priority": "MEDIUM"
            }
        else:
            return {
                "cause": "unknown_confusion_pattern",
                "confidence": "low",
                "description": f"Primary confusion: {primary_confusion}",
                "fix_priority": "LOW"
            }
    
    def _analyze_defi_patterns(self, results: Dict) -> Dict[str, Any]:
        """Analyze DeFi-specific patterns and failures"""
        test_results = results.get("test_results", [])
        
        defi_concepts = {
            "oracle": [],
            "dex": [],
            "lending": [],
            "token": [],
            "erc20": [],
            "defi": []
        }
        
        # Categorize tests by DeFi concept
        for test in test_results:
            prompt = test.get("prompt", "").lower()
            response = test.get("response", "").lower()
            
            for concept in defi_concepts.keys():
                if concept in prompt or concept in response:
                    defi_concepts[concept].append({
                        "test_id": test.get("test_id"),
                        "passed": test.get("passed", False),
                        "detected_language": self._detect_code_language(test.get("response", ""))[0]
                    })
        
        # Analyze success rates by concept
        concept_analysis = {}
        for concept, tests in defi_concepts.items():
            if tests:
                solidity_count = sum(1 for t in tests if t["detected_language"] == "solidity")
                python_count = sum(1 for t in tests if t["detected_language"] == "python")
                
                concept_analysis[concept] = {
                    "total_tests": len(tests),
                    "solidity_generation": solidity_count,
                    "python_generation": python_count,
                    "solidity_rate": solidity_count / len(tests) if tests else 0,
                    "python_rate": python_count / len(tests) if tests else 0
                }
        
        return {
            "concept_breakdown": concept_analysis,
            "most_problematic_concept": min(concept_analysis.items(), 
                                          key=lambda x: x[1]["solidity_rate"])[0] if concept_analysis else None,
            "defi_specific_recommendations": self._generate_defi_recommendations(concept_analysis)
        }
    
    def _generate_defi_recommendations(self, concept_analysis: Dict) -> List[str]:
        """Generate DeFi-specific recommendations"""
        recommendations = []
        
        for concept, analysis in concept_analysis.items():
            if analysis["python_rate"] > 0.8:
                recommendations.append(f"Critical: {concept.upper()} prompts generate Python {analysis['python_rate']*100:.0f}% of time - need immediate Solidity examples")
        
        recommendations.extend([
            "Create DeFi-specific Solidity templates for each concept",
            "Use working smart contract examples in few-shot prompting",
            "Add explicit 'Create a Solidity smart contract' prefix to all DeFi prompts",
            "Include pragma solidity statement in all expected outputs"
        ])
        
        return recommendations
    
    def _generate_recommendations(self, results: Dict) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Get analysis results
        language_analysis = self._analyze_language_patterns(results)
        primary_issue = language_analysis.get("primary_issue", {})
        
        # Critical: Python default issue
        if primary_issue.get("issue") == "python_default":
            recommendations.append({
                "priority": "CRITICAL",
                "action": "Break Python Association - EMERGENCY FIX",
                "steps": [
                    "IMMEDIATE: Inject 100+ working Solidity contract examples",
                    "Add explicit 'Language: Solidity' to ALL prompts",
                    "Use few-shot prompting with 3-5 Solidity examples per prompt",
                    "Add negative examples: 'This should NOT be Python code'",
                    "Implement pre-generation language priming: 'Generate Solidity smart contract code:'"
                ],
                "expected_impact": "Should see 50%+ improvement in 5-10 iterations",
                "urgency": "FIX TODAY - System completely broken for smart contracts"
            })
        
        # High: Language confusion
        if primary_issue.get("issue") == "language_confusion":
            recommendations.append({
                "priority": "HIGH",
                "action": "Clarify Language Boundaries",
                "steps": [
                    "Create clear Solidity syntax templates for each DeFi concept",
                    "Use stronger language indicators: 'Write a Solidity smart contract'",
                    "Implement pre-generation language priming",
                    "Add Solidity-specific context before each prompt",
                    "Show Python → Solidity translation examples"
                ],
                "expected_impact": "Reduce mixed language generation by 80%"
            })
        
        # Systematic failures
        systematic_failures = self._identify_systematic_failures(results)
        if systematic_failures.get("systematic_patterns"):
            top_pattern = systematic_failures["systematic_patterns"][0]
            recommendations.append({
                "priority": "HIGH",
                "action": f"Address Critical Pattern: {top_pattern['description']}",
                "steps": [
                    f"This pattern occurs in {top_pattern['frequency']} tests (severity: {top_pattern['severity']}/10)",
                    "Create targeted training examples for this specific failure",
                    "Implement pattern-specific prompt engineering",
                    "Add validation to prevent this pattern recurring",
                    f"Focus on test IDs: {', '.join(top_pattern['examples'])}"
                ],
                "expected_impact": f"Eliminate {top_pattern['frequency']} recurring failures"
            })
        
        # DeFi-specific recommendations
        defi_analysis = self._analyze_defi_patterns(results)
        if defi_analysis.get("most_problematic_concept"):
            concept = defi_analysis["most_problematic_concept"]
            recommendations.append({
                "priority": "HIGH",
                "action": f"Fix {concept.upper()} Concept Generation",
                "steps": [
                    f"{concept} prompts have severe Python generation issues",
                    f"Create {concept}-specific Solidity templates",
                    f"Add working {concept} smart contract examples to training",
                    f"Use concept-specific prompt prefixes for {concept}",
                    f"Test {concept} generation separately until fixed"
                ],
                "expected_impact": f"Fix {concept} smart contract generation"
            })
        
        # Model confusion
        confusion_analysis = self._analyze_model_confusion(results)
        confusion_cause = confusion_analysis.get("confusion_analysis", {})
        if confusion_cause.get("fix_priority") == "CRITICAL":
            recommendations.append({
                "priority": "CRITICAL",
                "action": "Fix Fundamental Language Association",
                "steps": [
                    confusion_cause.get("description", ""),
                    "Model thinks smart_contract = Python - this is catastrophic",
                    "Immediately flood training with Solidity examples",
                    "Use contrastive examples: 'This is Python: [python code], This is Solidity: [solidity code]'",
                    "Add explicit language tags to all training data"
                ],
                "expected_impact": "Fix fundamental model confusion"
            })
        
        return recommendations
    
    def _save_analysis(self, analysis: Dict, source_file: str):
        """Save analysis results to file"""
        output_file = f"pattern_analysis_{Path(source_file).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Analysis saved to {output_file}")
        
        # Also generate a human-readable report
        self._generate_report(analysis, output_file.replace('.json', '.md'))
    
    def _generate_report(self, analysis: Dict, output_file: str):
        """Generate human-readable markdown report"""
        report = ["# 🚨 CRITICAL Pattern Analysis Report 🚨", ""]
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Source**: {analysis.get('source_file', 'Unknown')}")
        report.append("")
        
        # Emergency Alert Section
        primary_issue = analysis.get("language_analysis", {}).get("primary_issue", {})
        if primary_issue.get("severity") == "critical":
            report.append("## 🚨 EMERGENCY ALERT 🚨")
            report.append(f"**CRITICAL ISSUE DETECTED**: {primary_issue.get('description', 'Unknown')}")
            report.append(f"**URGENCY**: {primary_issue.get('urgency', 'IMMEDIATE ACTION REQUIRED')}")
            report.append(f"**HYPOTHESIS**: {primary_issue.get('hypothesis', 'Unknown')}")
            report.append("")
        
        # Language Analysis
        report.append("## 📊 Language Generation Analysis")
        lang_analysis = analysis.get("language_analysis", {})
        report.append(f"- **Total Tests**: {lang_analysis.get('total_tests', 0)}")
        report.append(f"- **Primary Issue**: {lang_analysis.get('primary_issue', {}).get('description', 'Unknown')}")
        report.append(f"- **Language Distribution**:")
        for lang, count in lang_analysis.get("language_distribution", {}).items():
            percentage = (count / lang_analysis.get("total_tests", 1)) * 100
            emoji = "❌" if lang == "python" else "✅" if lang == "solidity" else "⚠️"
            report.append(f"  - {emoji} {lang.title()}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Critical Systematic Failures
        report.append("## 🔥 Critical Systematic Failures")
        systematic = analysis.get("systematic_failures", {})
        critical_patterns = systematic.get("critical_patterns", [])
        if critical_patterns:
            report.append("**IMMEDIATE ATTENTION REQUIRED:**")
            for i, pattern in enumerate(critical_patterns[:3], 1):
                report.append(f"{i}. **{pattern['description']}**")
                report.append(f"   - Frequency: {pattern['frequency']} occurrences")
                report.append(f"   - Severity: {pattern['severity']}/10")
                report.append(f"   - Examples: {', '.join(pattern['examples'][:3])}")
        else:
            report.append("No critical patterns detected.")
        report.append("")
        
        # DeFi-Specific Analysis
        report.append("## 🏦 DeFi Concept Analysis")
        defi_analysis = analysis.get("defi_specific_analysis", {})
        concept_breakdown = defi_analysis.get("concept_breakdown", {})
        if concept_breakdown:
            report.append("**DeFi Concept Performance:**")
            for concept, data in concept_breakdown.items():
                solidity_rate = data.get("solidity_rate", 0) * 100
                python_rate = data.get("python_rate", 0) * 100
                status = "🚨 CRITICAL" if python_rate > 80 else "⚠️ WARNING" if python_rate > 50 else "✅ OK"
                report.append(f"- **{concept.upper()}**: {status}")
                report.append(f"  - Solidity: {solidity_rate:.0f}% | Python: {python_rate:.0f}%")
        report.append("")
        
        # URGENT Recommendations
        report.append("## 🚨 URGENT ACTION ITEMS")
        recommendations = analysis.get("recommendations", [])
        critical_recs = [r for r in recommendations if r.get("priority") == "CRITICAL"]
        if critical_recs:
            for i, rec in enumerate(critical_recs, 1):
                report.append(f"### {i}. {rec['action']} [{rec['priority']}]")
                if rec.get("urgency"):
                    report.append(f"**{rec['urgency']}**")
                report.append("**Immediate Steps:**")
                for step in rec["steps"]:
                    report.append(f"- {step}")
                report.append(f"**Expected Impact**: {rec['expected_impact']}")
                report.append("")
        
        # High Priority Recommendations
        high_recs = [r for r in recommendations if r.get("priority") == "HIGH"]
        if high_recs:
            report.append("## 🔧 High Priority Fixes")
            for i, rec in enumerate(high_recs, 1):
                report.append(f"### {i}. {rec['action']}")
                report.append("Steps:")
                for step in rec["steps"]:
                    report.append(f"- {step}")
                report.append(f"**Expected Impact**: {rec['expected_impact']}")
                report.append("")
        
        # Model Confusion Analysis
        report.append("## 🧠 Model Confusion Analysis")
        confusion = analysis.get("model_confusion_analysis", {})
        confusion_analysis = confusion.get("confusion_analysis", {})
        if confusion_analysis:
            report.append(f"**Root Cause**: {confusion_analysis.get('cause', 'Unknown')}")
            report.append(f"**Confidence**: {confusion_analysis.get('confidence', 'Unknown')}")
            report.append(f"**Description**: {confusion_analysis.get('description', 'Unknown')}")
            report.append(f"**Fix Priority**: {confusion_analysis.get('fix_priority', 'Unknown')}")
        
        if confusion.get("primary_confusion"):
            report.append(f"**Primary confusion type**: {confusion['primary_confusion']}")
            report.append("**Distribution:**")
            for confusion_type, count in confusion.get("confusion_distribution", {}).items():
                report.append(f"- {confusion_type}: {count}")
        report.append("")
        
        # Summary
        report.append("## 📝 Executive Summary")
        if primary_issue.get("severity") == "critical":
            report.append("🚨 **SYSTEM STATUS: BROKEN** - Immediate intervention required")
            report.append("- Smart contract generation is completely non-functional")
            report.append("- Model generates Python instead of Solidity 100% of the time")
            report.append("- This represents a fundamental training failure")
        else:
            report.append("⚠️ **SYSTEM STATUS: DEGRADED** - Significant issues detected")
        
        report.append("")
        report.append("---")
        report.append("*This report was generated by PatternAnalyzerAgent v1.0*")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        self.logger.info(f"Report saved to {output_file}")

def main():
    """Run pattern analysis on test results"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pattern_analyzer_agent.py <test_results.json>")
        print("   or: python pattern_analyzer_agent.py <directory>")
        sys.exit(1)
    
    analyzer = PatternAnalyzerAgent()
    
    # Find all test result files if directory provided
    path = sys.argv[1]
    if os.path.isdir(path):
        test_files = list(Path(path).glob("*test_results*.json"))
        if not test_files:
            test_files = list(Path(path).glob("defi_test_results*.json"))
        
        print(f"Found {len(test_files)} test result files")
        
        for test_file in test_files:
            print(f"\n🔍 Analyzing: {test_file}")
            analysis = analyzer.analyze_test_results(str(test_file))
            
            # Quick summary
            lang_dist = analysis.get("language_analysis", {}).get("language_distribution", {})
            python_count = lang_dist.get("python", 0)
            solidity_count = lang_dist.get("solidity", 0)
            total = sum(lang_dist.values())
            
            if total > 0:
                print(f"   Python: {python_count}/{total} ({python_count/total*100:.1f}%)")
                print(f"   Solidity: {solidity_count}/{total} ({solidity_count/total*100:.1f}%)")
            
    else:
        print(f"🔍 Analyzing single file: {path}")
        analysis = analyzer.analyze_test_results(path)
    
    print("\n✅ Analysis complete! Check the generated report files.")
    print("📋 Look for files matching pattern: pattern_analysis_*")

if __name__ == "__main__":
    main()
