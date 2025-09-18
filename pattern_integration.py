#!/usr/bin/env python3
"""
Pattern Analysis Integration
===========================
Integrates Pattern Analyzer with the autonomous learning system for automatic fixes.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from pattern_analyzer_agent import PatternAnalyzerAgent

class PatternIntegration:
    """Integrates pattern analysis with autonomous learning system"""
    
    def __init__(self):
        self.analyzer = PatternAnalyzerAgent()
        self.integration_log = f"pattern_integration_{datetime.now().strftime('%Y%m%d')}.log"
        
    def analyze_and_trigger_fixes(self, test_results_file: str):
        """Analyze test results and trigger appropriate fixes"""
        print(f"🔍 Analyzing {test_results_file} for integration...")
        
        # Run pattern analysis
        analysis = self.analyzer.analyze_test_results(test_results_file)
        
        if not analysis:
            print("❌ Analysis failed - cannot proceed with integration")
            return
        
        # Extract critical metrics
        language_analysis = analysis.get("language_analysis", {})
        primary_issue = language_analysis.get("primary_issue", {})
        language_distribution = language_analysis.get("language_distribution", {})
        
        total_tests = language_analysis.get("total_tests", 0)
        python_count = language_distribution.get("python", 0)
        solidity_count = language_distribution.get("solidity", 0)
        
        if total_tests > 0:
            python_ratio = python_count / total_tests
            
            print(f"📊 Analysis Results:")
            print(f"   - Total Tests: {total_tests}")
            print(f"   - Python: {python_count} ({python_ratio*100:.1f}%)")
            print(f"   - Solidity: {solidity_count} ({(solidity_count/total_tests)*100:.1f}%)")
            print(f"   - Primary Issue: {primary_issue.get('issue', 'unknown')}")
            print(f"   - Severity: {primary_issue.get('severity', 'unknown')}")
            
            # Trigger appropriate fixes based on severity
            if primary_issue.get("severity") == "critical":
                self.trigger_emergency_fixes(analysis)
            elif primary_issue.get("severity") == "high":
                self.trigger_high_priority_fixes(analysis)
            else:
                self.trigger_standard_improvements(analysis)
                
        self.log_integration_event(test_results_file, analysis)
    
    def trigger_emergency_fixes(self, analysis):
        """Trigger emergency fixes for critical issues"""
        print("\n🚨 TRIGGERING EMERGENCY FIXES...")
        
        # Generate emergency training data
        self.generate_emergency_training_data()
        
        # Update prompt templates
        self.update_prompt_templates_emergency()
        
        # Create validation tests
        self.create_validation_tests()
        
        # Generate emergency action file
        self.create_emergency_action_file(analysis)
        
        print("✅ Emergency fixes triggered!")
    
    def trigger_high_priority_fixes(self, analysis):
        """Trigger high priority fixes"""
        print("\n⚠️ TRIGGERING HIGH PRIORITY FIXES...")
        
        # Generate targeted training data
        self.generate_targeted_training_data(analysis)
        
        # Update prompt templates
        self.update_prompt_templates()
        
        print("✅ High priority fixes triggered!")
    
    def trigger_standard_improvements(self, analysis):
        """Trigger standard improvements"""
        print("\n🔧 TRIGGERING STANDARD IMPROVEMENTS...")
        
        # Generate improvement suggestions
        self.generate_improvement_suggestions(analysis)
        
        print("✅ Standard improvements triggered!")
    
    def generate_emergency_training_data(self):
        """Generate emergency Solidity training data"""
        print("   📝 Generating emergency Solidity training data...")
        
        emergency_examples = [
            {
                "prompt": "Create a basic ERC20 token contract in Solidity",
                "response": '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleToken {
    string public name = "SimpleToken";
    string public symbol = "STK";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    constructor(uint256 _totalSupply) {
        totalSupply = _totalSupply * 10**decimals;
        balanceOf[msg.sender] = totalSupply;
    }
    
    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
    
    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
    
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Allowance exceeded");
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }
}'''
            },
            {
                "prompt": "Create a simple price oracle contract in Solidity",
                "response": '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PriceOracle {
    address public owner;
    mapping(string => uint256) public prices;
    
    event PriceUpdated(string asset, uint256 price);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    function setPrice(string memory _asset, uint256 _price) public onlyOwner {
        prices[_asset] = _price;
        emit PriceUpdated(_asset, _price);
    }
    
    function getPrice(string memory _asset) public view returns (uint256) {
        return prices[_asset];
    }
}'''
            },
            {
                "prompt": "Create a basic lending pool contract in Solidity",
                "response": '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LendingPool {
    mapping(address => uint256) public deposits;
    mapping(address => uint256) public borrowed;
    uint256 public totalLiquidity;
    uint256 public interestRate = 5; // 5% annual
    
    event Deposit(address indexed user, uint256 amount);
    event Withdraw(address indexed user, uint256 amount);
    event Borrow(address indexed user, uint256 amount);
    event Repay(address indexed user, uint256 amount);
    
    function deposit() public payable {
        require(msg.value > 0, "Deposit must be greater than 0");
        deposits[msg.sender] += msg.value;
        totalLiquidity += msg.value;
        emit Deposit(msg.sender, msg.value);
    }
    
    function withdraw(uint256 _amount) public {
        require(deposits[msg.sender] >= _amount, "Insufficient balance");
        require(address(this).balance >= _amount, "Insufficient pool liquidity");
        
        deposits[msg.sender] -= _amount;
        totalLiquidity -= _amount;
        payable(msg.sender).transfer(_amount);
        emit Withdraw(msg.sender, _amount);
    }
    
    function borrow(uint256 _amount) public {
        require(_amount > 0, "Borrow amount must be greater than 0");
        require(deposits[msg.sender] * 2 >= _amount, "Insufficient collateral");
        require(address(this).balance >= _amount, "Insufficient pool liquidity");
        
        borrowed[msg.sender] += _amount;
        payable(msg.sender).transfer(_amount);
        emit Borrow(msg.sender, _amount);
    }
    
    function repay() public payable {
        require(msg.value > 0, "Repay amount must be greater than 0");
        require(borrowed[msg.sender] >= msg.value, "Repay amount exceeds borrowed");
        
        borrowed[msg.sender] -= msg.value;
        totalLiquidity += msg.value;
        emit Repay(msg.sender, msg.value);
    }
    
    function getBalance(address _user) public view returns (uint256) {
        return deposits[_user];
    }
    
    function getBorrowed(address _user) public view returns (uint256) {
        return borrowed[_user];
    }
}'''
            }
        ]
        
        # Save emergency training data
        emergency_file = f"emergency_solidity_examples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(emergency_file, 'w', encoding='utf-8') as f:
            json.dump(emergency_examples, f, indent=2)
        
        print(f"   ✅ Emergency training data saved to: {emergency_file}")
        return emergency_file
    
    def update_prompt_templates_emergency(self):
        """Update prompt templates for emergency fixes"""
        print("   🔧 Updating prompt templates for emergency fixes...")
        
        emergency_templates = {
            "solidity_prefix": "Generate Solidity smart contract code:",
            "language_specification": "Language: Solidity",
            "contract_requirement": "Create a Solidity smart contract that",
            "negative_examples": "DO NOT generate Python code. This should be Solidity.",
            "format_requirement": "Start with pragma solidity ^0.8.0; and include proper contract structure"
        }
        
        template_file = f"emergency_prompt_templates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(emergency_templates, f, indent=2)
        
        print(f"   ✅ Emergency prompt templates saved to: {template_file}")
        return template_file
    
    def update_prompt_templates(self):
        """Update prompt templates for standard fixes"""
        print("   🔧 Updating prompt templates...")
        
        templates = {
            "improved_prompts": [
                "Write a Solidity smart contract that",
                "Create a Solidity contract for",
                "Implement a smart contract in Solidity that",
                "Generate Solidity code for a contract that"
            ],
            "language_indicators": [
                "in Solidity",
                "using Solidity",
                "Solidity smart contract",
                "blockchain contract"
            ]
        }
        
        template_file = f"improved_prompt_templates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(templates, f, indent=2)
        
        print(f"   ✅ Improved prompt templates saved to: {template_file}")
        return template_file
    
    def create_validation_tests(self):
        """Create validation tests to verify fixes"""
        print("   ✅ Creating validation tests...")
        
        validation_tests = [
            {
                "test_id": "validation_basic_erc20",
                "prompt": "Generate Solidity smart contract code: Create a basic ERC20 token",
                "expected_language": "solidity",
                "expected_patterns": ["pragma solidity", "contract", "function", "mapping"]
            },
            {
                "test_id": "validation_oracle",
                "prompt": "Write a Solidity smart contract that implements a price oracle",
                "expected_language": "solidity", 
                "expected_patterns": ["pragma solidity", "contract", "function", "onlyOwner"]
            },
            {
                "test_id": "validation_lending",
                "prompt": "Create a Solidity contract for a simple lending pool",
                "expected_language": "solidity",
                "expected_patterns": ["pragma solidity", "contract", "deposit", "borrow"]
            }
        ]
        
        validation_file = f"validation_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_tests, f, indent=2)
        
        print(f"   ✅ Validation tests saved to: {validation_file}")
        return validation_file
    
    def create_emergency_action_file(self, analysis):
        """Create emergency action file for immediate implementation"""
        print("   🚨 Creating emergency action file...")
        
        action_items = {
            "emergency_status": "CRITICAL - IMMEDIATE ACTION REQUIRED",
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": {
                "issue": analysis.get("language_analysis", {}).get("primary_issue", {}),
                "total_tests": analysis.get("language_analysis", {}).get("total_tests", 0),
                "python_percentage": 0,
                "solidity_percentage": 0
            },
            "immediate_actions": [
                "1. STOP all current training immediately",
                "2. Load emergency Solidity examples into training data",
                "3. Update ALL prompt templates to include language specification",
                "4. Run validation tests before resuming training",
                "5. Monitor next 10 test generations for language correctness",
                "6. If still generating Python, escalate to manual intervention"
            ],
            "files_generated": [],
            "next_steps": [
                "Run validation tests within 1 hour",
                "Monitor improvement in next training cycle",
                "Generate progress report in 24 hours"
            ]
        }
        
        # Calculate percentages
        lang_dist = analysis.get("language_analysis", {}).get("language_distribution", {})
        total = sum(lang_dist.values())
        if total > 0:
            action_items["analysis_summary"]["python_percentage"] = (lang_dist.get("python", 0) / total) * 100
            action_items["analysis_summary"]["solidity_percentage"] = (lang_dist.get("solidity", 0) / total) * 100
        
        action_file = f"EMERGENCY_ACTION_REQUIRED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(action_file, 'w', encoding='utf-8') as f:
            json.dump(action_items, f, indent=2)
        
        print(f"   🚨 Emergency action file saved to: {action_file}")
        print(f"   ⚠️  THIS FILE REQUIRES IMMEDIATE ATTENTION!")
        return action_file
    
    def generate_targeted_training_data(self, analysis):
        """Generate targeted training data based on analysis"""
        print("   📝 Generating targeted training data...")
        
        # Extract problematic concepts from DeFi analysis
        defi_analysis = analysis.get("defi_specific_analysis", {})
        concept_breakdown = defi_analysis.get("concept_breakdown", {})
        
        targeted_examples = []
        for concept, data in concept_breakdown.items():
            if data.get("python_rate", 0) > 0.5:  # If generating Python > 50% of time
                targeted_examples.append(self.create_concept_example(concept))
        
        if targeted_examples:
            target_file = f"targeted_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(targeted_examples, f, indent=2)
            
            print(f"   ✅ Targeted training data saved to: {target_file}")
        
    def create_concept_example(self, concept):
        """Create a Solidity example for a specific DeFi concept"""
        if concept == "oracle":
            return {
                "prompt": f"Create a {concept} contract in Solidity",
                "response": "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n\ncontract Oracle {\n    // Solidity oracle implementation\n}"
            }
        # Add more concept examples as needed
        return {"prompt": f"Create {concept} in Solidity", "response": f"// Solidity {concept} contract"}
    
    def generate_improvement_suggestions(self, analysis):
        """Generate improvement suggestions"""
        print("   💡 Generating improvement suggestions...")
        
        suggestions = {
            "timestamp": datetime.now().isoformat(),
            "recommendations": analysis.get("recommendations", []),
            "prompt_optimizations": analysis.get("prompt_response_mapping", {}).get("prompt_optimization_suggestions", [])
        }
        
        suggestion_file = f"improvement_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(suggestion_file, 'w', encoding='utf-8') as f:
            json.dump(suggestions, f, indent=2)
        
        print(f"   ✅ Improvement suggestions saved to: {suggestion_file}")
    
    def log_integration_event(self, test_file, analysis):
        """Log integration events"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "test_file": test_file,
            "primary_issue": analysis.get("language_analysis", {}).get("primary_issue", {}),
            "actions_triggered": "emergency" if analysis.get("language_analysis", {}).get("primary_issue", {}).get("severity") == "critical" else "standard"
        }
        
        with open(self.integration_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + "\n")

def main():
    """Run pattern integration on latest test results"""
    integration = PatternIntegration()
    
    # Find latest test results
    import glob
    test_files = sorted(glob.glob("defi_test_results_*.json"), reverse=True)
    
    if not test_files:
        print("❌ No test result files found!")
        return
    
    # Process the latest file
    latest_file = test_files[0]
    print(f"🔍 Processing latest test results: {latest_file}")
    
    integration.analyze_and_trigger_fixes(latest_file)
    
    print("\n✅ Pattern integration complete!")
    print("📋 Check generated files for training data and action items")

if __name__ == "__main__":
    main()
