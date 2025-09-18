#!/usr/bin/env python3
"""
Batch Pattern Analysis Runner
============================
Analyzes all DeFi test results and generates comprehensive reports.
"""

import os
import glob
from pathlib import Path
from pattern_analyzer_agent import PatternAnalyzerAgent
import json
from datetime import datetime

def run_batch_analysis():
    """Run pattern analysis on all DeFi test result files"""
    print("🔍 Starting Batch Pattern Analysis...")
    print("=" * 50)
    
    analyzer = PatternAnalyzerAgent()
    
    # Find all DeFi test result files
    test_files = glob.glob("defi_test_results_*.json")
    
    if not test_files:
        print("❌ No DeFi test result files found!")
        print("Looking for files matching pattern: defi_test_results_*.json")
        return
    
    print(f"📁 Found {len(test_files)} test result files:")
    for file in test_files:
        print(f"   - {file}")
    print()
    
    batch_summary = {
        "batch_timestamp": datetime.now().isoformat(),
        "total_files": len(test_files),
        "file_analyses": {},
        "aggregate_stats": {
            "total_tests": 0,
            "total_python_generated": 0,
            "total_solidity_generated": 0,
            "total_mixed_generated": 0,
            "critical_issues": []
        }
    }
    
    # Analyze each file
    for i, test_file in enumerate(test_files, 1):
        print(f"🔍 [{i}/{len(test_files)}] Analyzing: {test_file}")
        
        try:
            analysis = analyzer.analyze_test_results(test_file)
            
            if analysis:
                # Extract key stats
                lang_dist = analysis.get("language_analysis", {}).get("language_distribution", {})
                python_count = lang_dist.get("python", 0)
                solidity_count = lang_dist.get("solidity", 0)
                mixed_count = lang_dist.get("mixed", 0)
                total_tests = sum(lang_dist.values())
                
                # Update aggregate stats
                batch_summary["aggregate_stats"]["total_tests"] += total_tests
                batch_summary["aggregate_stats"]["total_python_generated"] += python_count
                batch_summary["aggregate_stats"]["total_solidity_generated"] += solidity_count
                batch_summary["aggregate_stats"]["total_mixed_generated"] += mixed_count
                
                # Check for critical issues
                primary_issue = analysis.get("language_analysis", {}).get("primary_issue", {})
                if primary_issue.get("severity") == "critical":
                    batch_summary["aggregate_stats"]["critical_issues"].append({
                        "file": test_file,
                        "issue": primary_issue.get("description", "Unknown"),
                        "urgency": primary_issue.get("urgency", "Unknown")
                    })
                
                # Store file-specific results
                batch_summary["file_analyses"][test_file] = {
                    "total_tests": total_tests,
                    "python_count": python_count,
                    "solidity_count": solidity_count,
                    "mixed_count": mixed_count,
                    "python_percentage": (python_count / total_tests * 100) if total_tests > 0 else 0,
                    "solidity_percentage": (solidity_count / total_tests * 100) if total_tests > 0 else 0,
                    "primary_issue": primary_issue.get("issue", "unknown"),
                    "severity": primary_issue.get("severity", "unknown")
                }
                
                print(f"   ✅ Results: {python_count}🐍 {solidity_count}⚡ {mixed_count}🔀 ({total_tests} total)")
                
            else:
                print(f"   ❌ Analysis failed for {test_file}")
                batch_summary["file_analyses"][test_file] = {"error": "Analysis failed"}
                
        except Exception as e:
            print(f"   ❌ Error analyzing {test_file}: {e}")
            batch_summary["file_analyses"][test_file] = {"error": str(e)}
    
    # Generate batch summary report
    print("\n" + "=" * 50)
    print("📊 BATCH ANALYSIS SUMMARY")
    print("=" * 50)
    
    agg = batch_summary["aggregate_stats"]
    total_tests = agg["total_tests"]
    
    if total_tests > 0:
        python_pct = (agg["total_python_generated"] / total_tests) * 100
        solidity_pct = (agg["total_solidity_generated"] / total_tests) * 100
        mixed_pct = (agg["total_mixed_generated"] / total_tests) * 100
        
        print(f"📋 Total Tests Analyzed: {total_tests}")
        print(f"🐍 Python Generated: {agg['total_python_generated']} ({python_pct:.1f}%)")
        print(f"⚡ Solidity Generated: {agg['total_solidity_generated']} ({solidity_pct:.1f}%)")
        print(f"🔀 Mixed Language: {agg['total_mixed_generated']} ({mixed_pct:.1f}%)")
        print()
        
        # Critical issue summary
        if agg["critical_issues"]:
            print("🚨 CRITICAL ISSUES DETECTED:")
            for issue in agg["critical_issues"]:
                print(f"   - {issue['file']}: {issue['issue']}")
        else:
            print("✅ No critical issues detected")
        
        # Overall system status
        print("\n🎯 SYSTEM STATUS:")
        if python_pct > 80:
            print("🚨 CRITICAL: System is fundamentally broken - generates Python instead of Solidity")
            print("   ➤ IMMEDIATE ACTION REQUIRED")
        elif python_pct > 50:
            print("⚠️ WARNING: Major Python bias detected")
            print("   ➤ HIGH PRIORITY FIX NEEDED")
        elif solidity_pct > 70:
            print("✅ GOOD: Mostly generating Solidity")
            print("   ➤ Fine-tuning recommended")
        else:
            print("⚠️ MIXED: Inconsistent language generation")
            print("   ➤ Training consistency needed")
    
    # Save batch summary
    summary_file = f"batch_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    print(f"\n💾 Batch summary saved to: {summary_file}")
    
    # Generate action plan
    generate_action_plan(batch_summary)
    
    print("\n✅ Batch analysis complete!")
    print(f"📁 Check pattern_analysis_*.md files for detailed reports")
    print(f"📊 Check {summary_file} for batch summary")

def generate_action_plan(batch_summary):
    """Generate immediate action plan based on batch analysis"""
    print("\n" + "=" * 50)
    print("🚀 IMMEDIATE ACTION PLAN")
    print("=" * 50)
    
    agg = batch_summary["aggregate_stats"]
    total_tests = agg["total_tests"]
    
    if total_tests == 0:
        print("❌ No test data to analyze")
        return
    
    python_pct = (agg["total_python_generated"] / total_tests) * 100
    
    action_plan = []
    
    if python_pct > 90:
        action_plan.extend([
            "🚨 EMERGENCY INTERVENTION REQUIRED:",
            "1. STOP all current training immediately",
            "2. Inject 200+ Solidity contract examples into training data",
            "3. Modify ALL prompts to include 'Generate Solidity smart contract code:'",
            "4. Use few-shot prompting with 5+ Solidity examples per request",
            "5. Add explicit negative examples: 'DO NOT generate Python code'",
            "6. Test with single contract generation before resuming batch training"
        ])
    elif python_pct > 70:
        action_plan.extend([
            "⚠️ HIGH PRIORITY FIXES:",
            "1. Add 100+ Solidity examples to training data",
            "2. Implement language-specific prompt prefixes",
            "3. Use contrastive examples showing Python vs Solidity",
            "4. Increase Solidity example weight in training"
        ])
    elif python_pct > 30:
        action_plan.extend([
            "🔧 MEDIUM PRIORITY IMPROVEMENTS:",
            "1. Add more Solidity examples to training data",
            "2. Improve prompt clarity with explicit language specification",
            "3. Fine-tune language detection and generation boundaries"
        ])
    else:
        action_plan.extend([
            "✅ MAINTENANCE MODE:",
            "1. Continue monitoring language generation patterns",
            "2. Fine-tune for syntax and compilation improvements",
            "3. Focus on security and gas efficiency optimization"
        ])
    
    for step in action_plan:
        print(step)
    
    # Save action plan
    action_file = f"emergency_action_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(action_file, 'w', encoding='utf-8') as f:
        f.write("EMERGENCY ACTION PLAN\n")
        f.write("Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"SYSTEM STATUS: {python_pct:.1f}% Python Generation\n\n")
        
        for step in action_plan:
            # Remove emoji characters for file compatibility
            clean_step = step.encode('ascii', 'ignore').decode('ascii')
            f.write(clean_step + "\n")
    
    print(f"\n💾 Action plan saved to: {action_file}")

if __name__ == "__main__":
    run_batch_analysis()
