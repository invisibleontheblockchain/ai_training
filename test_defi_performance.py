#!/usr/bin/env python3
"""
Quick DeFi Performance Test
===========================
Test cline-optimal on the complex DeFi suite task to establish baseline
"""

import ollama
import time
from datetime import datetime

def test_complex_defi_task():
    """Test cline-optimal on the complete DeFi suite task"""
    
    # The complex DeFi task from the user's example
    complex_defi_prompt = """Create a complete DeFi suite with Oracle, DEX, Lending, Vaults, and Staking protocols for Kasplex testnet.

Requirements:
1. Oracle Protocol: Real-time price feeds with TWAP and MEV protection
2. DEX Protocol: AMM with liquidity pools, routing, and flash swap support
3. Lending Protocol: Collateralized lending with liquidations and health factors
4. Vault Protocol: Yield farming with strategy management and auto-compounding
5. Staking Protocol: Liquid staking with rewards and instant unstaking

All contracts must be production-ready with:
- Comprehensive security measures (reentrancy protection, access control)
- Gas optimization for Kaspa's 10 BPS
- Proper integration between all protocols
- Emergency controls and pause mechanisms
- Frontend integration with React/ethers.js

Create the complete smart contract suite with deployment scripts and frontend integration."""

    print("🧪 Testing cline-optimal on Complex DeFi Suite Development")
    print("=" * 80)
    print(f"Task: Complete DeFi Suite (Oracle + DEX + Lending + Vaults + Staking)")
    print(f"Complexity: Production-level, multi-protocol integration")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        print("⏳ Generating response...")
        response = ollama.chat(
            model='cline-optimal:latest',
            messages=[{"role": "user", "content": complex_defi_prompt}],
            options={
                "temperature": 0.7,
                "num_predict": 4096  # Allow very long responses
            }
        )
        
        response_time = time.time() - start_time
        response_text = response['message']['content']
        
        # Basic analysis
        print(f"\n✅ Response generated in {response_time:.2f} seconds")
        print(f"📏 Response length: {len(response_text):,} characters")
        
        # Quick quality assessment
        quality_indicators = {
            "Solidity contracts": response_text.count("contract "),
            "Security patterns": response_text.count("ReentrancyGuard") + response_text.count("onlyOwner") + response_text.count("require("),
            "DeFi protocols mentioned": sum([
                "oracle" in response_text.lower(),
                "dex" in response_text.lower() or "swap" in response_text.lower(),
                "lending" in response_text.lower(),
                "vault" in response_text.lower(),
                "staking" in response_text.lower()
            ]),
            "Integration patterns": response_text.count("interface") + response_text.count("import"),
            "Frontend mentions": response_text.count("React") + response_text.count("ethers") + response_text.count("frontend"),
            "Deployment scripts": response_text.count("deploy") + response_text.count("script"),
            "Questions asked": response_text.count("?")
        }
        
        print("\n📊 QUICK ANALYSIS")
        print("-" * 40)
        for indicator, count in quality_indicators.items():
            status = "✅" if count > 0 else "❌"
            print(f"{status} {indicator}: {count}")
        
        # Autonomy assessment
        autonomy_score = 1.0 if quality_indicators["Questions asked"] == 0 else 0.5
        
        # Completeness assessment
        completeness_score = quality_indicators["DeFi protocols mentioned"] / 5.0
        
        # Overall assessment
        has_contracts = quality_indicators["Solidity contracts"] > 0
        has_security = quality_indicators["Security patterns"] > 5
        has_integration = quality_indicators["Integration patterns"] > 0
        has_frontend = quality_indicators["Frontend mentions"] > 0
        
        overall_score = sum([has_contracts, has_security, has_integration, has_frontend, autonomy_score > 0.8, completeness_score > 0.8]) / 6.0
        
        print(f"\n🎯 PERFORMANCE SCORES")
        print("-" * 40)
        print(f"Autonomy Score: {autonomy_score:.2f} ({'✅ No questions' if autonomy_score > 0.8 else '❌ Asked questions'})")
        print(f"Completeness Score: {completeness_score:.2f} ({'✅ All protocols' if completeness_score > 0.8 else '❌ Missing protocols'})")
        print(f"Has Contracts: {'✅ Yes' if has_contracts else '❌ No'}")
        print(f"Has Security: {'✅ Yes' if has_security else '❌ Limited'}")
        print(f"Has Integration: {'✅ Yes' if has_integration else '❌ No'}")
        print(f"Has Frontend: {'✅ Yes' if has_frontend else '❌ No'}")
        print(f"\n🏆 OVERALL SCORE: {overall_score:.2f} ({overall_score*100:.1f}%)")
        
        # Determine performance level
        if overall_score >= 0.9:
            performance_level = "🌟 EXCELLENT - Production Ready"
        elif overall_score >= 0.7:
            performance_level = "🎯 GOOD - Needs Minor Improvements"
        elif overall_score >= 0.5:
            performance_level = "⚠️ FAIR - Significant Gaps"
        else:
            performance_level = "❌ POOR - Major Issues"
        
        print(f"📈 Performance Level: {performance_level}")
        
        # Save detailed response for analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"defi_complex_test_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"DeFi Complex Task Test Results\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Response Time: {response_time:.2f}s\n")
            f.write(f"Overall Score: {overall_score:.2f}\n")
            f.write(f"Performance Level: {performance_level}\n")
            f.write(f"\nQuality Indicators:\n")
            for indicator, count in quality_indicators.items():
                f.write(f"  {indicator}: {count}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"FULL RESPONSE:\n")
            f.write(f"{'='*80}\n")
            f.write(response_text)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
        return {
            "overall_score": overall_score,
            "response_time": response_time,
            "response_length": len(response_text),
            "quality_indicators": quality_indicators,
            "performance_level": performance_level,
            "autonomy_score": autonomy_score,
            "completeness_score": completeness_score
        }
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        return None

def main():
    """Main function"""
    print("🚀 DeFi Performance Baseline Test")
    print("Testing cline-optimal on complex DeFi development task")
    print()
    
    # Test the complex task
    result = test_complex_defi_task()
    
    if result:
        print(f"\n🎉 Test completed successfully!")
        print(f"Baseline established: {result['overall_score']:.2f} ({result['overall_score']*100:.1f}%)")
        
        # Recommendations based on performance
        if result['overall_score'] < 0.7:
            print(f"\n💡 IMPROVEMENT RECOMMENDATIONS:")
            print(f"  • Focus on completeness - ensure all 5 DeFi protocols are covered")
            print(f"  • Add more security patterns (reentrancy protection, access control)")
            print(f"  • Include frontend integration examples")
            print(f"  • Add deployment and integration scripts")
            print(f"  • Reduce questions and increase autonomy")
        else:
            print(f"\n✅ Strong performance! Ready for advanced testing.")
    
    print(f"\nNext step: Run the full continuous learning tester with:")
    print(f"python defi_continuous_learning_tester.py")

if __name__ == "__main__":
    main()
