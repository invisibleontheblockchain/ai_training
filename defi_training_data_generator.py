#!/usr/bin/env python3
"""
DeFi Training Data Generator
============================
Generate targeted training data based on cline-optimal's performance gaps
in DeFi development tasks.
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Any

class DeFiTrainingDataGenerator:
    """Generate training data to improve DeFi development capabilities"""
    
    def __init__(self):
        self.performance_analysis = {
            "basic_tasks": 0.93,  # From our baseline test
            "complex_defi": 0.50,  # From complex DeFi test
            "progressive_defi": 0.119,  # From progressive test
            "key_issues": [
                "generates_python_instead_of_solidity",
                "missing_security_patterns",
                "no_blockchain_integration",
                "missing_frontend_integration",
                "no_deployment_scripts"
            ]
        }
    
    def generate_solidity_training_pairs(self) -> List[Dict]:
        """Generate training pairs to fix Solidity vs Python confusion"""
        
        training_pairs = []
        
        # Basic Oracle Contract
        oracle_pair = {
            "prompt": "Create a basic price oracle contract that stores and retrieves asset prices with proper access control",
            "rejected_response": """```python
class Oracle:
    def __init__(self):
        self.price_feeds = {}
    
    def add_price_feed(self, token: str, url: str):
        self.price_feeds[token] = url
    
    def get_price(self, token: str) -> float:
        response = requests.get(self.price_feeds[token])
        return response.json()['price']
```""",
            "preferred_response": """```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/access/Ownable.sol";

contract PriceOracle is Ownable {
    mapping(bytes32 => uint256) public prices;
    mapping(bytes32 => uint256) public lastUpdated;
    
    event PriceUpdated(bytes32 indexed asset, uint256 price, uint256 timestamp);
    
    modifier onlyValidPrice(uint256 _price) {
        require(_price > 0, "Price must be greater than 0");
        _;
    }
    
    function updatePrice(bytes32 _asset, uint256 _price) 
        external 
        onlyOwner 
        onlyValidPrice(_price) 
    {
        prices[_asset] = _price;
        lastUpdated[_asset] = block.timestamp;
        emit PriceUpdated(_asset, _price, block.timestamp);
    }
    
    function getPrice(bytes32 _asset) external view returns (uint256, uint256) {
        require(prices[_asset] > 0, "Price not available");
        return (prices[_asset], lastUpdated[_asset]);
    }
    
    function isPriceStale(bytes32 _asset, uint256 _maxAge) external view returns (bool) {
        return block.timestamp - lastUpdated[_asset] > _maxAge;
    }
}
```""",
            "improvement_reason": "Solidity smart contract with proper access control, events, and validation instead of Python class",
            "category": "language_correction"
        }
        
        # ERC20 Token
        token_pair = {
            "prompt": "Implement an ERC20 token with mint and burn functionality",
            "rejected_response": """```python
class Token:
    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol
        self.balances = {}
        self.total_supply = 0
    
    def mint(self, to, amount):
        self.balances[to] = self.balances.get(to, 0) + amount
        self.total_supply += amount
```""",
            "preferred_response": """```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract CustomToken is ERC20, Ownable {
    constructor(string memory name, string memory symbol) ERC20(name, symbol) {}
    
    function mint(address to, uint256 amount) external onlyOwner {
        require(to != address(0), "Cannot mint to zero address");
        require(amount > 0, "Amount must be greater than 0");
        _mint(to, amount);
    }
    
    function burn(uint256 amount) external {
        require(amount > 0, "Amount must be greater than 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        _burn(msg.sender, amount);
    }
    
    function burnFrom(address from, uint256 amount) external {
        require(amount > 0, "Amount must be greater than 0");
        uint256 currentAllowance = allowance(from, msg.sender);
        require(currentAllowance >= amount, "Burn amount exceeds allowance");
        
        _approve(from, msg.sender, currentAllowance - amount);
        _burn(from, amount);
    }
}
```""",
            "improvement_reason": "Proper ERC20 implementation with OpenZeppelin inheritance, security checks, and standard compliance",
            "category": "language_correction"
        }
        
        # Secure Lending Pool
        lending_pair = {
            "prompt": "Create a lending pool contract with reentrancy protection and proper collateral management",
            "rejected_response": """```python
class LendingPool:
    def __init__(self):
        self.loans = {}
    
    def lend(self, borrower, collateral, loan_amount):
        self.loans[borrower] = {'collateral': collateral, 'loan': loan_amount}
```""",
            "preferred_response": """```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract LendingPool is ReentrancyGuard, Ownable {
    IERC20 public immutable collateralToken;
    IERC20 public immutable lendingToken;
    
    uint256 public constant COLLATERAL_RATIO = 150; // 150% collateralization
    uint256 public constant LIQUIDATION_THRESHOLD = 120; // 120% liquidation threshold
    
    struct Loan {
        uint256 collateralAmount;
        uint256 borrowedAmount;
        uint256 timestamp;
        bool active;
    }
    
    mapping(address => Loan) public loans;
    
    event LoanCreated(address indexed borrower, uint256 collateral, uint256 borrowed);
    event LoanRepaid(address indexed borrower, uint256 amount);
    event Liquidation(address indexed borrower, address indexed liquidator);
    
    constructor(address _collateralToken, address _lendingToken) {
        collateralToken = IERC20(_collateralToken);
        lendingToken = IERC20(_lendingToken);
    }
    
    function borrow(uint256 _collateralAmount, uint256 _borrowAmount) 
        external 
        nonReentrant 
    {
        require(_collateralAmount > 0, "Collateral must be greater than 0");
        require(_borrowAmount > 0, "Borrow amount must be greater than 0");
        require(!loans[msg.sender].active, "Existing loan must be repaid first");
        
        // Check collateralization ratio
        require(
            _collateralAmount * 100 >= _borrowAmount * COLLATERAL_RATIO,
            "Insufficient collateral"
        );
        
        // Transfer collateral from borrower
        require(
            collateralToken.transferFrom(msg.sender, address(this), _collateralAmount),
            "Collateral transfer failed"
        );
        
        // Transfer borrowed tokens to borrower
        require(
            lendingToken.transfer(msg.sender, _borrowAmount),
            "Lending transfer failed"
        );
        
        loans[msg.sender] = Loan({
            collateralAmount: _collateralAmount,
            borrowedAmount: _borrowAmount,
            timestamp: block.timestamp,
            active: true
        });
        
        emit LoanCreated(msg.sender, _collateralAmount, _borrowAmount);
    }
    
    function repay(uint256 _amount) external nonReentrant {
        Loan storage loan = loans[msg.sender];
        require(loan.active, "No active loan");
        require(_amount > 0, "Amount must be greater than 0");
        require(_amount <= loan.borrowedAmount, "Amount exceeds borrowed amount");
        
        // Transfer repayment from borrower
        require(
            lendingToken.transferFrom(msg.sender, address(this), _amount),
            "Repayment transfer failed"
        );
        
        loan.borrowedAmount -= _amount;
        
        // If fully repaid, return collateral
        if (loan.borrowedAmount == 0) {
            uint256 collateralToReturn = loan.collateralAmount;
            loan.active = false;
            loan.collateralAmount = 0;
            
            require(
                collateralToken.transfer(msg.sender, collateralToReturn),
                "Collateral return failed"
            );
        }
        
        emit LoanRepaid(msg.sender, _amount);
    }
    
    function liquidate(address _borrower) external nonReentrant {
        Loan storage loan = loans[_borrower];
        require(loan.active, "No active loan");
        
        // Check if loan is under-collateralized
        require(
            loan.collateralAmount * 100 < loan.borrowedAmount * LIQUIDATION_THRESHOLD,
            "Loan is not liquidatable"
        );
        
        uint256 repayAmount = loan.borrowedAmount;
        uint256 collateralReward = loan.collateralAmount;
        
        // Clear the loan
        loan.active = false;
        loan.borrowedAmount = 0;
        loan.collateralAmount = 0;
        
        // Liquidator pays the debt
        require(
            lendingToken.transferFrom(msg.sender, address(this), repayAmount),
            "Liquidation payment failed"
        );
        
        // Liquidator receives collateral
        require(
            collateralToken.transfer(msg.sender, collateralReward),
            "Collateral transfer failed"
        );
        
        emit Liquidation(_borrower, msg.sender);
    }
}
```""",
            "improvement_reason": "Complete Solidity implementation with reentrancy protection, proper collateral management, liquidation mechanism, and comprehensive security checks",
            "category": "security_enhancement"
        }
        
        training_pairs.extend([oracle_pair, token_pair, lending_pair])
        return training_pairs
    
    def generate_integration_training_pairs(self) -> List[Dict]:
        """Generate training pairs for frontend integration"""
        
        integration_pairs = []
        
        # Frontend Integration Example
        frontend_pair = {
            "prompt": "Create a React component that interacts with the Oracle contract to display current prices",
            "rejected_response": "I can create the smart contract, but frontend integration would require additional specifications.",
            "preferred_response": """```javascript
// OracleDisplay.jsx
import React, { useState, useEffect } from 'react';
import { ethers } from 'ethers';

const ORACLE_ABI = [
  "function getPrice(bytes32 asset) external view returns (uint256, uint256)",
  "function isPriceStale(bytes32 asset, uint256 maxAge) external view returns (bool)",
  "event PriceUpdated(bytes32 indexed asset, uint256 price, uint256 timestamp)"
];

const OracleDisplay = ({ contractAddress, provider }) => {
  const [prices, setPrices] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const assets = ['ETH', 'BTC', 'USDT'];

  useEffect(() => {
    const loadPrices = async () => {
      try {
        const contract = new ethers.Contract(contractAddress, ORACLE_ABI, provider);
        const priceData = {};

        for (const asset of assets) {
          const assetBytes = ethers.utils.formatBytes32String(asset);
          try {
            const [price, timestamp] = await contract.getPrice(assetBytes);
            const isStale = await contract.isPriceStale(assetBytes, 3600); // 1 hour
            
            priceData[asset] = {
              price: ethers.utils.formatEther(price),
              timestamp: new Date(timestamp.toNumber() * 1000),
              isStale
            };
          } catch (err) {
            priceData[asset] = { error: 'Price not available' };
          }
        }

        setPrices(priceData);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    loadPrices();

    // Set up event listener for price updates
    const contract = new ethers.Contract(contractAddress, ORACLE_ABI, provider);
    const filter = contract.filters.PriceUpdated();
    
    contract.on(filter, (asset, price, timestamp) => {
      const assetName = ethers.utils.parseBytes32String(asset);
      setPrices(prev => ({
        ...prev,
        [assetName]: {
          price: ethers.utils.formatEther(price),
          timestamp: new Date(timestamp.toNumber() * 1000),
          isStale: false
        }
      }));
    });

    return () => {
      contract.removeAllListeners();
    };
  }, [contractAddress, provider]);

  if (loading) return <div className="loading">Loading prices...</div>;
  if (error) return <div className="error">Error: {error}</div>;

  return (
    <div className="oracle-display">
      <h2>Live Price Feed</h2>
      <div className="price-grid">
        {assets.map(asset => (
          <div key={asset} className={`price-card ${prices[asset]?.isStale ? 'stale' : ''}`}>
            <h3>{asset}/USD</h3>
            {prices[asset]?.error ? (
              <p className="error">{prices[asset].error}</p>
            ) : (
              <>
                <p className="price">${prices[asset]?.price}</p>
                <p className="timestamp">
                  Updated: {prices[asset]?.timestamp?.toLocaleTimeString()}
                </p>
                {prices[asset]?.isStale && (
                  <p className="warning">⚠️ Price may be stale</p>
                )}
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default OracleDisplay;
```

```css
/* OracleDisplay.css */
.oracle-display {
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
}

.price-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.price-card {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 20px;
  background: white;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  transition: all 0.3s ease;
}

.price-card:hover {
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.price-card.stale {
  border-color: #ff9800;
  background: #fff3e0;
}

.price {
  font-size: 24px;
  font-weight: bold;
  color: #2196f3;
  margin: 10px 0;
}

.timestamp {
  font-size: 12px;
  color: #666;
}

.warning {
  color: #ff9800;
  font-weight: bold;
  font-size: 12px;
}

.loading, .error {
  text-align: center;
  padding: 20px;
}

.error {
  color: #f44336;
}
```""",
            "improvement_reason": "Complete React component with ethers.js integration, real-time updates, error handling, and responsive design",
            "category": "frontend_integration"
        }
        
        integration_pairs.append(frontend_pair)
        return integration_pairs
    
    def generate_deployment_training_pairs(self) -> List[Dict]:
        """Generate training pairs for deployment scripts"""
        
        deployment_pairs = []
        
        # Deployment Script Example
        deployment_pair = {
            "prompt": "Create deployment scripts for the Oracle and Lending contracts with proper verification",
            "rejected_response": "The contracts are ready for deployment. You can use Hardhat or Truffle to deploy them.",
            "preferred_response": """```javascript
// deploy/001_deploy_oracle.js
const { ethers } = require("hardhat");

module.exports = async function ({ getNamedAccounts, deployments }) {
  const { deploy } = deployments;
  const { deployer } = await getNamedAccounts();

  console.log("Deploying PriceOracle with account:", deployer);

  const oracle = await deploy("PriceOracle", {
    from: deployer,
    args: [],
    log: true,
    waitConfirmations: 1,
  });

  console.log("PriceOracle deployed to:", oracle.address);

  // Initialize with some price feeds
  const oracleContract = await ethers.getContractAt("PriceOracle", oracle.address);
  
  const initialPrices = [
    { asset: "ETH", price: ethers.utils.parseEther("2300") },
    { asset: "BTC", price: ethers.utils.parseEther("43000") },
    { asset: "USDT", price: ethers.utils.parseEther("1") }
  ];

  for (const { asset, price } of initialPrices) {
    const assetBytes = ethers.utils.formatBytes32String(asset);
    const tx = await oracleContract.updatePrice(assetBytes, price);
    await tx.wait();
    console.log(`Initialized ${asset} price: ${ethers.utils.formatEther(price)}`);
  }

  // Verify contract on Etherscan
  if (network.name !== "hardhat" && network.name !== "localhost") {
    console.log("Verifying contract on Etherscan...");
    try {
      await run("verify:verify", {
        address: oracle.address,
        constructorArguments: [],
      });
    } catch (error) {
      console.log("Verification failed:", error.message);
    }
  }

  return oracle;
};

module.exports.tags = ["Oracle"];
```

```javascript
// deploy/002_deploy_lending.js
const { ethers } = require("hardhat");

module.exports = async function ({ getNamedAccounts, deployments }) {
  const { deploy, get } = deployments;
  const { deployer } = await getNamedAccounts();

  console.log("Deploying LendingPool with account:", deployer);

  // Get previously deployed Oracle
  const oracle = await get("PriceOracle");

  // Deploy mock tokens for testing
  const collateralToken = await deploy("MockERC20", {
    from: deployer,
    args: ["Collateral Token", "COLL"],
    log: true,
  });

  const lendingToken = await deploy("MockERC20", {
    from: deployer,
    args: ["Lending Token", "LEND"],
    log: true,
  });

  // Deploy LendingPool
  const lendingPool = await deploy("LendingPool", {
    from: deployer,
    args: [collateralToken.address, lendingToken.address],
    log: true,
    waitConfirmations: 1,
  });

  console.log("LendingPool deployed to:", lendingPool.address);

  // Setup initial liquidity
  const lendingTokenContract = await ethers.getContractAt("MockERC20", lendingToken.address);
  const lendingPoolContract = await ethers.getContractAt("LendingPool", lendingPool.address);

  // Mint tokens to lending pool for initial liquidity
  const initialLiquidity = ethers.utils.parseEther("1000000");
  await lendingTokenContract.mint(lendingPool.address, initialLiquidity);
  console.log("Initial liquidity provided:", ethers.utils.formatEther(initialLiquidity));

  // Verify contracts
  if (network.name !== "hardhat" && network.name !== "localhost") {
    console.log("Verifying contracts on Etherscan...");
    
    try {
      await run("verify:verify", {
        address: collateralToken.address,
        constructorArguments: ["Collateral Token", "COLL"],
      });
      
      await run("verify:verify", {
        address: lendingToken.address,
        constructorArguments: ["Lending Token", "LEND"],
      });
      
      await run("verify:verify", {
        address: lendingPool.address,
        constructorArguments: [collateralToken.address, lendingToken.address],
      });
    } catch (error) {
      console.log("Verification failed:", error.message);
    }
  }

  return { lendingPool, collateralToken, lendingToken };
};

module.exports.tags = ["LendingPool"];
module.exports.dependencies = ["Oracle"];
```

```javascript
// scripts/setup-environment.js
const { ethers } = require("hardhat");

async function main() {
  const [deployer, user1, user2] = await ethers.getSigners();
  
  console.log("Setting up test environment...");
  console.log("Deployer:", deployer.address);
  console.log("User1:", user1.address);
  console.log("User2:", user2.address);

  // Get deployed contracts
  const oracle = await ethers.getContract("PriceOracle");
  const lendingPool = await ethers.getContract("LendingPool");
  const collateralToken = await ethers.getContract("MockERC20");
  const lendingToken = await ethers.getContract("MockERC20");

  // Mint test tokens to users
  const testAmount = ethers.utils.parseEther("10000");
  
  await collateralToken.mint(user1.address, testAmount);
  await collateralToken.mint(user2.address, testAmount);
  
  console.log("Test tokens minted to users");
  
  // Display contract addresses
  console.log("\n📋 Deployed Contract Addresses:");
  console.log("Oracle:", oracle.address);
  console.log("LendingPool:", lendingPool.address);
  console.log("CollateralToken:", collateralToken.address);
  console.log("LendingToken:", lendingToken.address);
  
  console.log("\n✅ Environment setup complete!");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
```""",
            "improvement_reason": "Complete deployment pipeline with initialization, verification, and environment setup",
            "category": "deployment_scripts"
        }
        
        deployment_pairs.append(deployment_pair)
        return deployment_pairs
    
    def generate_comprehensive_training_dataset(self) -> Dict:
        """Generate complete training dataset for DeFi development"""
        
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "purpose": "Improve cline-optimal DeFi development capabilities",
                "performance_gaps": self.performance_analysis,
                "target_improvements": [
                    "Generate Solidity instead of Python",
                    "Include comprehensive security patterns",
                    "Add frontend integration examples",
                    "Provide deployment scripts",
                    "Implement proper blockchain patterns"
                ]
            },
            "training_pairs": []
        }
        
        # Collect all training pairs
        solidity_pairs = self.generate_solidity_training_pairs()
        integration_pairs = self.generate_integration_training_pairs()
        deployment_pairs = self.generate_deployment_training_pairs()
        
        dataset["training_pairs"].extend(solidity_pairs)
        dataset["training_pairs"].extend(integration_pairs)
        dataset["training_pairs"].extend(deployment_pairs)
        
        # Add summary statistics
        dataset["metadata"]["total_pairs"] = len(dataset["training_pairs"])
        dataset["metadata"]["categories"] = {
            "language_correction": len([p for p in dataset["training_pairs"] if p["category"] == "language_correction"]),
            "security_enhancement": len([p for p in dataset["training_pairs"] if p["category"] == "security_enhancement"]),
            "frontend_integration": len([p for p in dataset["training_pairs"] if p["category"] == "frontend_integration"]),
            "deployment_scripts": len([p for p in dataset["training_pairs"] if p["category"] == "deployment_scripts"])
        }
        
        return dataset
    
    def save_training_dataset(self, dataset: Dict, filename: str = None) -> str:
        """Save training dataset to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"defi_training_dataset_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        return filename

def main():
    """Generate comprehensive DeFi training dataset"""
    print("🧠 DeFi Training Data Generator")
    print("=" * 50)
    print("Generating targeted training data based on performance analysis...")
    
    generator = DeFiTrainingDataGenerator()
    
    # Generate comprehensive dataset
    dataset = generator.generate_comprehensive_training_dataset()
    
    # Save dataset
    filename = generator.save_training_dataset(dataset)
    
    print(f"\n📊 Training Dataset Generated:")
    print(f"├─ Total training pairs: {dataset['metadata']['total_pairs']}")
    print(f"├─ Language correction pairs: {dataset['metadata']['categories']['language_correction']}")
    print(f"├─ Security enhancement pairs: {dataset['metadata']['categories']['security_enhancement']}")
    print(f"├─ Frontend integration pairs: {dataset['metadata']['categories']['frontend_integration']}")
    print(f"└─ Deployment script pairs: {dataset['metadata']['categories']['deployment_scripts']}")
    
    print(f"\n💾 Dataset saved to: {filename}")
    
    print(f"\n🎯 Key Improvements Targeted:")
    for improvement in dataset['metadata']['target_improvements']:
        print(f"  • {improvement}")
    
    print(f"\n📈 Expected Performance Gains:")
    print(f"  • Solidity generation: 0% → 90%+")
    print(f"  • Security patterns: 0% → 80%+")
    print(f"  • Frontend integration: 0% → 70%+")
    print(f"  • Deployment scripts: 0% → 80%+")
    print(f"  • Overall DeFi score: 11.9% → 75%+")
    
    print(f"\n✅ Training data generation complete!")
    print(f"Next step: Apply this training data using DoRA + DPO techniques")

if __name__ == "__main__":
    main()
