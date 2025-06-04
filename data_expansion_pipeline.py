#!/usr/bin/env python3
"""
Advanced Data Collection Pipeline for AI Training Expansion
Integrates multiple data sources for comprehensive model training
"""

import os
import json
import requests
from datasets import Dataset, concatenate_datasets, load_dataset
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExpansionPipeline:
    def __init__(self, base_dataset_path: str = "datasets/processed/ghosteam_training"):
        self.base_dataset_path = base_dataset_path
        self.expanded_data_path = "datasets/expanded_training"
        self.sources = {
            "code_repos": [],
            "documentation": [],
            "conversations": [],
            "domain_specific": []
        }
        
    def collect_code_repositories(self, repo_list: List[str]) -> Dataset:
        """Collect code examples from GitHub repositories"""
        logger.info("🔍 Collecting code repository data...")
        
        # High-quality code repositories for training
        default_repos = [
            "microsoft/TypeScript",
            "facebook/react", 
            "python/cpython",
            "pytorch/pytorch",
            "huggingface/transformers"
        ]
        
        repos_to_process = repo_list or default_repos
        code_examples = []
        
        for repo in repos_to_process:
            try:
                # This would integrate with GitHub API
                # For now, using placeholder structure
                code_examples.append({
                    "instruction": f"Analyze and explain this code from {repo}",
                    "input": f"Repository: {repo}",
                    "output": f"Code analysis and explanation for {repo} patterns"
                })
            except Exception as e:
                logger.warning(f"Failed to process {repo}: {e}")
        
        return Dataset.from_list(code_examples)
    
    def collect_technical_documentation(self) -> Dataset:
        """Collect technical documentation for knowledge enhancement"""
        logger.info("📚 Collecting technical documentation...")
        
        # Use high-quality instruction datasets
        doc_datasets = [
            ("microsoft/DialoGPT-medium", "conversational"),
            ("databricks/databricks-dolly-15k", "instruction_following"),
            ("OpenAssistant/oasst1", "multi_turn_conversation")
        ]
        
        combined_docs = []
        for dataset_name, category in doc_datasets:
            try:
                # Load subset for memory efficiency
                dataset = load_dataset(dataset_name, split="train[:1000]")
                
                for item in dataset:
                    # Standardize format
                    if "prompt" in item and "response" in item:
                        combined_docs.append({
                            "instruction": item["prompt"],
                            "input": "",
                            "output": item["response"]
                        })
                    elif "instruction" in item and "output" in item:
                        combined_docs.append({
                            "instruction": item["instruction"],
                            "input": item.get("input", ""),
                            "output": item["output"]
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
        
        return Dataset.from_list(combined_docs)
    
    def collect_conversation_data(self) -> Dataset:
        """Collect multi-turn conversation data for dialogue improvement"""
        logger.info("💬 Collecting conversation data...")
        
        # Multi-turn conversation examples
        conversations = [
            {
                "instruction": "Help debug this Python error and explain the solution",
                "input": "User: I'm getting a KeyError when accessing my dictionary\nAssistant: I'd be happy to help! Can you show me the specific code that's causing the error?",
                "output": "User: Here's the code: data['missing_key']\nAssistant: The KeyError occurs because 'missing_key' doesn't exist in your dictionary. Here are several ways to handle this:\n\n1. Use .get() method: data.get('missing_key', 'default_value')\n2. Check if key exists: if 'missing_key' in data: ...\n3. Use try/except: try: value = data['missing_key'] except KeyError: ...\n\nWhich approach would work best for your use case?"
            },
            {
                "instruction": "Provide step-by-step guidance for a complex programming task",
                "input": "User: I need to build a web scraper that handles JavaScript-rendered content",
                "output": "I'll guide you through building a robust web scraper:\n\n**Step 1: Choose the right tools**\n- Use Selenium for JavaScript-heavy sites\n- BeautifulSoup for static content\n- Requests for API calls\n\n**Step 2: Set up the environment**\n```python\nfrom selenium import webdriver\nfrom bs4 import BeautifulSoup\nimport time\n```\n\n**Step 3: Handle dynamic content**\n- Wait for elements to load\n- Use explicit waits over time.sleep()\n- Handle pop-ups and modals\n\nWould you like me to show you a specific implementation example?"
            }
        ]
        
        return Dataset.from_list(conversations)
    
    def collect_domain_specific_data(self) -> Dict[str, Dataset]:
        """Collect domain-specific datasets for specialized capabilities"""
        logger.info("🎯 Collecting domain-specific data...")
        
        domains = {
            "coding": self._create_coding_examples(),
            "reasoning": self._create_reasoning_examples(),
            "knowledge": self._create_knowledge_examples(),
            "autonomy": self._create_autonomy_examples()
        }
        
        return domains
    
    def _create_coding_examples(self) -> Dataset:
        """Create advanced coding examples"""
        coding_data = [
            {
                "instruction": "Implement a production-ready REST API with error handling",
                "input": "Requirements: FastAPI, async/await, database integration, authentication",
                "output": """Here's a production-ready FastAPI implementation:

```python
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
from typing import Optional

app = FastAPI(title="Production API", version="1.0.0")
security = HTTPBearer()

# Database dependency
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

@app.get("/api/v1/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    token: str = Depends(verify_token)
):
    try:
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except Exception as e:
        logger.error(f"Error retrieving user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

Key features:
- Async/await for performance
- Proper error handling with specific HTTP status codes
- Database session management
- Authentication middleware
- Structured logging"""
            }
        ]
        return Dataset.from_list(coding_data)
    
    def _create_reasoning_examples(self) -> Dataset:
        """Create logical reasoning examples"""
        reasoning_data = [
            {
                "instruction": "Solve this system design problem step by step",
                "input": "Design a distributed cache system that can handle 1M requests per second",
                "output": """I'll design a distributed cache system step by step:

**1. Requirements Analysis**
- 1M RPS = ~1000 requests per millisecond
- Need: Low latency (<1ms), High availability, Scalability
- Assume: 80% reads, 20% writes

**2. High-Level Architecture**
- Redis Cluster (sharded across multiple nodes)
- Consistent hashing for data distribution
- Read replicas for load balancing

**3. Detailed Design**
```
Load Balancer → Cache Proxy → Redis Cluster
                    ↓
               Monitoring & Metrics
```

**4. Key Components**
- **Cache Proxy**: Route requests, handle failover
- **Redis Cluster**: 16 shards, 3 replicas each
- **Consistent Hashing**: Minimize resharding impact

**5. Performance Calculations**
- 1M RPS / 16 shards = 62.5K RPS per shard
- With replicas: ~20K RPS per Redis instance
- Memory: ~100GB total (assuming 100-byte average value)

**6. Fault Tolerance**
- Redis Sentinel for automatic failover
- Circuit breaker pattern in proxy
- Health checks and monitoring

This design provides the required throughput with sub-millisecond latency."""
            }
        ]
        return Dataset.from_list(reasoning_data)
    
    def _create_knowledge_examples(self) -> Dataset:
        """Create knowledge-based examples"""
        knowledge_data = [
            {
                "instruction": "Explain machine learning concepts with practical examples",
                "input": "What's the difference between supervised and unsupervised learning?",
                "output": """**Supervised vs Unsupervised Learning**

**Supervised Learning:**
- Uses labeled training data (input-output pairs)
- Goal: Learn mapping from inputs to outputs
- Examples:
  - Email spam detection (emails → spam/not spam)
  - Image classification (images → cat/dog/bird)
  - Price prediction (house features → price)

**Common Algorithms:**
- Classification: Random Forest, SVM, Neural Networks
- Regression: Linear Regression, Decision Trees

**Unsupervised Learning:**
- Uses unlabeled data (only inputs)
- Goal: Discover hidden patterns or structure
- Examples:
  - Customer segmentation (group similar customers)
  - Anomaly detection (find unusual patterns)
  - Dimensionality reduction (compress data)

**Common Algorithms:**
- Clustering: K-means, DBSCAN
- Association: Apriori algorithm
- Dimensionality Reduction: PCA, t-SNE

**Real-world Example:**
Netflix uses both:
- Supervised: Predict ratings based on user history
- Unsupervised: Group users with similar preferences

**When to use which:**
- Supervised: When you have clear target outcomes
- Unsupervised: When exploring data for insights"""
            }
        ]
        return Dataset.from_list(knowledge_data)
    
    def _create_autonomy_examples(self) -> Dataset:
        """Create examples for autonomous decision making"""
        autonomy_data = [
            {
                "instruction": "Make autonomous decisions without asking clarifying questions",
                "input": "Create a web application for a small business",
                "output": """I'll create a comprehensive web application suitable for most small businesses:

**Application: Small Business Management Suite**

**Core Features Implemented:**
1. **Customer Management**
   - Contact database with search/filter
   - Customer interaction history
   - Basic CRM functionality

2. **Product/Service Management**
   - Inventory tracking
   - Service catalog
   - Pricing management

3. **Sales & Invoicing**
   - Quote generation
   - Invoice creation and tracking
   - Payment status monitoring

4. **Dashboard & Analytics**
   - Revenue overview
   - Customer metrics
   - Popular products/services

**Technology Stack:**
- Frontend: React with Material-UI
- Backend: Node.js/Express
- Database: PostgreSQL
- Authentication: JWT tokens
- Hosting: Docker containers

**Key Files Created:**
```
/frontend (React app with responsive design)
/backend (REST API with business logic)
/database (Schema and migrations)
/deployment (Docker configuration)
```

**Why these choices:**
- Common for small businesses across industries
- Scalable architecture for growth
- Modern, maintainable codebase
- Mobile-responsive design

The application is ready for customization based on specific business needs."""
            }
        ]
        return Dataset.from_list(autonomy_data)
    
    def create_expanded_dataset(self, include_sources: List[str] = None) -> Dataset:
        """Create expanded training dataset with multiple sources"""
        if include_sources is None:
            include_sources = ["code_repos", "documentation", "conversations", "domain_specific"]
        
        logger.info("🚀 Creating expanded dataset...")
        
        # Load base dataset
        try:
            from datasets import load_from_disk
            base_dataset = load_from_disk(self.base_dataset_path)
            datasets_to_combine = [base_dataset["train"]]
            logger.info(f"Loaded base dataset: {len(base_dataset['train'])} examples")
        except Exception as e:
            logger.warning(f"Could not load base dataset: {e}")
            datasets_to_combine = []
        
        # Add new data sources
        if "code_repos" in include_sources:
            code_data = self.collect_code_repositories([])
            datasets_to_combine.append(code_data)
            logger.info(f"Added code repository data: {len(code_data)} examples")
        
        if "documentation" in include_sources:
            doc_data = self.collect_technical_documentation()
            datasets_to_combine.append(doc_data)
            logger.info(f"Added documentation data: {len(doc_data)} examples")
        
        if "conversations" in include_sources:
            conv_data = self.collect_conversation_data()
            datasets_to_combine.append(conv_data)
            logger.info(f"Added conversation data: {len(conv_data)} examples")
        
        if "domain_specific" in include_sources:
            domain_datasets = self.collect_domain_specific_data()
            for domain, dataset in domain_datasets.items():
                datasets_to_combine.append(dataset)
                logger.info(f"Added {domain} data: {len(dataset)} examples")
        
        # Combine all datasets
        if datasets_to_combine:
            combined_dataset = concatenate_datasets(datasets_to_combine)
            logger.info(f"📊 Total expanded dataset: {len(combined_dataset)} examples")
            
            # Save expanded dataset
            os.makedirs(self.expanded_data_path, exist_ok=True)
            combined_dataset.save_to_disk(f"{self.expanded_data_path}/train")
            
            return combined_dataset
        else:
            logger.error("No datasets to combine")
            return None

def main():
    """Main execution function"""
    pipeline = DataExpansionPipeline()
    
    # Create expanded dataset with all sources
    expanded_dataset = pipeline.create_expanded_dataset()
    
    if expanded_dataset:
        print(f"\n✅ Successfully created expanded dataset with {len(expanded_dataset)} examples")
        print(f"📁 Saved to: {pipeline.expanded_data_path}")
        
        # Show sample from expanded dataset
        print("\n📝 Sample from expanded dataset:")
        sample = expanded_dataset[0]
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"Input: {sample['input'][:50]}...")
        print(f"Output: {sample['output'][:150]}...")
    else:
        print("❌ Failed to create expanded dataset")

if __name__ == "__main__":
    main()
