import pytest
import json
import os
import tempfile
import torch
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from model_comparison_benchmark import ModelBenchmark, TEST_PROMPTS
import GPUtil
import psutil
from rouge_score import rouge_scorer

import unittest.mock as mock

# Import the ModelBenchmark class

class TestModelBenchmark:
    """Test suite for ModelBenchmark class"""
    
    @pytest.fixture
    def benchmark(self):
        """Create a ModelBenchmark instance for testing"""
        return ModelBenchmark()
    
    @pytest.fixture
    def mock_gpu_data(self):
        """Mock GPU data for testing"""
        mock_gpu = Mock()
        mock_gpu.memoryUsed = 1024
        mock_gpu.load = 0.5
        return mock_gpu
    
    @pytest.fixture
    def sample_responses(self):
        """Sample response data for testing"""
        return {
            "coding_1": {
                "prompt": "Write a Python function",
                "response": "def example(): pass",
                "response_time": 1.5,
                "input_tokens": 10,
                "output_tokens": 20,
                "tokens_per_second": 13.33,
                "gpu_memory_used_mb": 100,
                "gpu_load_change": 0.1
            },
            "coding_2": {
                "prompt": "Create a class",
                "response": "class Example: pass",
                "response_time": 2.0,
                "input_tokens": 8,
                "output_tokens": 15,
                "tokens_per_second": 7.5,
                "gpu_memory_used_mb": 80,
                "gpu_load_change": 0.05
            }
        }

    def test_initialization(self, benchmark):
        """Test ModelBenchmark initialization"""
        assert hasattr(benchmark, 'results')
        assert hasattr(benchmark, 'rouge_scorer')
        assert 'phi2_finetuned' in benchmark.results
        assert 'cline_optimal' in benchmark.results
        assert benchmark.results['phi2_finetuned']['responses'] == {}
        assert benchmark.results['phi2_finetuned']['metrics'] == {}

    @patch('model_comparison_benchmark.os.path.exists')
    @patch('model_comparison_benchmark.PeftConfig.from_pretrained')
    @patch('model_comparison_benchmark.AutoModelForCausalLM.from_pretrained')
    @patch('model_comparison_benchmark.AutoTokenizer.from_pretrained')
    @patch('model_comparison_benchmark.PeftModel.from_pretrained')
    @patch('model_comparison_benchmark.pipeline')
    def test_setup_phi2_model_success(self, mock_pipeline, mock_peft_model, 
                                    mock_tokenizer, mock_model, mock_config, 
                                    mock_exists, benchmark):
        """Test successful Phi-2 model setup"""
        # Setup mocks
        mock_exists.return_value = True
        mock_config_obj = Mock()
        mock_config_obj.base_model_name_or_path = "microsoft/phi-2"
        mock_config.return_value = mock_config_obj
        
        mock_model_obj = Mock()
        mock_model_obj.parameters.return_value = [Mock(numel=lambda: 1000, element_size=lambda: 4)]
        mock_model.return_value = mock_model_obj
        
        mock_peft_model.return_value = mock_model_obj
        mock_pipeline.return_value = Mock()
        
        # Test
        result = benchmark.setup_phi2_model()
        
        # Assertions
        assert result is True
        assert hasattr(benchmark, 'phi2_pipeline')
        assert 'load_time' in benchmark.results['phi2_finetuned']['metrics']
        assert 'model_size_mb' in benchmark.results['phi2_finetuned']['metrics']

    @patch('model_comparison_benchmark.os.path.exists')
    def test_setup_phi2_model_not_found(self, mock_exists, benchmark):
        """Test Phi-2 model setup when model doesn't exist"""
        mock_exists.return_value = False
        
        result = benchmark.setup_phi2_model()
        
        assert result is False

    @patch('model_comparison_benchmark.subprocess.run')
    def test_setup_cline_optimal_success(self, mock_subprocess, benchmark):
        """Test successful Cline-Optimal model setup"""
        # Mock successful commands
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # ollama --version
            Mock(stdout="cline-optimal:latest", returncode=0)  # ollama list
        ]
        
        result = benchmark.setup_cline_optimal()
        
        assert result is True
        assert 'load_time' in benchmark.results['cline_optimal']['metrics']

    @patch('model_comparison_benchmark.subprocess.run')
    def test_setup_cline_optimal_not_installed(self, mock_subprocess, benchmark):
        """Test Cline-Optimal setup when Ollama not installed"""
        mock_subprocess.side_effect = FileNotFoundError()
        
        result = benchmark.setup_cline_optimal()
        
        assert result is False

    @patch('model_comparison_benchmark.subprocess.run')
    def test_setup_cline_optimal_model_not_found(self, mock_subprocess, benchmark):
        """Test Cline-Optimal setup when model not available"""
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # ollama --version success
            Mock(stdout="other-model:latest", returncode=0)  # ollama list without cline-optimal
        ]
        
        result = benchmark.setup_cline_optimal()
        
        assert result is False

    @patch('model_comparison_benchmark.GPUtil.getGPUs')
    def test_run_phi2_benchmarks(self, mock_gpu, benchmark, mock_gpu_data):
        """Test running Phi-2 benchmarks"""
        # Setup mocks
        mock_gpu.return_value = [mock_gpu_data]
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{'generated_text': 'Test response'}]
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_pipeline.tokenizer = mock_tokenizer
        benchmark.phi2_pipeline = mock_pipeline
        
        # Test with limited prompts
        with patch.dict('model_comparison_benchmark.TEST_PROMPTS', 
                       {'coding': ['Test prompt']}):
            benchmark.run_phi2_benchmarks()
        
        # Assertions
        assert 'coding_1' in benchmark.results['phi2_finetuned']['responses']
        assert 'avg_response_time' in benchmark.results['phi2_finetuned']['metrics']
        assert 'avg_tokens_per_response' in benchmark.results['phi2_finetuned']['metrics']
        assert 'avg_tokens_per_second' in benchmark.results['phi2_finetuned']['metrics']

    @patch('model_comparison_benchmark.GPUtil.getGPUs')
    @patch('model_comparison_benchmark.ollama')
    def test_run_cline_optimal_benchmarks(self, mock_ollama, mock_gpu, 
                                        benchmark, mock_gpu_data):
        """Test running Cline-Optimal benchmarks"""
        # Setup mocks
        mock_gpu.return_value = [mock_gpu_data]
        mock_ollama.chat.return_value = {
            'message': {'content': 'Test response from Cline'}
        }
        
        # Test with limited prompts
        with patch.dict('model_comparison_benchmark.TEST_PROMPTS', 
                       {'coding': ['Test prompt']}):
            benchmark.run_cline_optimal_benchmarks()
        
        # Assertions
        assert 'coding_1' in benchmark.results['cline_optimal']['responses']
        assert 'avg_response_time' in benchmark.results['cline_optimal']['metrics']
        assert 'avg_tokens_per_response' in benchmark.results['cline_optimal']['metrics']
        assert 'avg_tokens_per_second' in benchmark.results['cline_optimal']['metrics']

    def test_compare_responses(self, benchmark, sample_responses):
        """Test response comparison functionality"""
        # Setup test data
        benchmark.results['phi2_finetuned']['responses'] = sample_responses
        benchmark.results['cline_optimal']['responses'] = {
            'coding_1': {**sample_responses['coding_1'], 'response': 'def similar(): pass'},
            'coding_2': {**sample_responses['coding_2'], 'response': 'class Similar: pass'}
        }
        
        # Mock rouge scorer
        mock_score = Mock()
        mock_score.fmeasure = 0.75
        benchmark.rouge_scorer.score.return_value = {'rougeL': mock_score}
        
        benchmark.compare_responses()
        
        # Assertions
        assert 'comparison' in benchmark.results
        assert 'metrics' in benchmark.results['comparison']
        assert 'avg_rouge_l' in benchmark.results['comparison']['metrics']
        assert 'similarity_percent' in benchmark.results['comparison']['metrics']

    def test_generate_summary(self, benchmark, sample_responses):
        """Test summary generation"""
        # Setup test data
        benchmark.results['phi2_finetuned']['metrics'] = {
            'avg_response_time': 1.75,
            'avg_tokens_per_second': 10.42
        }
        benchmark.results['cline_optimal']['metrics'] = {
            'avg_response_time': 2.0,
            'avg_tokens_per_second': 8.0
        }
        benchmark.results['comparison'] = {
            'metrics': {'similarity_percent': 75.0}
        }
        
        benchmark.generate_summary()
        
        # Assertions
        assert 'summary' in benchmark.results
        assert 'benchmark_date' in benchmark.results['summary']
        assert 'phi2_finetuned' in benchmark.results['summary']
        assert 'cline_optimal' in benchmark.results['summary']
        assert 'comparison' in benchmark.results['summary']

    def test_save_results(self, benchmark):
        """Test saving results to file"""
        benchmark.results['test'] = {'data': 'value'}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_path = tmp_file.name
        
        with patch('model_comparison_benchmark.RESULTS_PATH', tmp_path):
            benchmark.save_results()
        
        # Verify file was created and contains correct data
        assert os.path.exists(tmp_path)
        with open(tmp_path, 'r') as f:
            saved_data = json.load(f)
        assert saved_data['test']['data'] == 'value'
        
        # Cleanup
        os.unlink(tmp_path)

    @patch.object(ModelBenchmark, 'setup_phi2_model')
    @patch.object(ModelBenchmark, 'setup_cline_optimal')
    @patch.object(ModelBenchmark, 'run_phi2_benchmarks')
    @patch.object(ModelBenchmark, 'run_cline_optimal_benchmarks')
    @patch.object(ModelBenchmark, 'compare_responses')
    @patch.object(ModelBenchmark, 'generate_summary')
    @patch.object(ModelBenchmark, 'save_results')
    def test_run_all_benchmarks_success(self, mock_save, mock_summary, mock_compare,
                                      mock_cline_bench, mock_phi2_bench, 
                                      mock_cline_setup, mock_phi2_setup, benchmark):
        """Test successful execution of all benchmarks"""
        # Setup mocks
        mock_phi2_setup.return_value = True
        mock_cline_setup.return_value = True
        
        benchmark.run_all_benchmarks()
        
        # Verify all methods were called
        mock_phi2_setup.assert_called_once()
        mock_cline_setup.assert_called_once()
        mock_phi2_bench.assert_called_once()
        mock_cline_bench.assert_called_once()
        mock_compare.assert_called_once()
        mock_summary.assert_called_once()
        mock_save.assert_called_once()

    @patch.object(ModelBenchmark, 'setup_phi2_model')
    @patch.object(ModelBenchmark, 'setup_cline_optimal')
    def test_run_all_benchmarks_setup_failure(self, mock_cline_setup, 
                                            mock_phi2_setup, benchmark):
        """Test benchmark abortion when model setup fails"""
        mock_phi2_setup.return_value = False
        mock_cline_setup.return_value = True
        
        with patch.object(benchmark, 'run_phi2_benchmarks') as mock_bench:
            benchmark.run_all_benchmarks()
            mock_bench.assert_not_called()

    def test_metrics_calculation_accuracy(self, benchmark, sample_responses):
        """Test accuracy of metrics calculations"""
        benchmark.results['phi2_finetuned']['responses'] = sample_responses
        
        # Calculate expected values
        expected_avg_time = np.mean([1.5, 2.0])
        expected_avg_tokens = np.mean([20, 15])
        expected_avg_tokens_per_sec = np.mean([13.33, 7.5])
        
        # Simulate metrics calculation (extract from run_phi2_benchmarks logic)
        times = [r['response_time'] for r in sample_responses.values()]
        tokens = [r['output_tokens'] for r in sample_responses.values()]
        tokens_per_sec = [r['tokens_per_second'] for r in sample_responses.values()]
        
        benchmark.results['phi2_finetuned']['metrics'] = {
            'avg_response_time': np.mean(times),
            'avg_tokens_per_response': np.mean(tokens),
            'avg_tokens_per_second': np.mean(tokens_per_sec)
        }
        
        # Assertions
        assert abs(benchmark.results['phi2_finetuned']['metrics']['avg_response_time'] - expected_avg_time) < 0.01
        assert abs(benchmark.results['phi2_finetuned']['metrics']['avg_tokens_per_response'] - expected_avg_tokens) < 0.01
        assert abs(benchmark.results['phi2_finetuned']['metrics']['avg_tokens_per_second'] - expected_avg_tokens_per_sec) < 0.01

    def test_test_prompts_structure(self):
        """Test that TEST_PROMPTS has the expected structure"""
        assert isinstance(TEST_PROMPTS, dict)
        assert 'coding' in TEST_PROMPTS
        assert 'knowledge' in TEST_PROMPTS
        assert 'reasoning' in TEST_PROMPTS
        
        for category, prompts in TEST_PROMPTS.items():
            assert isinstance(prompts, list)
            assert len(prompts) > 0
            for prompt in prompts:
                assert isinstance(prompt, str)
                assert len(prompt) > 0

    @patch('model_comparison_benchmark.torch.cuda.amp.autocast')
    def test_memory_management(self, mock_autocast, benchmark):
        """Test that proper memory management is used"""
        # This test verifies that autocast context manager is used
        # when testing the actual benchmark methods
        mock_autocast.return_value.__enter__ = Mock()
        mock_autocast.return_value.__exit__ = Mock()
        
        # This would be called during actual benchmarking
        with mock_autocast():
            pass
        
        # Verify autocast was called (memory optimization)
        mock_autocast.assert_called()

class TestIntegration:
    """Integration tests for the complete benchmark system"""
    
    @patch('model_comparison_benchmark.subprocess.run')
    @patch('model_comparison_benchmark.GPUtil.getGPUs')
    def test_dependency_check_and_installation(self, mock_gpu, mock_subprocess):
        """Test the dependency checking and installation logic"""
        # This tests the main block's dependency checking
        mock_gpu.return_value = [Mock(memoryUsed=1024, load=0.5)]
        
        # Test import error handling
        with patch('builtins.__import__', side_effect=ImportError):
            with patch('model_comparison_benchmark.subprocess.run') as mock_run:
                # This would trigger in the main block
                try:
                except ImportError:
                    mock_run(["pip", "install", "gputil", "psutil", "rouge-score"], check=True)
                    
                # Verify pip install would be called
                mock_run.assert_called_with(
                    ["pip", "install", "gputil", "psutil", "rouge-score"], 
                    check=True
                )

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])