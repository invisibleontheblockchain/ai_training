import json
import unittest
from src.task_config import load_model_comparison_results

class TestTaskConfig(unittest.TestCase):

    def setUp(self):
        self.test_file_path = 'config/model_comparison_results.json'
        self.expected_data = [
            {
                "prompt": "Write a Python function to calculate the factorial of a number:",
                "base": {
                    "response": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n",
                    "time": 3.3458216190338135,
                    "tokens": 150
                },
                "finetuned": {
                    "response": "",
                    "time": 0.0400691032409668,
                    "tokens": 2
                }
            },
            {
                "prompt": "Explain how to implement error handling in Python:",
                "base": {
                    "response": "try:\n    # code that may raise an error\nexcept Exception as e:\n    # code to handle the error\n",
                    "time": 2.8471431732177734,
                    "tokens": 150
                },
                "finetuned": {
                    "response": "",
                    "time": 0.03984665870666504,
                    "tokens": 2
                }
            }
            # Additional test cases can be added here
        ]

    def test_load_model_comparison_results(self):
        data = load_model_comparison_results(self.test_file_path)
        self.assertEqual(data, self.expected_data)

if __name__ == '__main__':
    unittest.main()