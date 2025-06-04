# My AI Project

## Overview
This project is designed to utilize AI models for various tasks based on configurations defined in a JSON format. The model comparison results are stored in `config/model_comparison_results.json`, which serves as the foundation for task configuration.

## Project Structure
```
my-ai-project
├── config
│   └── model_comparison_results.json
├── src
│   ├── main.py
│   └── task_config.py
├── tests
│   └── test_task_config.py
├── requirements.txt
└── README.md
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-ai-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the AI application, execute the following command:
```
python src/main.py
```

## Task Configuration
The task configuration logic is defined in `src/task_config.py`, which loads the model comparison results from the JSON file and sets up the necessary tasks for the AI model.

## Testing
Unit tests for the task configuration logic can be found in `tests/test_task_config.py`. To run the tests, use:
```
pytest tests/test_task_config.py
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.