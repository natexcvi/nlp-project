# NLP Regex Generation Project

This project evaluates different prompting techniques for Large Language Models (LLMs) on the task of generating correct regular expressions (regex) based on natural language descriptions.

## Project Overview

This research project compares the effectiveness of various prompting strategies when using LLMs to generate regular expressions from natural language descriptions. The project implements and evaluates three main techniques:

1. **Dynamic Few Shot (DyFS)**: Selects relevant examples dynamically based on the problem.
2. **Chain of Thought (CoT)**: Encourages the model to reason step-by-step before generating the final regex.
3. **Self Refine**: Allows the model to iteratively refine its own solutions.

## Setup and Installation

### Prerequisites
- Python 3.10+
- Poetry (for dependency management)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   poetry install
   ```
3. Set up environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Project Structure

- **nlp_project/**: Main project module
  - **clients/**: API clients for different LLM providers
  - **dataset/**: Problem definitions and scoring utilities
    - `regex_problem.py`: Defines regex problems for evaluation
    - `score_utils.py`: Utilities for scoring generated regex solutions
  - **solvers/**: Implementation of different prompting techniques
    - `base_solver.py`: Base solver class
    - `chain_of_thought.py`: Chain of Thought prompting approach
    - `dyfs.py`: Dynamic Few Shot prompting approach
    - `self_refine.py`: Self-refinement prompting approach
  - `experiment.py`: Main experiment runner script

- **data/**: Contains datasets and test cases
- **reports/**: Generated experiment reports

## Running Experiments

To run the full experiment suite:

```bash
python main.py
```

This will:
1. Load all problem definitions
2. Test each solver on the problems
3. Generate detailed reports with metrics

## Experiment Reports

Reports are saved to the `reports/` directory with a timestamp and include:

- **Performance metrics** for each solver
- **Token usage** statistics
- **Generation times**
- **Complete conversation logs** for analysis

## Development

### Adding New Problems

Add new regex problems in `nlp_project/dataset/regex_problem.py` following the existing format.

### Implementing New Solvers

Create a new solver in the `nlp_project/solvers/` directory:
1. Inherit from `base_solver.py`
2. Implement the `solve()` method
3. Add the solver to the experiment in `experiment.py`

## Project Details

### Evaluation Metrics

Solutions are evaluated based on:
- Correctness against test cases
- Token efficiency
- Generation time

### Technologies Used

- **Python**: Core programming language
- **OpenAI API**: For accessing GPT models
- **Google Gemini API**: For accessing Gemini models
- **Pandas**: For data manipulation
- **PyYAML**: For report generation
- **Pydantic**: For data validation and serialization

## License

[Project License Information]

## Authors

AA, RR & NL 