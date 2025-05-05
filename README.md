# NLP Regex Generation Project

This project evaluates different few-shot learning techniques and advanced prompting strategies for Large Language Models (LLMs) on the task of generating correct regular expressions (regex) based on natural language descriptions. The research compares several approaches and their impact on regex generation performance using both the OpenAI and Google Gemini APIs.

## Project Overview

Regular expressions are notoriously difficult to write correctly, even for experienced programmers. This research project explores how different few-shot learning approaches and LLM prompting techniques can improve regex generation accuracy. The project implements and evaluates three sophisticated approaches:

1. **Dynamic Few Shot (DyFS)**: This technique leverages few-shot learning with dynamically selected examples:
   - Instead of using static examples, it uses an LLM to identify problem-specific edge cases
   - Then it incorporates these examples and feedback on failing cases into the prompt
   - The implementation parallels edge case generation and initial solution creation using concurrent processing
   - This adaptive few-shot approach outperforms static examples by tailoring the demonstration set to each problem

2. **Chain of Thought (CoT)**: This cognitive enhancement prompting technique:
   - Explicitly instructs the model to "solve the problem step-by-step, reasoning about each step"
   - Encourages the LLM to break down the regex problem into logical components before synthesis
   - Helps the model avoid common errors by making its reasoning process explicit
   - Works without examples but can be combined with few-shot learning for further improvements

3. **Self Refine**: This iterative improvement approach allows models to critique and improve their own solutions:
   - Generates an initial solution to the regex problem
   - Uses a separate LLM call to identify specific issues and provide improvement suggestions
   - Presents this feedback to the original model along with the request to generate an improved solution
   - Can perform multiple refinement iterations until no further issues are identified or a maximum iteration count is reached
   - Functions as a form of implicit few-shot learning where the model learns from its own previous attempts

The project framework allows for comparing these approaches individually and in combination to determine optimal strategies for regex generation tasks.


## Project Structure

- **nlp_project/**: Main project module
  - **clients/**: API clients for different LLM providers
  - **dataset/**: Problem definitions and scoring utilities
    - `regex_problem.py`: Defines regex problems and evaluation criteria
    - `score_utils.py`: Utilities for scoring regex solutions against test cases
    - `regex_models.py`: Pydantic models for structured regex responses
    - `gt_generator.py`: Ground truth data generation utilities
  - **solvers/**: Implementation of few-shot techniques and prompting strategies
    - `base_solver.py`: Abstract base solver with shared functionality
    - `chain_of_thought.py`: Chain of Thought prompting implementation
    - `dyfs.py`: Dynamic Few Shot learning with edge case detection
    - `self_refine.py`: Implements the self-refinement feedback loop
  - `experiment.py`: Main experiment runner with metrics collection

- **data/**: Contains regex datasets and test cases
- **reports/**: Generated experiment reports and performance analysis

## Running Experiments

To run the full experiment suite:

```bash
python main.py
```

This will:
1. Load all regex problem definitions from the dataset
2. Test each solver strategy on the problems with multiple iterations
3. Generate detailed reports with performance metrics and token usage statistics
4. Save conversation logs for qualitative analysis

## Experiment Reports

Reports are saved to the `reports/` directory with a timestamp and include:

- **Performance metrics**: Detailed scoring of each solver on different regex problems
- **Token usage**: Input and output token counts for efficiency evaluation
- **Generation times**: Time taken to generate solutions by each approach
- **Complete conversation logs**: Full prompts and responses for qualitative analysis

## Evaluation Methodology

The project employs a rigorous evaluation methodology:
- Each solver is tested on the same set of regex problems
- Multiple iterations per problem to account for variation in LLM outputs
- Test cases include both matching and non-matching strings to ensure comprehensive evaluation
- Metrics include accuracy, token efficiency, and generation time

## Authors

AA, RR & NL 