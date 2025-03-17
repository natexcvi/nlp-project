from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt

from nlp_project.dataset.base_problem import Problem
from nlp_project.dataset.score_utils import RegexResponse
from nlp_project.solvers.base_solver import Solver

from pydantic import BaseModel, Field

from nlp_project.dataset.base_problem import Problem
from nlp_project.solvers.base_solver import Solver

MAX_EDGE_CASES = 5

# 🔹 Initial Prompt (Before Step 1)
INITIAL_PROMPT = """\
You are an expert in generating regex patterns based on textual instructions. Your goal is to produce a regex pattern that matches exactly what the user describes.

Here is the instruction:
{problem_statement}

Instructions:
- Follow a structured multi-step approach.
- First, clearly interpret the instructions and identify key matching criteria.
- Next, list potential edge cases or ambiguities.
- Then, consider possible simplifications or optimizations.
- Finally, generate a clear and efficient regex pattern and validate it against identified test cases.

Do not generate the regex immediately. Start by analyzing the instruction first.
"""

# 🔹 Step 1 - Understanding the Instructions
UNDERSTAND_PROMPT = """\
Carefully analyze the user's instruction and identify:
- Exactly what text patterns need to be matched.
- Key regex components likely required (character classes, quantifiers, anchors, groups, etc.).
- Potential ambiguities or multiple interpretations of the instructions.

Provide a structured breakdown, including a concise summary of the matching criteria.
"""

# 🔹 Step 2 - Identifying Edge Cases
EDGE_CASES_PROMPT = """\
Based on your interpretation of the instruction, list at least **5 edge cases** that might cause issues in matching. Consider:
- Empty or minimal inputs.
- Overlapping or repetitive patterns.
- Unexpected or special characters.
- Case sensitivity or insensitivity.
- Performance pitfalls, like excessive backtracking.

Provide specific example strings for each edge case to help validate the regex.
"""

# 🔹 Step 3 - Reducing and Simplifying the Problem
REDUCE_PROMPT = """\
Now, consider potential simplifications or optimizations that can make the regex more efficient and readable.

Reflect on:
- If the instruction can be rewritten or clarified for simpler regex generation.
- Ways to avoid unnecessary complexity or excessive backtracking.
- Alternative regex techniques or constructs that might enhance clarity and performance.

Clearly describe any simplifications before proceeding.
"""

# 🔹 Step 4 - Generating the Regex Pattern
GENERATE_PROMPT = """\
Generate the regex pattern that fulfills the user's instruction and addresses the identified edge cases. Your regex should:
- Precisely match the intended patterns.
- Be optimized for readability and efficiency.
- Avoid unnecessary complexity and excessive backtracking.

Provide the best possible regex solution, explaining briefly how it works.
"""

# 🔹 Step 5 - Verifying the Solution
VERIFY_PROMPT = """\
Finally, test your generated regex pattern against the sample edge cases identified earlier.

- Clearly report any mismatches or failures, refining the regex as needed.
- If multiple valid patterns exist, choose the most optimal one based on simplicity and efficiency.

Return your final validated regex pattern, along with a concise explanation of why it accurately fulfills the instruction.
"""


class RegexFewShotSolver(Solver):
    def __init__(self, system_message: str):
        super().__init__()
        self.system_message = system_message

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5), reraise=True)
    def solve(self, problem: Problem) -> BaseModel:
        history = [
            {
                "role": "user",
                "content": INITIAL_PROMPT.format(problem_statement=problem.statement)
            }
        ]

        # Define stepwise prompts
        steps = [
            UNDERSTAND_PROMPT,
            EDGE_CASES_PROMPT,
            REDUCE_PROMPT,
            GENERATE_PROMPT,
            VERIFY_PROMPT
        ]
        llm_response = ""
        for step in steps:
            # Append the current step prompt to the conversation
            history.append({"role": "user", "content": step})

            # Send request to OpenAI API
            if step in [GENERATE_PROMPT, VERIFY_PROMPT]:
                response = self.openai_client.beta.chat.completions.parse(
                    model=self.llm_config.model,
                    messages=history,
                    response_format=RegexResponse
                )
                llm_response = response.choices[0].message.parsed

            else:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=history
                )
                llm_response = response.choices[0].message.content

            # Get the LLM's response
            print("\n==== LLM RESPONSE ====\n", llm_response)  # Debugging output

            # Append response to history for context tracking
            history.append({"role": "assistant", "content": llm_response})

        return llm_response


class EdgeCase(BaseModel):
    input: str
    is_match: bool = Field(
        ..., description="Whether the input matches the regex."
    )
    explanation: str = Field(
        ..., description="What aspects of the problem are highlighted by this case."
    )


class EdgeCases(BaseModel):
    edge_cases: list[EdgeCase] = Field(
        ..., description=f"List of up to {MAX_EDGE_CASES} edge cases."
    )


class DynamicRegexSolver(Solver):
    def __init__(self, system_message: str):
        super().__init__()
        self.system_message = system_message

    def generate_step_response(self, prompt: str, problem: Problem, step_description: str):
        response = self.openai_client.chat.completions.create(
            model=self.llm_config.model,
            messages=[
                {"role": "system", "content": step_description},
                {"role": "user", "content": prompt.format(problem_statement=problem.statement)},
            ],
        )
        return response.choices[0].message.content

    def generate_edge_cases(self, problem: Problem) -> list[EdgeCase]:
        response = self.openai_client.beta.chat.completions.parse(
            model=self.llm_config.model,
            messages=[
                {
                    "role": "system",
                    "content": "Identify edge cases for a regex based on the provided instructions.",
                },
                {"role": "user", "content": problem.statement},
            ],
            response_format=EdgeCases,
        )
        return response.choices[0].message.parsed.edge_cases

    def stringify_edge_cases(self, edge_cases: list[EdgeCase]) -> str:
        return "\n".join(
            [
                f'{edge_case.input} -> {"matches" if edge_case.is_match else "does not match"}'
                for edge_case in edge_cases
            ]
        )

    def solve(self, problem: Problem) -> BaseModel:
        history = [{"role": "system", "content": self.system_message}]

        # Step 1: Understand the instructions
        history.append({"role": "user", "content": UNDERSTAND_PROMPT.format(problem_statement=problem.statement)})
        understanding = self.openai_client.chat.completions.create(
            model=self.llm_config.model, messages=history
        ).choices[0].message
        history.append(understanding)

        # Step 2: Identify edge cases
        edge_cases = self.generate_edge_cases(problem)
        edge_case_str = self.stringify_edge_cases(edge_cases)
        history.append({"role": "assistant", "content": edge_case_str})

        # Step 3: Reducing/Simplifying
        history.append({"role": "user", "content": REDUCE_PROMPT})
        simplifications = self.openai_client.chat.completions.create(
            model=self.llm_config.model, messages=history
        ).choices[0].message
        history.append({"role": "assistant", "content": simplifications.content})

        # Step 4: Generate the regex solution
        history.append({"role": "user", "content": GENERATE_PROMPT})
        regex_pattern = self.openai_client.beta.chat.completions.parse(
            model=self.llm_config.model, messages=history, response_format=problem.response_format
        ).choices[0].message.parsed
        history.append({"role": "assistant", "content": regex_pattern.regex})

        # Evaluate against edge cases
        evaluator = problem.solution_evaluator(regex_pattern)

        failing_edge_cases = [
            edge_case
            for edge_case in edge_cases
            if evaluator(edge_case.input) != edge_case.is_match
        ]

        if failing_edge_cases:
            failing_edge_str = self.stringify_edge_cases(failing_edge_cases)
            history.append({
                "role": "user",
                "content": f"{VERIFY_PROMPT}\n\nFailing cases:\n{failing_edge_str}"
            })

            refined_regex_pattern = self.openai_client.beta.chat.completions.parse(
                model=self.llm_config.model, messages=history, response_format=problem.response_format
            ).choices[0].message.parsed

            return refined_regex_pattern

        return regex_pattern
