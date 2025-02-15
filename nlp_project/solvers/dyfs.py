from typing import Any

from pydantic import BaseModel, Field

from nlp_project.dataset.base_problem import Problem
from nlp_project.dataset.score_utils import RegexResponse
from nlp_project.solvers.base_solver import Solver

MAX_EDGE_CASES = 5


class EdgeCase(BaseModel):
    input: str
    output: str
    explanation: str = Field(
        ...,
        description="What aspects of the problem are highlighted by this case.",
    )


class EdgeCases(BaseModel):
    edge_cases: list[EdgeCase] = Field(
        ...,
        description=f"List of up to {MAX_EDGE_CASES} edge cases to help guide the process of solving a problem.",
    )


class DynamicFewShotSolver(Solver):
    def __init__(self, system_message: str):
        super().__init__()
        self.system_message = system_message

    def __generate_edge_cases(self, problem: Problem) -> list[EdgeCase]:
        response = self.openai_client.beta.chat.completions.parse(
            model=self.llm_config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are responsible for finding edge cases to help guide the process of solving a problem in the general case.",
                },
                {
                    "role": "user",
                    "content": f"Here is the problem statement:\n\n{problem.statement}",
                },
                {
                    "role": "user",
                    "content": "Generate edge cases for this problem. Each case should highlight a different aspect of the problem. Remember: the goal is to help guide the process of solving the problem in the general case.",
                },
            ],
            response_format=EdgeCases,
        )
        return response.choices[0].message.parsed.edge_cases

    def solve(self, problem: Problem) -> BaseModel:
        if problem.response_format:
            completion_model = self.openai_client.beta.chat.completions.parse
        else:
            completion_model = self.openai_client.chat.completions.create

        edge_cases = self.__generate_edge_cases(problem)
        response = completion_model(
            model=self.llm_config.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": problem.statement},
                {
                    "role": "user",
                    "content": f"Here are some edge cases to help guide the process of solving the problem in the general case:\n{edge_cases}",
                },
                {
                    "role": "user",
                    "content": "Solve the problem step-by-step, reasoning about each step.",
                },
            ],
            response_format=problem.response_format,
        )
        if hasattr(response.choices[0].message, "parsed"):
            return response.choices[0].message.parsed

        return response.choices[0].message.content
