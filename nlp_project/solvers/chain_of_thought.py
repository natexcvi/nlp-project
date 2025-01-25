from typing import Any

from nlp_project.solvers.base_solver import Solver


class ChainOfThoughtSolver(Solver):
    def __init__(self, system_message: str):
        super().__init__()
        self.system_message = system_message

    def solve(self, problem: Any) -> str:
        response = self.openai_client.chat.completions.create(
            model=self.llm_config.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": problem.statement},
                {
                    "role": "user",
                    "content": "Solve the problem step-by-step, reasoning about each step.",
                },
            ],
        )
        return response.choices[0].message.content
