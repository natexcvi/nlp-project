from abc import abstractmethod
from typing import Any

from solvers import Solver


class ChainOfThoughtSolver(Solver):
    def __init__(self, openai_api_key: str, system_message: str):
        super().__init__(openai_api_key)
        self.system_message = system_message

    def solve(self, problem: Any) -> str:
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
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
