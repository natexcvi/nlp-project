from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from nlp_project.dataset.base_problem import Problem
from nlp_project.solvers.base_solver import Solver


class ChainOfThoughtSolver(Solver):
    def __init__(self, system_message: str):
        super().__init__()
        self.system_message = system_message

    def solve(self, problem: Problem) -> Tuple[BaseModel, List[Dict[str, Any]]]:
        if issubclass(problem.response_format, BaseModel):
            completion_model = self.openai_client.beta.chat.completions.parse
        else:
            completion_model = self.openai_client.chat.completions.create

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": problem.statement},
            {
                "role": "user",
                "content": "Solve the problem step-by-step, reasoning about each step.",
            },
        ]

        response = completion_model(
            model=self.llm_config.model,
            messages=messages,
            response_format=problem.response_format,
        )

        self.token_usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        conversation = messages.copy()
        conversation.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        self.conversation_history = conversation

        if hasattr(response.choices[0].message, "parsed"):
            return response.choices[0].message.parsed, self.conversation_history

        return response.choices[0].message.content, self.conversation_history
