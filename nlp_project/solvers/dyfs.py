from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from nlp_project.dataset.base_problem import Problem
from nlp_project.solvers.base_solver import Solver

MAX_EDGE_CASES = 5


class EdgeCase(BaseModel):
    input: str
    is_match: bool = Field(
        ...,
        description="Whether the input matches the regex.",
    )
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
        self.edge_case_conversation = []

    def __generate_edge_cases(self, problem: Problem) -> list[EdgeCase]:
        edge_case_messages = [
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
        ]

        response = self.openai_client.beta.chat.completions.parse(
            model=self.llm_config.model,
            messages=edge_case_messages,
            response_format=EdgeCases,
        )

        self.edge_case_conversation = edge_case_messages.copy()
        self.edge_case_conversation.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )

        return response.choices[0].message.parsed.edge_cases

    def __stringify_edge_cases(self, edge_cases: list[EdgeCase]) -> str:
        return "\n".join(
            [
                f'{edge_case.input} -> {"matches" if edge_case.is_match else "does not match"}'
                for edge_case in edge_cases
            ]
        )

    def solve(self, problem: Problem) -> Tuple[BaseModel, List[Dict[str, Any]]]:
        if not problem.solution_evaluator:
            raise ValueError(
                "Problem must have a solution evaluator to use this solver."
            )
        if issubclass(problem.response_format, BaseModel):
            completion_model = self.openai_client.beta.chat.completions.parse
        else:
            completion_model = self.openai_client.chat.completions.create

        edge_cases = self.__generate_edge_cases(problem)

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

        self.conversation_history = self.edge_case_conversation.copy()
        conversation = messages.copy()
        conversation.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )

        final_response = (
            response.choices[0].message.parsed
            if hasattr(response.choices[0].message, "parsed")
            else response.choices[0].message.content
        )

        evaluator = problem.solution_evaluator(final_response)

        failing_edge_cases = [
            edge_case
            for edge_case in edge_cases
            if evaluator(edge_case.input) != edge_case.is_match
        ]

        if not failing_edge_cases:
            self.conversation_history.extend(conversation)
            return final_response, self.conversation_history

        edge_case_str = self.__stringify_edge_cases(failing_edge_cases)

        conversation.append(
            {
                "role": "user",
                "content": f"Here are some edge cases that your solution does not handle correctly:\n\n{edge_case_str}",
            }
        )

        response = completion_model(
            model=self.llm_config.model,
            messages=conversation,
            response_format=problem.response_format,
        )

        self.token_usage["input_tokens"] += response.usage.prompt_tokens
        self.token_usage["output_tokens"] += response.usage.completion_tokens

        conversation.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        self.conversation_history.extend(conversation)

        final_response = (
            response.choices[0].message.parsed
            if hasattr(response.choices[0].message, "parsed")
            else response.choices[0].message.content
        )

        return final_response, self.conversation_history
