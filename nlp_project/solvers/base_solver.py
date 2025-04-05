from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from nlp_project.clients.openai_client import LLMConfig, get_openai_client
from nlp_project.dataset.base_problem import Problem


class Solver(ABC):
    def __init__(self):
        self.llm_config = LLMConfig.from_config_toml()
        self.openai_client = get_openai_client(self.llm_config)
        self.token_usage = {"input_tokens": 0, "output_tokens": 0}

    @abstractmethod
    def solve(self, problem: Problem) -> Tuple[BaseModel, List[Dict[str, Any]]]:
        """
        Solve the given problem.

        Args:
            problem: The problem to solve

        Returns:
            A tuple containing:
            - The solution to the problem
            - The conversation history with the LLM
        """
        pass
