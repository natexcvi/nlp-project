import os
from abc import ABC, abstractmethod

from pydantic import BaseModel

from nlp_project.clients.openai_client import LLMConfig, get_openai_client
from nlp_project.dataset.base_problem import Problem


class Solver(ABC):
    def __init__(self):
        self.llm_config = LLMConfig.from_config_toml()
        self.openai_client = get_openai_client(self.llm_config)

    @abstractmethod
    def solve(self, problem: Problem) -> BaseModel:
        pass
